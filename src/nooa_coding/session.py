"""Formal local AgentSession API with persistence, streaming, and recovery."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import re
import sqlite3
import uuid
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from nooa.interactive import AgentMessage
from nooa.storage import SQLiteStorageManager
from nooa.unifiedllm import UnifiedLLM
from pydantic import BaseModel, Field

from .agent import CodingAgent, CodingTaskResult
from .config import CodingSettings, load_settings
from .events import SessionEvent, SessionEventKind, TokenUsage
from .hooks import HookRunner, load_hooks_config
from .learning import LessonExtractor, LessonStore
from .llm import build_llm
from .mcp import MCPServerStatus
from .multi_agent import Coordinator, MergeResult, TaskPackage, WorkerReport, merge_worktrees
from .policy import ApprovalManager, PermissionPolicy, PolicyShellTools
from .workspace import Checkpoint, DiffResult, WorkspaceInfo, WorkspaceManager

SessionStatus = Literal["idle", "running", "cancelling", "cancelled", "failed"]
_SESSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class GoalState(BaseModel):
    """Tracks an active goal-driven reconcile loop."""

    objective: str
    turn_budget: int = 10
    turns_used: int = 0
    status: Literal["active", "achieved", "exhausted", "cleared"] = "active"
    started_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    last_evaluation: str = ""

    @property
    def turns_remaining(self) -> int:
        return max(0, self.turn_budget - self.turns_used)


class SessionMetadata(BaseModel):
    session_id: str
    base_repo: str
    workspace: WorkspaceInfo
    status: SessionStatus = "idle"
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    parent_session_id: str | None = None
    latest_snapshot_id: str | None = None
    recovered_after_crash: bool = False
    error: str | None = None
    checkpoints: list[Checkpoint] = Field(default_factory=list)


class SessionSummary(BaseModel):
    session_id: str
    status: SessionStatus
    workspace: str
    updated_at: str
    parent_session_id: str | None = None
    checkpoint_count: int = 0
    recovered_after_crash: bool = False


def _repo_key(repo: Path) -> str:
    return hashlib.sha256(str(repo).encode()).hexdigest()[:12]


class AgentSession:
    """One resumable coding conversation bound to one isolated worktree."""

    def __init__(
        self,
        manager: AgentSessionManager,
        metadata: SessionMetadata,
        settings: CodingSettings,
        *,
        llm: UnifiedLLM | None = None,
        restore: bool = False,
    ) -> None:
        self.manager = manager
        self.metadata = metadata
        self.settings = settings
        self.session_dir = manager.session_dir(metadata.session_id)
        self.database_path = self.session_dir / "session.db"
        self.trace_path = self.session_dir / "events.jsonl"
        self.metadata_path = self.session_dir / "metadata.json"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._sequence = self._last_sequence()
        self._event_queue: asyncio.Queue[SessionEvent] = asyncio.Queue()
        self._run_lock = asyncio.Lock()
        self._active_task: asyncio.Task[CodingTaskResult] | None = None
        self._closed = False
        self._session_start_fired = False
        self._token_usage = TokenUsage()
        self._last_prompt_tokens = 0
        self._resolved_context_window: int | None = None
        self._goal: GoalState | None = None
        self._coordinator: Coordinator | None = None
        self._is_sub_agent = False
        self._lesson_store: LessonStore | None = None
        self._lesson_extractor: LessonExtractor | None = None

        hooks_config = load_hooks_config(
            metadata.workspace.path,
            settings_hooks=getattr(settings, "hooks", None),
        )
        self._hook_runner = HookRunner(
            hooks_config,
            workspace=metadata.workspace.path,
            session_id=metadata.session_id,
            event_sink=self._tool_event,
        )
        self.approvals = ApprovalManager(self._policy_event, hook_runner=self._hook_runner)
        policy = PermissionPolicy(settings.permissions, self.approvals)
        shell = PolicyShellTools(
            metadata.workspace.path,
            policy,
            settings.limits,
            self._tool_event,
            hook_runner=self._hook_runner,
        )
        self.llm = llm or build_llm(settings.models, on_failover=self._model_failover)
        self.storage = SQLiteStorageManager(self.database_path)
        self.agent = CodingAgent(
            llm=self.llm,
            repo=metadata.workspace.path,
            settings=settings,
            shell=shell,
            approvals=self.approvals,
            event_sink=self._tool_event,
            storage=self.storage,
        )
        self.agent.event_manager.register_event_type(AgentMessage)
        self._unsubscribe = self.agent.event_manager.on("*", self._on_nooa_event)

        if restore:
            self.storage.restore_latest_snapshot(self.agent)
        self._persist_metadata()

    async def _fire_session_start(self) -> None:
        """Fire SessionStart hooks. Called once after construction."""
        with contextlib.suppress(Exception):
            await self._hook_runner.trigger_session_start()

    @property
    def session_id(self) -> str:
        return self.metadata.session_id

    @property
    def workspace(self) -> Path:
        return Path(self.metadata.workspace.path)

    @property
    def status(self) -> SessionStatus:
        return self.metadata.status

    @property
    def current_model(self) -> str:
        active = getattr(self.llm, "active", self.llm)
        return str(getattr(active, "model", getattr(self.llm, "model", "unknown")))

    @property
    def available_models(self) -> list[str]:
        """List all model names in the failover chain."""
        clients = getattr(self.llm, "clients", None)
        if clients:
            return [str(c.model) for c in clients]
        return [self.current_model]

    def switch_model(self, model_name: str) -> str:
        """Switch the active model at runtime. Returns the new active model name."""
        clients = getattr(self.llm, "clients", None)
        if not clients:
            raise RuntimeError("model switching requires a failover LLM with multiple models")
        for index, client in enumerate(clients):
            if client.model == model_name:
                self.llm.active_index = index  # type: ignore[union-attr]
                self.llm.model = client.model  # type: ignore[union-attr]
                self._emit("session", "model_switched", {"model": model_name})
                return model_name
        available = ", ".join(c.model for c in clients)
        raise ValueError(f"unknown model '{model_name}'. Available: {available}")

    def switch_permissions(self, mode: str) -> None:
        """Switch permission mode at runtime.

        Supported modes:
        - ``yolo`` / ``allow``: auto-approve reads, writes, and shell.
        - ``auto-edit``: auto-approve file reads/writes, but still ask for shell.
        - ``ask``: ask for writes and shell (reads stay allowed).
        - ``default``: restore the configured settings.
        """
        policy = self.agent.shell._policy  # noqa: SLF001
        if mode in ("allow", "yolo"):
            policy.settings = self.settings.permissions.model_copy(
                update={"file_read": "allow", "file_write": "allow", "shell": "allow"}
            )
        elif mode in ("auto-edit", "autoedit", "auto"):
            policy.settings = self.settings.permissions.model_copy(
                update={"file_read": "allow", "file_write": "allow", "shell": "ask"}
            )
        elif mode == "ask":
            policy.settings = self.settings.permissions.model_copy(
                update={"file_write": "ask", "shell": "ask"}
            )
        elif mode == "default":
            policy.settings = self.settings.permissions
        else:
            raise ValueError(
                f"unknown permission mode '{mode}'. Use: yolo, auto-edit, ask, default"
            )
        self._emit("session", "permissions_changed", {"mode": mode})

    async def review(self) -> str:
        """Run an LLM-powered code review on the current worktree diff."""
        diff = WorkspaceManager.diff(self.workspace)
        patch_text = diff.patch.strip()
        if not patch_text:
            return "No changes to review. The worktree is clean."
        review_prompt = (
            "You are a senior code reviewer. Review the following diff and provide:\n"
            "1. A brief summary of what changed\n"
            "2. Potential bugs or logic errors\n"
            "3. Security concerns\n"
            "4. Style/readability suggestions\n\n"
            f"```diff\n{patch_text[:50000]}\n```"
        )
        messages = [{"role": "user", "content": review_prompt}]
        response = await self.llm.acall(messages)
        self._track_direct_llm_call(response)
        self._emit("session", "review_completed", {"files": diff.stat})
        return response.content or "Review produced no output."

    async def plan(self, task: str) -> str:
        """Generate an implementation plan for a task without executing it."""
        plan_prompt = (
            "You are a senior software architect. Create a detailed implementation plan "
            "for the following task. Include:\n"
            "1. Analysis of what needs to change\n"
            "2. Step-by-step implementation plan with file paths\n"
            "3. Potential risks and edge cases\n"
            "4. Suggested verification steps\n\n"
            "Do NOT write any code. Only produce the plan.\n\n"
            f"Task: {task}"
        )
        messages = [{"role": "user", "content": plan_prompt}]
        response = await self.llm.acall(messages)
        self._track_direct_llm_call(response)
        self._emit("session", "plan_generated", {"task": task[:200]})
        return response.content or "Plan produced no output."

    async def suggest(self, task: str) -> str:
        """Generate a concrete diff suggestion without executing any changes."""
        diff = WorkspaceManager.diff(self.workspace)
        context_parts: list[str] = []
        if diff.patch.strip():
            context_parts.append(f"Current uncommitted changes:\n```diff\n{diff.patch[:20000]}\n```")
        suggest_prompt = (
            "You are a code suggestion engine. The user wants the following change.\n"
            "Produce a CONCISE unified diff showing exactly what to modify.\n"
            "Rules:\n"
            "- Output ONLY the diff blocks (```diff ... ```), no prose.\n"
            "- Use real file paths from the repository.\n"
            "- Keep changes minimal and focused.\n"
            "- If you need more context, output a brief question instead.\n\n"
            f"Task: {task}\n\n"
            + "\n".join(context_parts)
        )
        messages = [{"role": "user", "content": suggest_prompt}]
        response = await self.llm.acall(messages)
        self._track_direct_llm_call(response)
        self._emit("session", "suggest_completed", {"task": task[:200]})
        return response.content or "No suggestion produced."

    async def stream_direct(self, prompt: str) -> AsyncIterator[str]:
        """Stream a direct LLM response token-by-token via the active failover client."""
        import litellm

        from .system_prompt import render_runtime_system_context

        active = getattr(self.llm, "active", self.llm)
        model = str(getattr(active, "model", getattr(self.llm, "model", "")))
        api_base = getattr(active, "api_base", None) or getattr(
            getattr(active, "config", {}), "get", lambda *_: None
        )("api_base")
        kwargs: dict[str, Any] = {"model": model, "stream": True}
        if api_base:
            kwargs["api_base"] = api_base
        api_key = getattr(active, "api_key", None)
        if api_key:
            kwargs["api_key"] = api_key

        system_ctx = render_runtime_system_context(
            active_model=model, worktree=self.workspace
        )
        messages = [
            {"role": "system", "content": system_ctx},
            {"role": "user", "content": prompt},
        ]

        response = await litellm.acompletion(messages=messages, **kwargs)
        async for chunk in response:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content

    async def web_search(self, query: str) -> str:
        """Search the web and return summarized results."""
        import urllib.parse
        import urllib.request

        encoded = urllib.parse.quote_plus(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded}"

        def _fetch() -> str:
            req = urllib.request.Request(url, headers={"User-Agent": "nooa-coding/0.1"})
            with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310
                return resp.read().decode("utf-8", errors="replace")

        try:
            html = await asyncio.to_thread(_fetch)
        except Exception as exc:
            self._emit("error", "web_search_failed", {"query": query, "error": str(exc)})
            return f"Web search failed: {exc}"

        # Extract result snippets from DuckDuckGo HTML.
        results: list[str] = []
        for match in re.finditer(
            r'class="result__a"[^>]*>(.*?)</a>.*?class="result__snippet"[^>]*>(.*?)</span>',
            html,
            re.DOTALL,
        ):
            title = re.sub(r"<[^>]+>", "", match.group(1)).strip()
            snippet = re.sub(r"<[^>]+>", "", match.group(2)).strip()
            if title:
                results.append(f"- {title}: {snippet}")
            if len(results) >= 8:
                break

        if not results:
            return f"No results found for: {query}"
        self._emit("session", "web_search_completed", {"query": query, "count": len(results)})
        return f"Web search results for '{query}':\n\n" + "\n".join(results)

    # ─── Goal mode ───────────────────────────────────────────────────────────

    @property
    def coordinator(self) -> Coordinator:
        """Lazy-initialized sub-agent coordinator for this session.

        Only the main (non-sub-agent) session can access this.
        Sub-agents are forbidden from spawning further sub-agents.
        """
        if self._is_sub_agent:
            raise PermissionError("Sub-agents cannot spawn further sub-agents")
        if self._coordinator is None:
            self._coordinator = Coordinator(
                self.manager, self, hook_runner=self._hook_runner
            )
        return self._coordinator

    async def delegate_tasks(
        self, tasks: list[TaskPackage], *, on_progress: Any = None
    ) -> list[WorkerReport]:
        """Delegate tasks to isolated sub-agents and wait for results.

        This is the main entry point for multi-agent delegation.
        The main agent constructs TaskPackages, calls this method,
        and receives structured WorkerReports back.
        """
        coordinator = self.coordinator
        handles = coordinator.spawn(tasks)
        self._emit(
            "session",
            "subagents_spawned",
            {"count": len(handles), "tasks": [t.task_id for t in tasks]},
        )
        reports = await coordinator.wait_all(handles, on_progress=on_progress)
        self._emit(
            "session",
            "subagents_completed",
            {
                "completed": sum(1 for r in reports if r.status == "completed"),
                "failed": sum(1 for r in reports if r.status in ("failed", "timeout")),
            },
        )
        return reports

    async def delegate_preset(
        self, preset_name: str, objective: str, *, on_progress: Any = None
    ) -> WorkerReport:
        """Run a single preset sub-agent and merge its work if it produced commits.

        Read-only presets (search, explore, architect, pm) never write, so their
        reports are returned without a merge step.
        """
        from .presets import build_preset_task, get_preset

        preset = get_preset(preset_name)
        task = build_preset_task(
            preset,
            objective,
            context_summary=f"Delegated by session {self.session_id}",
            base_commit="HEAD",
            timeout_seconds=self.settings.subagent.timeout_seconds,
            token_budget=self.settings.subagent.token_budget,
        )
        reports = await self.delegate_tasks([task], on_progress=on_progress)
        report = reports[0]
        if not preset.read_only and report.status == "completed" and report.commits:
            self.merge_subagent_results([report])
        return report

    def merge_subagent_results(self, reports: list[WorkerReport]) -> MergeResult:
        """Merge sub-agent worktree commits into this session's worktree."""
        result = merge_worktrees(self.workspace, reports)
        self._emit(
            "session",
            "worktrees_merged",
            {
                "merged": result.merged,
                "conflicts": result.conflicts,
                "success": result.success,
            },
        )
        return result

    @property
    def lesson_store(self) -> LessonStore:
        """Lesson store backed by the agent's nooa-memory MemoryManager."""
        if self._lesson_store is None:
            memory = getattr(self.agent, "_memory", None)
            if memory is None:
                raise RuntimeError(
                    "Memory is not enabled. Set memory.enabled=true in settings."
                )
            self._lesson_store = LessonStore(memory)
        return self._lesson_store

    @property
    def lesson_extractor(self) -> LessonExtractor:
        """Lazy-initialized lesson extractor."""
        if self._lesson_extractor is None:
            self._lesson_extractor = LessonExtractor(self.lesson_store, self.llm)
        return self._lesson_extractor

    @property
    def skill_manifest(self):
        """Public access to the skill routing manifest."""
        return getattr(self.agent, "_skill_manifest", None)

    async def learn_from_failure(
        self, task: str, error: str, evidence: str = ""
    ) -> None:
        """Extract and store a lesson from a failed task."""
        lesson = await self.lesson_extractor.extract_from_failure(
            task, error, evidence, session_id=self.session_id
        )
        if lesson:
            self._emit(
                "session",
                "lesson_learned",
                {"title": lesson.title, "category": lesson.category},
            )

    def recall_lessons(self, task: str, limit: int = 3) -> str:
        """Recall relevant lessons for a task, formatted as context."""
        lessons = self.lesson_extractor.recall_relevant(task, limit=limit)
        return self.lesson_extractor.format_for_context(lessons)

    @property
    def goal(self) -> GoalState | None:
        """Current active goal, or None."""
        return self._goal

    def set_goal(self, objective: str, *, turn_budget: int = 10) -> GoalState:
        """Set a goal-driven reconcile loop."""
        self._goal = GoalState(objective=objective, turn_budget=turn_budget)
        self._emit("session", "goal_set", {"objective": objective, "turn_budget": turn_budget})
        return self._goal

    def clear_goal(self) -> None:
        """Clear the active goal."""
        if self._goal is not None:
            self._goal.status = "cleared"
            self._emit("session", "goal_cleared", {"objective": self._goal.objective})
            self._goal = None

    async def evaluate_goal(self, last_result: CodingTaskResult) -> tuple[bool, str]:
        """Evaluate whether the active goal has been achieved.

        Returns (achieved, reason). Uses the LLM as an independent evaluator.
        """
        if self._goal is None:
            return True, "no active goal"
        self._goal.turns_used += 1

        if self._goal.turns_remaining <= 0:
            self._goal.status = "exhausted"
            reason = f"Turn budget exhausted ({self._goal.turn_budget} turns used)."
            self._goal.last_evaluation = reason
            self._emit("session", "goal_exhausted", {"turns_used": self._goal.turns_used})
            return False, reason

        # Use the LLM as an independent evaluator.
        eval_prompt = (
            "You are an independent evaluator. Your ONLY job is to determine whether "
            "the following goal has been achieved based on the evidence provided.\n\n"
            f"GOAL: {self._goal.objective}\n\n"
            "LAST TURN RESULT:\n"
            f"- status: {last_result.status}\n"
            f"- summary: {last_result.summary}\n"
            f"- changed_files: {last_result.changed_files}\n"
            f"- evidence: {last_result.evidence[:2000]}\n\n"
            "Respond with EXACTLY one of:\n"
            "ACHIEVED: <brief reason>\n"
            "NOT_ACHIEVED: <what is still missing and what should be done next>\n"
        )
        messages = [{"role": "user", "content": eval_prompt}]
        response = await self.llm.acall(messages)
        self._track_direct_llm_call(response)
        evaluation = response.content or "NOT_ACHIEVED: evaluator produced no output"
        self._goal.last_evaluation = evaluation

        achieved = evaluation.strip().upper().startswith("ACHIEVED")
        if achieved:
            self._goal.status = "achieved"
            self._emit("session", "goal_achieved", {
                "objective": self._goal.objective,
                "turns_used": self._goal.turns_used,
                "evaluation": evaluation,
            })
        else:
            self._emit("session", "goal_not_achieved", {
                "turns_used": self._goal.turns_used,
                "turns_remaining": self._goal.turns_remaining,
                "evaluation": evaluation,
            })
        return achieved, evaluation

    def _last_sequence(self) -> int:
        if not self.trace_path.is_file():
            return 0
        last = 0
        with self.trace_path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    last = max(last, int(json.loads(line).get("sequence", 0)))
                except (ValueError, json.JSONDecodeError):
                    continue
        return last

    def _persist_metadata(self) -> None:
        self.metadata.updated_at = datetime.now(UTC).isoformat()
        temporary = self.metadata_path.with_suffix(".tmp")
        temporary.write_text(self.metadata.model_dump_json(indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.metadata_path)

    def _set_status(self, status: SessionStatus, *, error: str | None = None) -> None:
        self.metadata.status = status
        self.metadata.error = error
        self._persist_metadata()

    def _emit(
        self,
        kind: SessionEventKind,
        name: str,
        data: dict[str, Any] | None = None,
    ) -> SessionEvent:
        self._sequence += 1
        event = SessionEvent(
            sequence=self._sequence,
            session_id=self.session_id,
            kind=kind,
            name=name,
            data=data or {},
        )
        with self.trace_path.open("a", encoding="utf-8") as stream:
            stream.write(event.model_dump_json() + "\n")
        self._event_queue.put_nowait(event)
        return event

    def _policy_event(self, name: str, data: dict[str, Any]) -> None:
        self._emit("approval", name, data)

    def _tool_event(self, name: str, data: dict[str, Any]) -> None:
        kind: SessionEventKind = "command" if name.startswith("command_") else "tool"
        self._emit(kind, name, data)

    def _model_failover(self, source: str, target: str, error: str) -> None:
        self._emit(
            "model_failover",
            "model_failover",
            {"from": source, "to": target, "error": error},
        )

    def _track_direct_llm_call(self, response: Any) -> None:
        """Accumulate token usage from a direct LLM call that bypasses agent events."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        prompt_tokens = getattr(usage, "prompt_tokens", 0)
        self._last_prompt_tokens = prompt_tokens
        self._token_usage.add(
            prompt_tokens=prompt_tokens,
            completion_tokens=getattr(usage, "completion_tokens", 0),
            cached_tokens=getattr(usage, "cached_tokens", 0),
            reasoning_tokens=getattr(usage, "reasoning_tokens", 0),
            cost_usd=getattr(usage, "cost_usd", 0.0),
        )
        self._emit(
            "usage",
            "token_usage",
            {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "source": "direct",
                "cumulative_tokens": self._token_usage.total_tokens,
            },
        )

    @property
    def token_usage(self) -> TokenUsage:
        return self._token_usage.model_copy()

    @property
    def current_context_tokens(self) -> int:
        """Prompt tokens of the most recent LLM call (current context fill)."""
        return self._last_prompt_tokens

    @property
    def context_window(self) -> int:
        """Resolved context window size for the active model (cached)."""
        if self._resolved_context_window is None:
            from .llm import resolve_context_window

            self._resolved_context_window = resolve_context_window(self.llm)
        return self._resolved_context_window

    def _on_nooa_event(self, event: Any) -> None:
        event_type = getattr(event, "event_type", type(event).__name__)
        # Emit thinking indicators for LLM call lifecycle.
        if event_type == "LLMCallStart":
            self._emit(
                "thinking",
                "llm_call_started",
                {
                    "method": getattr(event, "method_name", ""),
                    "turn": getattr(event, "turn_number", 0),
                },
            )
            return
        if event_type == "LLMCallEnd":
            self._emit(
                "thinking",
                "llm_call_ended",
                {
                    "method": getattr(event, "method_name", ""),
                    "success": getattr(event, "success", True),
                },
            )
            return
        # Accumulate token usage from LLMComplete metrics.
        if event_type == "LLMComplete":
            prompt_tokens = getattr(event, "prompt_tokens", 0)
            if prompt_tokens:
                self._last_prompt_tokens = prompt_tokens
            self._token_usage.add(
                prompt_tokens=prompt_tokens,
                completion_tokens=getattr(event, "completion_tokens", 0),
                cached_tokens=getattr(event, "cached_tokens", 0),
                reasoning_tokens=getattr(event, "reasoning_tokens", 0),
                cost_usd=getattr(event, "cost_usd", 0.0),
            )
            self._emit(
                "usage",
                "token_usage",
                {
                    "prompt_tokens": getattr(event, "prompt_tokens", 0),
                    "completion_tokens": getattr(event, "completion_tokens", 0),
                    "cached_tokens": getattr(event, "cached_tokens", 0),
                    "reasoning_tokens": getattr(event, "reasoning_tokens", 0),
                    "cost_usd": getattr(event, "cost_usd", 0.0),
                    "model": getattr(event, "model_name", ""),
                    "cumulative_tokens": self._token_usage.total_tokens,
                    "cumulative_cost_usd": self._token_usage.total_cost_usd,
                },
            )
            return
        try:
            payload = event.model_dump(mode="json")
        except Exception:
            payload = {"text": str(event)}
        kind: SessionEventKind = "message" if isinstance(event, AgentMessage) else "agent"
        self._emit(kind, event_type, payload)
        # Notification hook for agent messages.
        if isinstance(event, AgentMessage):
            text = getattr(event, "text", "") or str(event)
            with contextlib.suppress(Exception):
                asyncio.get_event_loop().create_task(
                    self._hook_runner.trigger_notification(text)
                )

    def _save_snapshot(self) -> str:
        snapshot_id = self.storage.save_snapshot(self.agent)
        self.metadata.latest_snapshot_id = snapshot_id
        self._persist_metadata()
        return snapshot_id

    async def prompt(self, text: str) -> CodingTaskResult:
        """Run one interactive turn and return its structured coding result."""
        task = text.strip()
        if not task:
            raise ValueError("prompt must be non-empty")
        if self._closed:
            raise RuntimeError("session is closed")
        if not self._session_start_fired:
            self._session_start_fired = True
            await self._fire_session_start()
        async with self._run_lock:
            if self._active_task is not None:
                raise RuntimeError("session already has an active turn")
            run_id = uuid.uuid4().hex
            self._set_status("running")
            self._auto_checkpoint()
            self._emit("session", "turn_started", {"run_id": run_id, "prompt": task})
            self._active_task = asyncio.create_task(
                self.agent.run_task(task, continued=bool(self.agent.task)),
                name=f"nooa-coding-{self.session_id}",
            )
            try:
                result = await self._active_task
            except asyncio.CancelledError:
                self._set_status("cancelled")
                self._emit("session", "turn_cancelled", {"run_id": run_id})
                raise
            except Exception as exc:
                self._set_status("failed", error=f"{type(exc).__name__}: {exc}")
                self._emit(
                    "error",
                    "turn_failed",
                    {"run_id": run_id, "type": type(exc).__name__, "message": str(exc)},
                )
                # Auto-extract lesson from failure.
                with contextlib.suppress(Exception):
                    await self.learn_from_failure(task, f"{type(exc).__name__}: {exc}")
                raise
            else:
                self._set_status("idle")
                self._emit(
                    "session",
                    "turn_finished",
                    {"run_id": run_id, "result": result.model_dump(mode="json")},
                )
                # Stop hook — runs after agent finishes.
                with contextlib.suppress(Exception):
                    await self._hook_runner.trigger_stop(summary=result.summary)
                return result
            finally:
                self._active_task = None
                self._save_snapshot()

    def start(self, text: str) -> asyncio.Task[CodingTaskResult]:
        """Start a turn so the caller can concurrently consume events or approvals."""
        if self._active_task is not None:
            raise RuntimeError("session already has an active turn")
        return asyncio.create_task(self.prompt(text), name=f"session-prompt-{self.session_id}")

    async def cancel(self) -> bool:
        """Cancel the active model/tool turn and persist recoverable state."""
        task = self._active_task
        if task is None or task.done():
            return False
        self._set_status("cancelling")
        self.approvals.cancel_all()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        return True

    def approve(self, request_id: str) -> None:
        self.approvals.decide(request_id, allow=True)

    def deny(self, request_id: str) -> None:
        self.approvals.decide(request_id, allow=False)

    def clear_history(self) -> None:
        """Clear the agent conversation history (in-memory event store)."""
        backend = self.agent.event_manager._backend  # noqa: SLF001
        backend._events.clear()  # noqa: SLF001
        backend._active_tags.clear()  # noqa: SLF001
        self._emit("session", "history_cleared", {})

    def mcp_status(self) -> list[MCPServerStatus]:
        return self.agent._mcp.statuses()

    def mcp_config_errors(self) -> list[str]:
        return list(self.agent._mcp.config_errors)

    def mcp_tools(self, server_name: str | None = None) -> dict[str, list[str]]:
        return self.agent._mcp.tools(server_name)

    def mcp_enable(self, server_name: str) -> MCPServerStatus:
        if self._active_task is not None:
            raise RuntimeError("cannot change MCP servers while a turn is running")
        return self.agent._mcp.enable(server_name)

    def mcp_disable(self, server_name: str) -> MCPServerStatus:
        if self._active_task is not None:
            raise RuntimeError("cannot change MCP servers while a turn is running")
        return self.agent._mcp.disable(server_name)

    def mcp_reload(self, server_name: str | None = None) -> list[MCPServerStatus]:
        if self._active_task is not None:
            raise RuntimeError("cannot reload MCP servers while a turn is running")
        return self.agent._mcp.reload(server_name)

    async def compact(self, *, preserve_recent: int | None = None) -> str | None:
        if self._active_task is not None:
            raise RuntimeError("cannot compact while a turn is running")
        # PreCompact hook — allows external tools to save context before compaction.
        with contextlib.suppress(Exception):
            await self._hook_runner.trigger_pre_compact(
                context_summary=f"preserve_recent={preserve_recent}"
            )
        tag = await self.agent.compact_history(preserve_recent=preserve_recent)
        self._save_snapshot()
        self._emit("session", "history_compacted", {"summary_tag": tag})
        # PostCompact hook — verify summary integrity.
        with contextlib.suppress(Exception):
            await self._hook_runner.trigger_post_compact(summary=tag or "")
        return tag

    def diff(self) -> DiffResult:
        return WorkspaceManager.diff(self.workspace)

    def checkpoint(self, label: str = "manual") -> Checkpoint:
        if self._active_task is not None:
            raise RuntimeError("cannot checkpoint while a turn is running")
        snapshot_id = self._save_snapshot()
        checkpoint = WorkspaceManager.checkpoint(self.workspace, label).model_copy(
            update={"snapshot_id": snapshot_id}
        )
        self.metadata.checkpoints.append(checkpoint)
        self._persist_metadata()
        self._emit("checkpoint", "checkpoint_created", checkpoint.model_dump(mode="json"))
        return checkpoint

    def _auto_checkpoint(self) -> None:
        """Snapshot agent state and commit the worktree before a turn runs.

        This powers ``/undo``: each turn is bracketed by a recoverable checkpoint
        so the previous turn's changes can be reverted exactly.
        """
        snapshot_id = self._save_snapshot()
        checkpoint = WorkspaceManager.checkpoint(
            self.workspace, f"auto-turn-{self._sequence}"
        ).model_copy(update={"snapshot_id": snapshot_id})
        self.metadata.checkpoints.append(checkpoint)
        self._persist_metadata()

    def undo(self) -> Checkpoint:
        """Revert the worktree and agent state to before the most recent turn."""
        if self._active_task is not None:
            raise RuntimeError("cannot undo while a turn is running")
        auto = [c for c in self.metadata.checkpoints if c.label.startswith("auto-turn")]
        if not auto:
            raise KeyError("nothing to undo")
        target = auto[-1]
        # Drop this checkpoint so a repeated /undo steps further back.
        self.metadata.checkpoints = [
            c for c in self.metadata.checkpoints if c is not target
        ]
        WorkspaceManager.rollback(self.workspace, target)
        if target.snapshot_id:
            self.storage.restore_snapshot(target.snapshot_id, self.agent)
        self._persist_metadata()
        self._emit("checkpoint", "undo", target.model_dump(mode="json"))
        self._save_snapshot()
        return target

    def rollback(self, checkpoint_id: str) -> Checkpoint:
        if self._active_task is not None:
            raise RuntimeError("cannot rollback while a turn is running")
        matches = [
            item
            for item in self.metadata.checkpoints
            if item.checkpoint_id == checkpoint_id or item.commit == checkpoint_id
        ]
        if len(matches) != 1:
            raise KeyError(f"unknown or ambiguous checkpoint: {checkpoint_id}")
        checkpoint = matches[0]
        WorkspaceManager.rollback(self.workspace, checkpoint)
        if checkpoint.snapshot_id:
            self.storage.restore_snapshot(checkpoint.snapshot_id, self.agent)
        self._emit("checkpoint", "checkpoint_rolled_back", checkpoint.model_dump(mode="json"))
        self._save_snapshot()
        return checkpoint

    async def fork(
        self,
        new_session_id: str | None = None,
        *,
        llm: UnifiedLLM | None = None,
    ) -> AgentSession:
        if self._active_task is not None:
            raise RuntimeError("cannot fork while a turn is running")
        checkpoint = self.checkpoint("fork")
        return self.manager.fork(self, checkpoint, new_session_id=new_session_id, llm=llm)

    def replay(self, *, after_sequence: int = 0) -> list[SessionEvent]:
        if not self.trace_path.is_file():
            return []
        events: list[SessionEvent] = []
        with self.trace_path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    event = SessionEvent.model_validate_json(line)
                except ValueError:
                    continue
                if event.sequence > after_sequence:
                    events.append(event)
        return events

    async def stream(
        self,
        *,
        after_sequence: int = 0,
        follow: bool = True,
    ) -> AsyncIterator[SessionEvent]:
        cursor = after_sequence
        for event in self.replay(after_sequence=cursor):
            cursor = max(cursor, event.sequence)
            yield event
        if not follow:
            return
        while not self._closed:
            event = await self._event_queue.get()
            if event.sequence > cursor:
                cursor = event.sequence
                yield event

    async def close(self) -> None:
        if self._closed:
            return
        if self._active_task is not None:
            await self.cancel()
        self._save_snapshot()
        self._unsubscribe()
        await self.agent.close()
        self.storage.close()
        await self.llm.aclose()
        self._closed = True


class AgentSessionManager:
    """Create, resume, list, and fork project-scoped local sessions."""

    def __init__(
        self,
        repo: str | Path,
        settings: CodingSettings | None = None,
        *,
        settings_file: str | Path | None = None,
    ) -> None:
        self.repo = Path(repo).expanduser().resolve()
        if not self.repo.is_dir():
            raise ValueError(f"repo must be an existing directory: {self.repo}")
        self.settings = settings or load_settings(self.repo, settings_file)
        self.project_sessions_dir = self.settings.sessions_path() / _repo_key(self.repo)
        self.workspace_manager = WorkspaceManager(self.repo, self.settings.worktrees_path())

    @staticmethod
    def validate_session_id(session_id: str) -> str:
        value = session_id.strip()
        if value in {".", ".."} or not _SESSION_ID.fullmatch(value):
            raise ValueError("session id must be 1-128 path-safe characters")
        return value

    @staticmethod
    def generate_session_id() -> str:
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        return f"{stamp}-{uuid.uuid4().hex[:8]}"

    def session_dir(self, session_id: str) -> Path:
        return self.project_sessions_dir / self.validate_session_id(session_id)

    def _normalized_settings(self) -> CodingSettings:
        memory = self.settings.memory
        path = Path(memory.path).expanduser()
        if not path.is_absolute():
            path = self.project_sessions_dir / "memory.sqlite"
        return self.settings.model_copy(
            update={"memory": memory.model_copy(update={"path": str(path)})}
        )

    def _metadata_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "metadata.json"

    def _write_new_metadata(self, metadata: SessionMetadata) -> None:
        directory = self.session_dir(metadata.session_id)
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "metadata.json").write_text(
            metadata.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )

    def create(
        self,
        session_id: str | None = None,
        *,
        start_ref: str = "HEAD",
        parent_session_id: str | None = None,
        llm: UnifiedLLM | None = None,
    ) -> AgentSession:
        value = self.validate_session_id(session_id or self.generate_session_id())
        if self.session_dir(value).exists():
            raise FileExistsError(f"session already exists: {value}")
        workspace = self.workspace_manager.create(value, start_ref=start_ref)
        metadata = SessionMetadata(
            session_id=value,
            base_repo=str(self.repo),
            workspace=workspace,
            parent_session_id=parent_session_id,
        )
        try:
            self._write_new_metadata(metadata)
            session = AgentSession(
                self,
                metadata,
                self._normalized_settings(),
                llm=llm,
            )
            session._emit("session", "session_created", {"workspace": workspace.model_dump()})
            session.checkpoint("initial")
            return session
        except Exception:
            # Preserve the worktree for diagnosis; only incomplete session metadata is removed.
            raise

    def resume(self, session_id: str, *, llm: UnifiedLLM | None = None) -> AgentSession:
        path = self._metadata_path(session_id)
        if not path.is_file():
            raise FileNotFoundError(f"session does not exist: {session_id}")
        metadata = SessionMetadata.model_validate_json(path.read_text(encoding="utf-8"))
        was_running = metadata.status in {"running", "cancelling"}
        if was_running:
            metadata.recovered_after_crash = True
            metadata.status = "idle"
            metadata.error = "Previous process exited during an active turn."
        session = AgentSession(
            self,
            metadata,
            self._normalized_settings(),
            llm=llm,
            restore=True,
        )
        session._emit("session", "session_resumed", {"crash_recovery": was_running})
        return session

    def list(self) -> list[SessionSummary]:
        if not self.project_sessions_dir.is_dir():
            return []
        result: list[SessionSummary] = []
        for path in self.project_sessions_dir.glob("*/metadata.json"):
            try:
                metadata = SessionMetadata.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            result.append(
                SessionSummary(
                    session_id=metadata.session_id,
                    status=metadata.status,
                    workspace=metadata.workspace.path,
                    updated_at=metadata.updated_at,
                    parent_session_id=metadata.parent_session_id,
                    checkpoint_count=len(metadata.checkpoints),
                    recovered_after_crash=metadata.recovered_after_crash,
                )
            )
        return sorted(result, key=lambda item: item.updated_at, reverse=True)

    def fork(
        self,
        source: AgentSession,
        checkpoint: Checkpoint,
        *,
        new_session_id: str | None = None,
        llm: UnifiedLLM | None = None,
    ) -> AgentSession:
        value = self.validate_session_id(new_session_id or self.generate_session_id())
        workspace = self.workspace_manager.create(value, start_ref=checkpoint.commit)
        metadata = SessionMetadata(
            session_id=value,
            base_repo=str(self.repo),
            workspace=workspace,
            parent_session_id=source.session_id,
            checkpoints=[checkpoint],
        )
        self._write_new_metadata(metadata)
        destination = self.session_dir(value) / "session.db"
        source_connection = sqlite3.connect(source.database_path)
        destination_connection = sqlite3.connect(destination)
        try:
            source_connection.backup(destination_connection)
        finally:
            destination_connection.close()
            source_connection.close()
        session = AgentSession(
            self,
            metadata,
            self._normalized_settings(),
            llm=llm,
            restore=True,
        )
        session._emit(
            "session",
            "session_forked",
            {"parent_session_id": source.session_id, "checkpoint": checkpoint.checkpoint_id},
        )
        return session

    async def delegate_parallel(
        self,
        source: AgentSession,
        subtasks: list[str],
        *,
        llm_factory: Callable[[], UnifiedLLM] | None = None,
    ) -> list[dict[str, Any]]:
        """Spawn parallel child sessions for independent subtasks.

        Each subtask runs in its own forked worktree. Results are collected
        and returned as a list of dicts with session_id, status, and summary.
        """
        checkpoint = source.checkpoint("delegate")
        results: list[dict[str, Any]] = []

        async def _run_one(index: int, task: str) -> dict[str, Any]:
            child_llm = llm_factory() if llm_factory else None
            child = self.fork(
                source, checkpoint, new_session_id=f"{source.session_id}-sub{index}", llm=child_llm
            )
            try:
                result = await child.prompt(task)
                return {
                    "session_id": child.session_id,
                    "task": task[:100],
                    "status": result.status,
                    "summary": result.summary[:200],
                }
            except Exception as exc:
                return {
                    "session_id": child.session_id,
                    "task": task[:100],
                    "status": "failed",
                    "summary": f"{type(exc).__name__}: {exc}",
                }
            finally:
                await child.close()

        tasks = [_run_one(i, subtask) for i, subtask in enumerate(subtasks)]
        results = list(await asyncio.gather(*tasks))
        return results


__all__ = [
    "AgentSession",
    "AgentSessionManager",
    "GoalState",
    "SessionMetadata",
    "SessionStatus",
    "SessionSummary",
]
