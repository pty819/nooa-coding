"""NOOA agent implementation; lifecycle decisions remain in deterministic Python."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

from nooa import CodeActStrategy, Context, hidden, strategy
from nooa.agentdoc import doc, spec
from nooa.agents.summarization import TokenBudgetSummarizer, context_budget
from nooa.config import CodeActConfig, PredictConfig, TokenBudgetConfig
from nooa.interactive import InteractiveAgent, RespondReason, RespondResult
from nooa.runtime.restrictions import RESTRICTED_MODULES, RestrictionsConfig
from nooa.skill_registry import SkillRegistry
from nooa.storage.markers import nosnapshot
from nooa.strategies import PredictStrategy
from nooa.tools.todo import TodoManager
from nooa_cli.tools.repo_tools import RepoTools
from nooa_memory import MemoryManager, MemoryToolsMixin
from pydantic import BaseModel, Field

from .config import CodingSettings
from .mcp import MCPRuntime
from .plugin import PluginRegistry
from .policy import ApprovalManager, PolicyShellTools
from .resources import install_resources, load_agents_context
from .tools import CodeSearch, LSPTools

with hidden:
    import ast
    import hashlib

    from .system_prompt import render_runtime_system_context

if TYPE_CHECKING:
    from nooa.storage.manager import StorageManager
    from nooa.unifiedllm import UnifiedLLM


class CodingTaskDraft(BaseModel):
    status: Literal["completed", "blocked"]
    summary: str
    root_cause: str = "not applicable"
    changed_files: list[str] = Field(default_factory=list)
    evidence: str
    suggested_verification: str = ""


class InspectionDraft(BaseModel):
    status: Literal["completed", "blocked"]
    summary: str
    evidence: str
    root_cause: str = "not applicable"


class ConversationDraft(BaseModel):
    answer: str


class RequestRoute(BaseModel):
    mode: Literal["conversation", "inspect", "change"]
    reason: str


class VerificationResult(BaseModel):
    command: str
    passed: bool
    returncode: int
    stdout: str = ""
    stderr: str = ""


class CodingTaskResult(BaseModel):
    mode: Literal["conversation", "inspect", "change"] = "change"
    status: Literal["answered", "inspected", "completed", "verification_failed", "blocked"]
    summary: str
    root_cause: str = "not applicable"
    changed_files: list[str] = Field(default_factory=list)
    evidence: str = ""
    verifications: list[VerificationResult] = Field(default_factory=list)
    model: str = ""


_CODEACT_CONFIG = CodeActConfig(
    max_iterations=120,
    max_retries=10,
    max_consecutive_text_only=4,
    text_only_stop_behavior="return_result",
    cell_timeout=120,
    restrictions=RestrictionsConfig(restricted_imports=RESTRICTED_MODULES),
)


def _cell_policy_violation(code: str) -> str | None:
    """Reject common direct-I/O paths so generated code uses approved tools."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    blocked_names = {
        "Path",
        "open",
        "os",
        "pathlib",
        "requests",
        "shutil",
        "socket",
        "subprocess",
        "urllib",
    }
    blocked_file_methods = {
        "chmod",
        "hardlink_to",
        "mkdir",
        "open",
        "rename",
        "rmdir",
        "symlink_to",
        "touch",
        "unlink",
        "write_bytes",
        "write_text",
    }
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name.split(".", 1)[0] for alias in node.names]
            if isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module.split(".", 1)[0])
            match = blocked_names.intersection(names)
            if match:
                return f"direct import {sorted(match)[0]!r} is blocked; use self.shell"
        if isinstance(node, ast.Name) and node.id in blocked_names:
            return f"direct access to {node.id!r} is blocked; use self.shell"
        if isinstance(node, ast.Attribute):
            if node.attr in blocked_file_methods:
                return f"direct file method {node.attr!r} is blocked; use self.shell"
            current: ast.AST = node
            attributes: list[str] = []
            while isinstance(current, ast.Attribute):
                attributes.append(current.attr)
                current = current.value
            if (
                isinstance(current, ast.Name)
                and current.id == "self"
                and any(value.startswith("_") for value in attributes)
            ):
                return "private agent attributes are blocked in generated code"
    return None


class CodingAgent(MemoryToolsMixin, InteractiveAgent):
    """Operate as NOOA Coding Agent inside an isolated Git worktree.

    {self._runtime_system_context()}
    """

    shell: Annotated[PolicyShellTools, nosnapshot]
    repo: Annotated[RepoTools, nosnapshot]
    todo: TodoManager
    skills: Annotated[SkillRegistry, nosnapshot]
    search: Annotated[CodeSearch, nosnapshot]
    lsp: Annotated[LSPTools, nosnapshot]
    task: str = ""
    last_result: CodingTaskResult | None = None
    _repo_root: Annotated[Path, hidden, nosnapshot]
    _settings: Annotated[CodingSettings, hidden, nosnapshot]
    _event_sink: Annotated[Any, hidden, nosnapshot]
    _memory: Annotated[MemoryManager, hidden, nosnapshot]
    _mcp: Annotated[MCPRuntime, hidden, nosnapshot]
    _plugins: Annotated[PluginRegistry, hidden, nosnapshot]

    def __init__(
        self,
        llm: UnifiedLLM,
        repo: str | Path,
        settings: CodingSettings,
        shell: PolicyShellTools,
        approvals: ApprovalManager,
        event_sink: Any,
        *,
        storage: StorageManager | None = None,
    ) -> None:
        super().__init__(llm=llm, storage=storage)  # pyright: ignore[reportCallIssue]
        self._repo_root = Path(repo).expanduser().resolve()
        self._settings = settings
        self._event_sink = event_sink
        self.shell = shell
        self.repo = RepoTools(root=self._repo_root, session=self.shell._session)
        self.todo = TodoManager()
        self.skills = install_resources(self, self._repo_root, settings.resources)
        self.search = CodeSearch(self._repo_root)
        self.lsp = LSPTools(self._repo_root)
        self._mcp = MCPRuntime(
            self._repo_root,
            settings.mcp,
            approvals,
            event_sink,
        )
        self._mcp.install(self)
        self._plugins = PluginRegistry()
        plugin_dir = Path.home() / ".config" / "nooa-coding" / "plugins"
        self._plugins.discover_directory(plugin_dir)
        self._plugins.install_all(self)
        self.context_manager["coding_tools"] = Context(
            doc(PolicyShellTools, RepoTools, TodoManager), prefix=True
        )
        self.context_manager.set_dynamic("task", "self.task")
        self.context_manager.set_dynamic("todo_status", "self.todo.status()")
        spec(self, "context", hidden=False)
        spec(self, "events", hidden=False)
        self._install_memory()
        self._install_compaction()
        self._install_cell_policy()

    def _install_cell_policy(self) -> None:
        async def guard(ctx: Any, nxt: Any) -> Any:
            code = getattr(ctx, "code", None)
            violation = _cell_policy_violation(code) if isinstance(code, str) else None
            if violation:
                self._event_sink("cell_guard_blocked", {"reason": violation})
                ctx.code = f"print({'BLOCKED: ' + violation!r})"
            return await nxt(ctx)

        self.event_manager.intercept("execute_python", guard)

    def _install_memory(self) -> None:
        config = self._settings.memory
        if not config.enabled:
            return
        path = Path(config.path).expanduser()
        if not path.is_absolute():
            path = self._repo_root / path
        MemoryManager.install(
            self,
            config=config.model_copy(update={"enabled": True, "path": str(path)}),
        )

    def _install_compaction(self) -> None:
        config = self._settings.compaction
        if not config.enabled:
            return
        max_tokens = config.max_tokens or context_budget(
            self.llm,
            percent=config.context_fraction,
            fallback=100_000,
        )
        TokenBudgetSummarizer.install(
            self,
            config=TokenBudgetConfig(
                max_tokens=max_tokens,
                preserve_recent=config.preserve_recent,
                target_chars=config.target_chars,
            ),
        )

    def reload_project_resources(self) -> str:
        """Reload AGENTS.md content; new skills are discovered and activated."""
        from .resources import build_skill_manifest

        instructions = load_agents_context(self._repo_root, self._settings.resources)
        if instructions:
            self.context_manager["project_instructions"] = Context(instructions, prefix=True)
        directories = []
        for value in self._settings.resources.skills_dirs:
            path = Path(value).expanduser()
            directories.append(path if path.is_absolute() else self._repo_root / path)
        self.skills.discover_skills_dirs(directories)
        self.skills.activate(list(self._settings.resources.activate_skills))
        # Rebuild the skill manifest so routing metadata stays current.
        self._skill_manifest = build_skill_manifest(directories)  # noqa: SLF001
        return f"Reloaded project instructions and {len(self.skills.activated())} active skills."

    @hidden
    async def close(self) -> None:
        for summarizer in list(getattr(self, "_summarizers", [])):
            summarizer._uninstall()
        memory = getattr(self, "_memory", None)
        if isinstance(memory, MemoryManager):
            memory.uninstall()
        self._plugins.notify_close()
        self._mcp.close()
        await self.shell.close()

    @hidden
    def message(self, text: str, *, echo: bool = False) -> None:
        """Keep the inherited synchronous primitive out of the generated API."""
        super().message(text, echo=echo)

    async def notify(self, text: str) -> None:
        """Send a concise progress update to the user. This method is asynchronous."""
        self.message(text)

    def _current_model(self) -> str:
        active = getattr(self.llm, "active", self.llm)
        return str(getattr(active, "model", getattr(self.llm, "model", "unknown")))

    def _runtime_system_context(self) -> str:
        return render_runtime_system_context(
            active_model=self._current_model(),
            worktree=self._repo_root,
        )

    def _local_answer(self, task: str) -> str | None:
        normalized = " ".join(task.lower().split())
        if any(
            marker in normalized
            for marker in (
                "什么模型",
                "哪个模型",
                "模型是什么",
                "which model",
                "what model",
                "model are you",
                "model is this",
            )
        ):
            return f"当前会话实际使用的模型是 `{self._current_model()}`。"
        if normalized in {"你是谁", "你是什么", "who are you", "what are you"}:
            return (
                "我是 NOOA Coding Agent：在隔离的 Git worktree 中审阅、修改并验证代码。"
                f"当前模型是 `{self._current_model()}`。"
            )
        if any(marker in normalized for marker in ("你能做什么", "有什么能力", "capabilities")):
            return (
                "我可以解释和审阅仓库、定位问题、实施代码改动、运行验证，并通过审批门控制高风险操作。"
                "审阅任务默认只读，改动任务会在隔离 worktree 中执行。"
            )
        if normalized in {"你好", "hi", "hello", "hey"}:
            return "你好。我可以先只读分析仓库，也可以在隔离 worktree 中实现并验证一个具体改动。"
        return None

    @staticmethod
    def _obvious_route(task: str) -> Literal["conversation", "inspect", "change"] | None:
        normalized = task.lower()
        question_form = normalized.rstrip().endswith(("?", "？")) or any(
            marker in normalized for marker in ("如何", "怎么", "怎样", "how ")
        )
        explicit_action = any(
            marker in normalized for marker in ("帮我", "请", "直接", "can you", "please")
        )
        repository_terms = (
            "代码",
            "函数",
            "类",
            "仓库",
            "项目",
            "模块",
            "测试",
            "报错",
            "文件",
            "code",
            "function",
            "class",
            "repo",
            "module",
            "test",
            "error",
        )
        if (
            question_form
            and not explicit_action
            and any(marker in normalized for marker in repository_terms)
        ):
            return "inspect"
        change_markers = (
            "修复",
            "修改",
            "实现",
            "添加",
            "新增",
            "删除",
            "重构",
            "优化",
            "升级",
            "接入",
            "改成",
            "fix ",
            "implement",
            "add ",
            "create ",
            "write ",
            "change ",
            "update ",
            "refactor",
            "remove ",
        )
        if any(marker in normalized for marker in change_markers):
            return "change"
        inspect_markers = (
            "审阅",
            "分析",
            "解释",
            "排查",
            "检查",
            "评估",
            "理解",
            "为什么",
            "是什么原因",
            "review",
            "analyze",
            "explain",
            "inspect",
            "investigate",
            "diagnose",
        )
        if any(marker in normalized for marker in inspect_markers):
            return "inspect"
        if normalized.rstrip().endswith(("?", "？")):
            return "conversation"
        return None

    async def _route_request(self, task: str) -> Literal["conversation", "inspect", "change"]:
        obvious = self._obvious_route(task)
        if obvious is not None:
            return obvious
        return (await self._classify_request(task)).mode

    @staticmethod
    def _paths_from_status(status: str) -> list[str]:
        paths: list[str] = []
        records = status.split("\0")
        index = 0
        while index < len(records):
            record = records[index]
            if len(record) < 4:
                index += 1
                continue
            paths.append(record[3:])
            renamed_or_copied = "R" in record[:2] or "C" in record[:2]
            index += 2 if renamed_or_copied else 1
        return paths

    def _untracked_signature(self, status: str) -> str:
        digest = hashlib.sha256()
        remaining_bytes = 50_000_000
        for record in status.split("\0"):
            if not record.startswith("?? "):
                continue
            relative = record[3:]
            candidate = self._repo_root / relative
            digest.update(relative.encode(errors="surrogateescape"))
            try:
                metadata = candidate.lstat()
                digest.update(f":{metadata.st_mode}:{metadata.st_size}:".encode())
                if candidate.is_symlink():
                    digest.update(str(candidate.readlink()).encode(errors="surrogateescape"))
                elif candidate.is_file() and metadata.st_size <= remaining_bytes:
                    with candidate.open("rb") as stream:
                        while chunk := stream.read(1_048_576):
                            digest.update(chunk)
                    remaining_bytes -= metadata.st_size
                else:
                    digest.update(str(metadata.st_mtime_ns).encode())
            except OSError as exc:
                digest.update(f"unavailable:{type(exc).__name__}".encode())
        return digest.hexdigest()

    async def _worktree_state(self) -> tuple[str, str, str, str]:
        """Capture enough host state to prove that this turn actually changed the worktree."""
        head = await self.shell._run_trusted("git rev-parse HEAD", timeout=15)
        status = await self.shell._run_trusted(
            "git status --porcelain=v1 -z --untracked-files=all", timeout=15
        )
        diff_hash = await self.shell._run_trusted(
            "git diff --binary --no-ext-diff HEAD | git hash-object --stdin",
            timeout=30,
        )
        return (
            head.stdout,
            status.stdout,
            diff_hash.stdout,
            self._untracked_signature(status.stdout),
        )

    @hidden
    async def run_task(self, description: str, *, continued: bool = False) -> CodingTaskResult:
        """Route one request and enforce mode-specific host policy and verification."""
        task = description.strip()
        if not task:
            raise ValueError("task must be non-empty")
        self.task = task
        self._plugins.notify_turn_start(task)
        local_answer = self._local_answer(task)
        if local_answer is not None:
            result = CodingTaskResult(
                mode="conversation",
                status="answered",
                summary=local_answer,
                evidence="Answered from live host session configuration.",
                model=self._current_model(),
            )
            self.last_result = result
            self._plugins.notify_turn_end(result)
            return result

        mode = await self._route_request(task)
        if not continued:
            self.todo.clear()
        if mode == "conversation":
            draft = await self._answer_question(task)
            result = CodingTaskResult(
                mode=mode,
                status="answered",
                summary=draft.answer,
                evidence="Generated without repository mutation tools.",
                model=self._current_model(),
            )
            self.last_result = result
            self._plugins.notify_turn_end(result)
            return result

        if not self.todo.list_todos():
            self.todo.add("Inspect the repository and establish concrete evidence")
        if mode == "inspect":
            async with self.shell._read_only_scope(), self._mcp.read_only():
                inspection = await self._inspect_repository(task)
            result = CodingTaskResult(
                mode=mode,
                status="inspected" if inspection.status == "completed" else "blocked",
                summary=inspection.summary,
                root_cause=inspection.root_cause,
                evidence=inspection.evidence,
                model=self._current_model(),
            )
            self.last_result = result
            self._plugins.notify_turn_end(result)
            return result

        state_before = await self._worktree_state()
        draft = await self._implement_change(task)
        state_after = await self._worktree_state()
        changed_files = self._paths_from_status(state_after[1])
        verifications: list[VerificationResult] = []
        status: Literal["completed", "verification_failed", "blocked"] = draft.status
        if draft.status == "completed":
            if not changed_files or state_before == state_after:
                status = "verification_failed"
                draft.evidence += "\nThe host found no new worktree change produced by this turn."
            configured = list(self._settings.verification_commands)
            suggested = draft.suggested_verification.strip()
            commands = ["git diff --check", *configured]
            if not configured and suggested:
                commands.append(suggested)
            if not configured and not suggested:
                status = "verification_failed"
                draft.evidence += (
                    "\nNo behavioral verification command was configured or suggested."
                )
            for index, command in enumerate(commands):
                try:
                    if index == 0 or command in configured:
                        shell_result = await self.shell._run_trusted(
                            command,
                            timeout=self._settings.limits.verification_timeout,
                        )
                    else:
                        shell_result = await self.shell.run(
                            command,
                            timeout=self._settings.limits.verification_timeout,
                        )
                except PermissionError as exc:
                    verifications.append(
                        VerificationResult(
                            command=command,
                            passed=False,
                            returncode=126,
                            stderr=str(exc),
                        )
                    )
                    status = "verification_failed"
                    break
                verifications.append(
                    VerificationResult(
                        command=command,
                        passed=shell_result.success,
                        returncode=shell_result.returncode,
                        stdout=shell_result.stdout[-8_000:],
                        stderr=shell_result.stderr[-8_000:],
                    )
                )
                if not shell_result.success:
                    status = "verification_failed"
                    break

        result = CodingTaskResult(
            mode="change",
            status=status,
            summary=draft.summary,
            root_cause=draft.root_cause,
            changed_files=changed_files,
            evidence=draft.evidence,
            verifications=verifications,
            model=self._current_model(),
        )
        self.last_result = result
        self._plugins.notify_turn_end(result)
        return result

    @hidden
    async def handle(self, notification: dict[str, list]) -> RespondResult:
        """Deterministically turn queued user text into one coding task."""
        messages = [str(item) for item in notification.get("user_messages", [])]
        if not messages:
            return RespondResult(
                kind=RespondReason.NEED_INPUT, explanation="waiting for a coding request"
            )
        result = await self.run_task("\n".join(messages), continued=bool(self.task))
        self.message(result.summary)
        reason = (
            RespondReason.DONE
            if result.status in {"answered", "inspected", "completed"}
            else RespondReason.NEED_INPUT
        )
        return RespondResult(kind=reason, explanation=f"coding task ended with {result.status}")

    @hidden
    async def compact_history(self, *, preserve_recent: int | None = None) -> str | None:
        """Explicitly summarize and collapse older active events."""
        tags = self.events.keys()
        keep = preserve_recent or self._settings.compaction.preserve_recent
        if len(tags) <= keep + 1:
            return None
        old_tags = tags[:-keep]
        rendered = "\n\n".join(f"[{tag}] {self.events[tag]}" for tag in old_tags)
        summary = await self._summarize_history(rendered)
        return self.events.collapse(old_tags[0], old_tags[-1], summary)

    @strategy(PredictStrategy(config=PredictConfig(max_param_chars=200_000)))
    async def _summarize_history(self, history: str) -> str:
        """Preserve decisions, evidence, modifications, open work, and identifiers from history."""
        ...

    @strategy(PredictStrategy(config=PredictConfig(max_param_chars=20_000)))
    async def _classify_request(self, request: str) -> RequestRoute:
        """Classify one request as conversation, read-only repository inspection, or code change."""
        ...

    @strategy(PredictStrategy(config=PredictConfig(max_param_chars=100_000)))
    async def _answer_question(self, question: str) -> ConversationDraft:
        """Answer a non-repository conversational question concisely and honestly."""
        ...

    @strategy(CodeActStrategy(config=_CODEACT_CONFIG))
    async def _inspect_repository(self, request: str) -> InspectionDraft:
        """Investigate one repository question without modifying files.

        Establish evidence with policy-controlled read and shell tools. Cite concrete
        paths, symbols, and command output. Use `await self.notify(...)` for only
        meaningful progress. Return `blocked` if evidence cannot be obtained.
        """
        ...

    @strategy(CodeActStrategy(config=_CODEACT_CONFIG))
    async def _implement_change(self, request: str) -> CodingTaskDraft:
        """Implement one code change in the isolated repository worktree.

        Inspect before editing and preserve unrelated work. Use the supplied
        policy-controlled shell and repository tools. Keep the todo list current.
        Run focused checks while implementing. The deterministic host will run
        configured final verification after this method returns.

        Use `await self.notify(...)` for concise, useful progress. Use
        `self.recall(...)` before work and store only verified, durable knowledge
        with `self.remember(...)`. Reload instructions if the repository changes
        them. Never claim completion without concrete command output.

        Return `blocked` when required information or permission is unavailable.
        """
        ...


__all__ = [
    "CodingAgent",
    "CodingTaskDraft",
    "CodingTaskResult",
    "ConversationDraft",
    "InspectionDraft",
    "RequestRoute",
    "VerificationResult",
]
