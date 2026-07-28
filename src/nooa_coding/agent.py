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
from .policy import PolicyShellTools
from .resources import install_resources, load_agents_context

with hidden:
    import ast

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


class VerificationResult(BaseModel):
    command: str
    passed: bool
    returncode: int
    stdout: str = ""
    stderr: str = ""


class CodingTaskResult(BaseModel):
    status: Literal["completed", "verification_failed", "blocked"]
    summary: str
    root_cause: str
    changed_files: list[str]
    evidence: str
    verifications: list[VerificationResult] = Field(default_factory=list)


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
            while isinstance(current, ast.Attribute):
                if current.attr.startswith("_"):
                    return "private runtime attributes are blocked in generated code"
                current = current.value
            if isinstance(current, ast.Name) and current.id == "self" and node.attr.startswith("_"):
                return "private agent attributes are blocked in generated code"
    return None


class CodingAgent(MemoryToolsMixin, InteractiveAgent):
    """Solve coding tasks inside one isolated worktree with evidence and approval gates."""

    shell: Annotated[PolicyShellTools, nosnapshot]
    repo: Annotated[RepoTools, nosnapshot]
    todo: TodoManager
    skills: Annotated[SkillRegistry, nosnapshot]
    task: str = ""
    last_result: CodingTaskResult | None = None
    _repo_root: Annotated[Path, hidden, nosnapshot]
    _settings: Annotated[CodingSettings, hidden, nosnapshot]
    _event_sink: Annotated[Any, hidden, nosnapshot]
    _memory: Annotated[MemoryManager, hidden, nosnapshot]

    def __init__(
        self,
        llm: UnifiedLLM,
        repo: str | Path,
        settings: CodingSettings,
        shell: PolicyShellTools,
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
        instructions = load_agents_context(self._repo_root, self._settings.resources)
        if instructions:
            self.context_manager["project_instructions"] = Context(instructions, prefix=True)
        directories = []
        for value in self._settings.resources.skills_dirs:
            path = Path(value).expanduser()
            directories.append(path if path.is_absolute() else self._repo_root / path)
        self.skills.discover_skills_dirs(directories)
        self.skills.activate(list(self._settings.resources.activate_skills))
        return f"Reloaded project instructions and {len(self.skills.activated())} active skills."

    @hidden
    async def close(self) -> None:
        for summarizer in list(getattr(self, "_summarizers", [])):
            summarizer._uninstall()
        memory = getattr(self, "_memory", None)
        if isinstance(memory, MemoryManager):
            memory.uninstall()
        await self.shell.close()

    @hidden
    async def run_task(self, description: str, *, continued: bool = False) -> CodingTaskResult:
        """Run one model task, then enforce host-configured verification."""
        task = description.strip()
        if not task:
            raise ValueError("task must be non-empty")
        self.task = task
        if not continued:
            self.todo.clear()
        if not self.todo.list_todos():
            self.todo.add("Inspect the repository and establish concrete evidence")

        draft = await self._solve_task(task)
        verifications: list[VerificationResult] = []
        status: Literal["completed", "verification_failed", "blocked"] = draft.status
        if draft.status == "completed":
            commands = list(self._settings.verification_commands)
            if not commands and draft.suggested_verification.strip():
                commands = [draft.suggested_verification.strip()]
            if not commands:
                status = "verification_failed"
                draft.evidence += "\nNo verification command was configured or suggested."
            for command in commands:
                result = await self.shell._run_trusted(
                    command,
                    timeout=self._settings.limits.verification_timeout,
                )
                verifications.append(
                    VerificationResult(
                        command=command,
                        passed=result.success,
                        returncode=result.returncode,
                        stdout=result.stdout[-8_000:],
                        stderr=result.stderr[-8_000:],
                    )
                )
                if not result.success:
                    status = "verification_failed"
                    break

        result = CodingTaskResult(
            status=status,
            summary=draft.summary,
            root_cause=draft.root_cause,
            changed_files=draft.changed_files,
            evidence=draft.evidence,
            verifications=verifications,
        )
        self.last_result = result
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
        self.message(result.model_dump_json(indent=2))
        reason = RespondReason.DONE if result.status == "completed" else RespondReason.NEED_INPUT
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

    @strategy(CodeActStrategy(config=_CODEACT_CONFIG))
    async def _solve_task(self, description: str) -> CodingTaskDraft:
        """Solve one coding task in the isolated repository worktree.

        Inspect before editing and preserve unrelated work. Use the supplied
        policy-controlled shell and repository tools. Keep the todo list current.
        Run focused checks while implementing. The deterministic host will run
        configured final verification after this method returns.

        Use `self.message(...)` for useful progress visible to the user. Use
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
    "VerificationResult",
]
