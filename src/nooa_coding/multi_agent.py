"""Production-grade sub-agent orchestration with full isolation.

Architecture:
    Coordinator (lifecycle manager)
      └── Main Agent (decision maker, only entity that can spawn)
            ├── Sub-Agent A → independent session + worktree
            ├── Sub-Agent B → independent session + worktree
            └── Sub-Agent C → independent session + worktree

Each sub-agent:
- Gets a fresh CodingSession with clean context window
- Operates in an isolated git worktree branched from base_commit
- Has restricted permissions (whitelist shell, no push/merge/rebase)
- Cannot spawn further sub-agents
- Returns a structured WorkerReport on completion

The Coordinator handles:
- Spawning sub-agent sessions via AgentSessionManager
- Timeout watchdog (per-task and global)
- Concurrency limiting (semaphore)
- Result collection (asyncio.Queue)
- Resource cleanup (worktree removal on failure)
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .hooks import HookRunner
    from .session import AgentSession, AgentSessionManager

logger = logging.getLogger(__name__)


# ─── Models ──────────────────────────────────────────────────────────────────


class TaskStatus(StrEnum):
    """Lifecycle status of a delegated sub-task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class TaskPackage(BaseModel):
    """Self-contained task description injected into a sub-agent session.

    The sub-agent starts with a clean context window, so this package must
    contain ALL information needed to complete the task independently.
    """

    task_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    objective: str
    context_summary: str = ""
    file_scope: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    expected_output: str = ""
    base_commit: str = "HEAD"
    token_budget: int = 100_000
    timeout_seconds: float = 600
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    def build_prompt(self) -> str:
        """Construct the enriched prompt injected into the sub-agent session."""
        parts: list[str] = []
        parts.append(f"## Task: {self.objective}")
        if self.context_summary:
            parts.append(f"\n## Background\n{self.context_summary}")
        if self.file_scope:
            scope = "\n".join(f"- {f}" for f in self.file_scope)
            parts.append(f"\n## File Scope\nFocus on these files:\n{scope}")
        if self.constraints:
            cons = "\n".join(f"- {c}" for c in self.constraints)
            parts.append(f"\n## Constraints\n{cons}")
        if self.expected_output:
            parts.append(f"\n## Expected Output\n{self.expected_output}")
        parts.append(
            "\n## Instructions\n"
            "Complete this task autonomously. You have full read/write access to "
            "this isolated worktree. Run tests to verify your changes. "
            "When done, commit your changes with a descriptive message."
        )
        return "\n".join(parts)


class WorkerReport(BaseModel):
    """Structured report returned by a sub-agent upon completion."""

    task_id: str
    status: TaskStatus
    summary: str = ""
    files_changed: list[str] = Field(default_factory=list)
    diff_stat: str = ""
    tests_passed: bool | None = None
    error: str = ""
    commits: list[str] = Field(default_factory=list)
    worktree_path: str = ""
    branch: str = ""
    started_at: str = ""
    completed_at: str = ""
    duration_seconds: float = 0


class OrchestrationPlan(BaseModel):
    """A plan produced by the main agent to decompose a complex objective."""

    objective: str
    tasks: list[TaskPackage] = Field(default_factory=list)
    strategy: str = "parallel"  # "parallel" | "sequential"
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def completed_count(self) -> int:
        return sum(1 for t in self.tasks if t.task_id in self._done_ids)

    @property
    def is_done(self) -> bool:
        return len(self._done_ids) >= len(self.tasks)

    # Internal tracking — not serialized.
    _done_ids: set[str] = set()


# ─── Sub-Agent Handle ────────────────────────────────────────────────────────


class SubAgentHandle:
    """Tracks a spawned sub-agent's lifecycle."""

    def __init__(
        self,
        task: TaskPackage,
        session_id: str,
        worktree_path: str,
        branch: str,
        asyncio_task: asyncio.Task[WorkerReport],
    ) -> None:
        self.task = task
        self.session_id = session_id
        self.worktree_path = worktree_path
        self.branch = branch
        self._asyncio_task = asyncio_task
        self.report: WorkerReport | None = None
        self.started_at = datetime.now(UTC).isoformat()

    @property
    def status(self) -> TaskStatus:
        if self.report is not None:
            return self.report.status
        if self._asyncio_task.done():
            if self._asyncio_task.cancelled():
                return TaskStatus.CANCELLED
            if self._asyncio_task.exception():
                return TaskStatus.FAILED
        return TaskStatus.RUNNING

    @property
    def is_done(self) -> bool:
        return self._asyncio_task.done()

    async def wait(self) -> WorkerReport:
        """Wait for the sub-agent to finish and return its report."""
        self.report = await self._asyncio_task
        return self.report

    def cancel(self) -> None:
        """Request cancellation of the sub-agent."""
        self._asyncio_task.cancel()


# ─── Coordinator ─────────────────────────────────────────────────────────────


class Coordinator:
    """Manages sub-agent lifecycle: spawn, monitor, collect, cleanup.

    The Coordinator is the ONLY entity that can instantiate sub-agent sessions.
    The main agent requests delegation by calling ``spawn()``; the Coordinator
    handles isolation, timeouts, and result routing.

    Usage::

        coordinator = Coordinator(session_manager, parent_session)
        handles = await coordinator.spawn([task_pkg_a, task_pkg_b])
        reports = await coordinator.wait_all(handles)
        # Main agent merges results...
        await coordinator.cleanup(handles)
    """

    def __init__(
        self,
        session_manager: AgentSessionManager,
        parent_session: AgentSession,
        *,
        hook_runner: HookRunner | None = None,
    ) -> None:
        self._manager = session_manager
        self._parent = parent_session
        self._hook_runner = hook_runner
        self._settings = parent_session.settings.subagent
        self._semaphore = asyncio.Semaphore(self._settings.max_concurrent)
        self._results: asyncio.Queue[WorkerReport] = asyncio.Queue()
        self._handles: list[SubAgentHandle] = []
        self._closed = False

    @property
    def handles(self) -> list[SubAgentHandle]:
        return list(self._handles)

    @property
    def active_count(self) -> int:
        return sum(1 for h in self._handles if not h.is_done)

    def spawn(self, tasks: list[TaskPackage]) -> list[SubAgentHandle]:
        """Spawn sub-agent sessions for a batch of task packages.

        Each task gets:
        - An independent AgentSession (clean context)
        - An isolated git worktree branched from task.base_commit
        - Restricted permissions (whitelist shell, no spawn)
        - A timeout watchdog

        Returns handles immediately; execution runs in background.
        """
        if self._closed:
            raise RuntimeError("Coordinator is closed")
        handles: list[SubAgentHandle] = []
        for task in tasks:
            handle = self._spawn_one(task)
            handles.append(handle)
            self._handles.append(handle)
        return handles

    def _spawn_one(self, task: TaskPackage) -> SubAgentHandle:
        """Create one isolated sub-agent session and start execution."""
        session_id = f"sub-{task.task_id}-{uuid.uuid4().hex[:6]}"

        # Create isolated session with its own worktree.
        sub_session = self._manager.create(
            session_id,
            start_ref=task.base_commit,
            parent_session_id=self._parent.session_id,
        )

        # Mark as sub-agent (prevents spawning further sub-agents).
        sub_session._is_sub_agent = True  # noqa: SLF001

        # Apply restricted permissions.
        self._apply_restrictions(sub_session)

        # Build the enriched prompt.
        prompt = task.build_prompt()

        # Create the execution coroutine with timeout + semaphore.
        asyncio_task = asyncio.create_task(
            self._run_sub_agent(task, sub_session, prompt),
            name=f"subagent-{task.task_id}",
        )

        handle = SubAgentHandle(
            task=task,
            session_id=session_id,
            worktree_path=sub_session.metadata.workspace.path,
            branch=sub_session.metadata.workspace.branch,
            asyncio_task=asyncio_task,
        )

        logger.info(
            "Spawned sub-agent %s for task %s (worktree: %s)",
            session_id,
            task.task_id,
            handle.worktree_path,
        )
        return handle

    async def _run_sub_agent(
        self, task: TaskPackage, session: AgentSession, prompt: str
    ) -> WorkerReport:
        """Execute a sub-agent with concurrency limit, timeout, and reporting."""
        started = datetime.now(UTC)
        async with self._semaphore:
            # SubagentStart hook.
            if self._hook_runner:
                with contextlib.suppress(Exception):
                    await self._hook_runner.trigger_subagent_start(
                        task.task_id, task=task.objective
                    )
            try:
                result = await asyncio.wait_for(
                    session.prompt(prompt),
                    timeout=task.timeout_seconds or self._settings.timeout_seconds,
                )
                # Map the actual result status to task status.
                if result.status in ("completed", "answered", "inspected"):
                    task_status = TaskStatus.COMPLETED
                else:
                    task_status = TaskStatus.FAILED
                # Only commit and allow merge for genuinely completed work.
                commit_sha = (
                    self._commit_changes(session, task)
                    if task_status == TaskStatus.COMPLETED
                    else None
                )
                # Build diff info.
                diff_info = self._collect_diff(session)
                completed = datetime.now(UTC)
                report = WorkerReport(
                    task_id=task.task_id,
                    status=task_status,
                    summary=result.summary,
                    error=result.summary if task_status == TaskStatus.FAILED else "",
                    files_changed=diff_info["files"],
                    diff_stat=diff_info["stat"],
                    tests_passed=None,
                    commits=[commit_sha] if commit_sha else [],
                    worktree_path=session.metadata.workspace.path,
                    branch=session.metadata.workspace.branch,
                    started_at=started.isoformat(),
                    completed_at=completed.isoformat(),
                    duration_seconds=(completed - started).total_seconds(),
                )
            except TimeoutError:
                completed = datetime.now(UTC)
                report = WorkerReport(
                    task_id=task.task_id,
                    status=TaskStatus.TIMEOUT,
                    error=f"Sub-agent exceeded {task.timeout_seconds}s timeout",
                    worktree_path=session.metadata.workspace.path,
                    branch=session.metadata.workspace.branch,
                    started_at=started.isoformat(),
                    completed_at=completed.isoformat(),
                    duration_seconds=(completed - started).total_seconds(),
                )
            except asyncio.CancelledError:
                completed = datetime.now(UTC)
                report = WorkerReport(
                    task_id=task.task_id,
                    status=TaskStatus.CANCELLED,
                    error="Sub-agent was cancelled",
                    worktree_path=session.metadata.workspace.path,
                    branch=session.metadata.workspace.branch,
                    started_at=started.isoformat(),
                    completed_at=completed.isoformat(),
                    duration_seconds=(completed - started).total_seconds(),
                )
                raise
            except Exception as exc:
                completed = datetime.now(UTC)
                report = WorkerReport(
                    task_id=task.task_id,
                    status=TaskStatus.FAILED,
                    error=f"{type(exc).__name__}: {exc}",
                    worktree_path=session.metadata.workspace.path,
                    branch=session.metadata.workspace.branch,
                    started_at=started.isoformat(),
                    completed_at=completed.isoformat(),
                    duration_seconds=(completed - started).total_seconds(),
                )
            finally:
                # SubagentStop hook.
                if self._hook_runner:
                    with contextlib.suppress(Exception):
                        await self._hook_runner.trigger_subagent_stop(
                            task.task_id,
                            result=report.summary or report.error,
                            status=report.status.value,
                        )
                # Route result to the collection queue.
                await self._results.put(report)

        return report

    async def wait_all(
        self,
        handles: list[SubAgentHandle] | None = None,
        *,
        on_progress: Any = None,
    ) -> list[WorkerReport]:
        """Wait for all spawned sub-agents to complete.

        Args:
            handles: Specific handles to wait for. Defaults to all.
            on_progress: Optional callback(report) called as each completes.

        Returns:
            List of WorkerReports in completion order.
        """
        handles = handles or self._handles
        if not handles:
            return []

        reports: list[WorkerReport] = []
        pending = {h for h in handles if not h.is_done}

        # Collect from the results queue as they arrive.
        while len(reports) < len(handles):
            # Check if all remaining are already done.
            newly_done = [h for h in pending if h.is_done]
            for h in newly_done:
                pending.discard(h)
                if h.report is None:
                    with contextlib.suppress(Exception):
                        h.report = h._asyncio_task.result()
                if h.report and h.report not in reports:
                    reports.append(h.report)
                    if on_progress:
                        on_progress(h.report)

            if len(reports) >= len(handles):
                break

            # Wait for the next result from the queue.
            try:
                report = await asyncio.wait_for(self._results.get(), timeout=1.0)
                if report.task_id in {h.task.task_id for h in handles}:
                    if report not in reports:
                        reports.append(report)
                        # Update the handle's report.
                        for h in handles:
                            if h.task.task_id == report.task_id:
                                h.report = report
                        if on_progress:
                            on_progress(report)
            except TimeoutError:
                continue

        return reports

    async def cleanup(self, handles: list[SubAgentHandle] | None = None) -> None:
        """Clean up sub-agent resources (worktrees for failed/cancelled tasks)."""
        handles = handles or self._handles
        for handle in handles:
            # Only remove worktrees for tasks that didn't complete successfully.
            if handle.report and handle.report.status in (
                TaskStatus.FAILED,
                TaskStatus.TIMEOUT,
                TaskStatus.CANCELLED,
            ):
                self._remove_worktree(handle)
        self._closed = True

    def _remove_worktree(self, handle: SubAgentHandle) -> None:
        """Remove a sub-agent's worktree (best-effort)."""
        import subprocess

        worktree = Path(handle.worktree_path)
        if not worktree.exists():
            return
        with contextlib.suppress(Exception):
            subprocess.run(
                ["git", "-C", str(self._manager.repo), "worktree", "remove", "--force", str(worktree)],
                capture_output=True,
                timeout=30,
            )
        with contextlib.suppress(Exception):
            subprocess.run(
                ["git", "-C", str(self._manager.repo), "branch", "-D", handle.branch],
                capture_output=True,
                timeout=10,
            )

    def _apply_restrictions(self, session: AgentSession) -> None:
        """Apply restricted permissions to a sub-agent session."""
        from .config import PermissionSettings

        restricted = PermissionSettings(
            file_read="allow",
            file_write="allow",  # Within its own worktree.
            shell="allow",  # Only whitelisted commands pass.
            allow_shell=self._settings.allow_shell,
            deny_shell=self._settings.deny_shell,
        )
        # Replace the policy settings on the session's shell tools.
        policy = session.agent.shell._policy  # noqa: SLF001
        policy.settings = restricted

    @staticmethod
    def _commit_changes(session: AgentSession, task: TaskPackage) -> str | None:
        """Commit all changes in the sub-agent worktree. Returns commit SHA."""
        import subprocess

        worktree = Path(session.metadata.workspace.path)
        # Check if there are changes.
        status = subprocess.run(
            ["git", "-C", str(worktree), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if not status.stdout.strip():
            return None  # No changes to commit.

        subprocess.run(
            ["git", "-C", str(worktree), "add", "-A"],
            capture_output=True,
            timeout=10,
        )
        result = subprocess.run(
            [
                "git", "-C", str(worktree),
                "-c", "user.name=NOOA Sub-Agent",
                "-c", "user.email=subagent@nooa-coding.local",
                "commit",
                "-m", f"sub-agent: {task.objective[:72]}",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return None
        sha = subprocess.run(
            ["git", "-C", str(worktree), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return sha.stdout.strip()

    @staticmethod
    def _collect_diff(session: AgentSession) -> dict[str, Any]:
        """Collect diff information from the sub-agent worktree."""
        import subprocess

        worktree = Path(session.metadata.workspace.path)
        stat = subprocess.run(
            ["git", "-C", str(worktree), "diff", "--stat", "HEAD~1", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        files_result = subprocess.run(
            ["git", "-C", str(worktree), "diff", "--name-only", "HEAD~1", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        files = [f for f in files_result.stdout.strip().splitlines() if f]
        return {"files": files, "stat": stat.stdout.strip()}

    def status(self) -> dict[str, Any]:
        """Return current coordination status."""
        return {
            "active": self.active_count,
            "total": len(self._handles),
            "results_queued": self._results.qsize(),
            "closed": self._closed,
            "handles": [
                {
                    "task_id": h.task.task_id,
                    "objective": h.task.objective[:60],
                    "status": h.status.value,
                    "session_id": h.session_id,
                }
                for h in self._handles
            ],
        }


# ─── Worktree Merge ──────────────────────────────────────────────────────────


class MergeResult(BaseModel):
    """Result of merging sub-agent worktrees into the main session."""

    merged: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    failed: list[str] = Field(default_factory=list)
    success: bool = True


def merge_worktrees(
    main_worktree: str | Path,
    reports: list[WorkerReport],
) -> MergeResult:
    """Cherry-pick sub-agent commits into the main worktree.

    Strategy: for each completed report with commits, cherry-pick those
    commits into the main worktree. On conflict, abort and record.
    """
    import subprocess

    main = Path(main_worktree)
    result = MergeResult()

    for report in reports:
        if report.status != TaskStatus.COMPLETED or not report.commits:
            if report.status in (TaskStatus.FAILED, TaskStatus.TIMEOUT):
                result.failed.append(report.task_id)
            continue

        for commit_sha in report.commits:
            # Cherry-pick the sub-agent commit into main worktree.
            pick = subprocess.run(
                ["git", "-C", str(main), "cherry-pick", "--no-commit", commit_sha],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if pick.returncode != 0:
                # Conflict — abort and record.
                subprocess.run(
                    ["git", "-C", str(main), "cherry-pick", "--abort"],
                    capture_output=True,
                    timeout=10,
                )
                result.conflicts.append(f"{report.task_id}:{commit_sha[:8]}")
                result.success = False
            else:
                # Commit the cherry-picked changes.
                subprocess.run(
                    [
                        "git", "-C", str(main),
                        "-c", "user.name=NOOA Coding Agent",
                        "-c", "user.email=nooa-coding@localhost",
                        "commit",
                        "-m", f"merge sub-agent {report.task_id}: {report.summary[:60]}",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                result.merged.append(report.task_id)

    return result


__all__ = [
    "Coordinator",
    "MergeResult",
    "OrchestrationPlan",
    "SubAgentHandle",
    "TaskPackage",
    "TaskStatus",
    "WorkerReport",
    "merge_worktrees",
]
