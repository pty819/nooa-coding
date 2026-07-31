"""Multi-agent orchestration: decompose tasks and coordinate workers."""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class TaskStatus(StrEnum):
    """Status of a delegated sub-task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkerMessage(BaseModel):
    """Message passed between orchestrator and workers."""

    message_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    sender: str
    recipient: str
    kind: str  # "task", "result", "progress", "error", "cancel"
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class SubTask(BaseModel):
    """A unit of work delegated to a worker agent."""

    task_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    description: str
    status: TaskStatus = TaskStatus.PENDING
    result: str = ""
    error: str = ""
    worker_id: str = ""
    started_at: str | None = None
    completed_at: str | None = None


class OrchestrationPlan(BaseModel):
    """A plan produced by the orchestrator to decompose a complex task."""

    objective: str
    subtasks: list[SubTask] = Field(default_factory=list)
    strategy: str = "sequential"  # "sequential" | "parallel" | "pipeline"
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def completed_count(self) -> int:
        return sum(1 for t in self.subtasks if t.status == TaskStatus.COMPLETED)

    @property
    def failed_count(self) -> int:
        return sum(1 for t in self.subtasks if t.status == TaskStatus.FAILED)

    @property
    def is_done(self) -> bool:
        return all(
            t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)
            for t in self.subtasks
        )


class WorkerAgent:
    """A lightweight worker that executes a single sub-task."""

    def __init__(self, worker_id: str, session: Any) -> None:
        self.worker_id = worker_id
        self._session = session
        self._inbox: asyncio.Queue[WorkerMessage] = asyncio.Queue()
        self._running = False

    async def execute(self, task: SubTask) -> SubTask:
        """Execute a sub-task using the session's prompt mechanism."""
        task.status = TaskStatus.RUNNING
        task.worker_id = self.worker_id
        task.started_at = datetime.now(UTC).isoformat()
        self._running = True

        try:
            result = await self._session.prompt(task.description)
            task.status = TaskStatus.COMPLETED
            task.result = result.summary
        except asyncio.CancelledError:
            task.status = TaskStatus.CANCELLED
            task.error = "Task was cancelled"
            raise
        except Exception as exc:
            task.status = TaskStatus.FAILED
            task.error = str(exc)
        finally:
            self._running = False
            task.completed_at = datetime.now(UTC).isoformat()

        return task

    def cancel(self) -> None:
        """Signal the worker to stop."""
        self._running = False


class Orchestrator:
    """Coordinate multiple worker agents to complete a complex objective.

    Usage::

        orchestrator = Orchestrator(session)
        plan = await orchestrator.decompose("Refactor auth module and add tests")
        results = await orchestrator.execute_plan(plan)
    """

    def __init__(self, session: Any, *, max_workers: int = 3) -> None:
        self._session = session
        self._max_workers = max_workers
        self._workers: dict[str, WorkerAgent] = {}
        self._message_log: list[WorkerMessage] = []
        self._plan: OrchestrationPlan | None = None

    @property
    def plan(self) -> OrchestrationPlan | None:
        return self._plan

    @property
    def message_log(self) -> list[WorkerMessage]:
        return list(self._message_log)

    def _log_message(self, msg: WorkerMessage) -> None:
        self._message_log.append(msg)

    async def decompose(self, objective: str) -> OrchestrationPlan:
        """Use the LLM to decompose a complex task into sub-tasks."""
        decompose_prompt = (
            "You are a task orchestrator. Decompose the following objective into "
            "2-5 concrete, independent sub-tasks that can be executed by worker agents.\n\n"
            "Rules:\n"
            "- Each sub-task must be self-contained and actionable.\n"
            "- Order sub-tasks by dependency (earlier tasks should not depend on later ones).\n"
            "- Keep each sub-task focused on ONE logical change.\n\n"
            "Respond in this EXACT format (one task per line):\n"
            "TASK: <description>\n"
            "TASK: <description>\n"
            "...\n\n"
            f"OBJECTIVE: {objective}"
        )

        messages = [{"role": "user", "content": decompose_prompt}]
        response = await self._session.llm.acall(messages)
        content = response.content or ""

        # Track token usage.
        if hasattr(self._session, "_track_direct_llm_call"):
            self._session._track_direct_llm_call(response)

        subtasks: list[SubTask] = []
        for line in content.splitlines():
            line = line.strip()
            if line.upper().startswith("TASK:"):
                desc = line[5:].strip()
                if desc:
                    subtasks.append(SubTask(description=desc))

        if not subtasks:
            # Fallback: single task.
            subtasks = [SubTask(description=objective)]

        strategy = "parallel" if len(subtasks) > 1 else "sequential"
        self._plan = OrchestrationPlan(
            objective=objective, subtasks=subtasks, strategy=strategy
        )

        self._log_message(
            WorkerMessage(
                sender="orchestrator",
                recipient="all",
                kind="plan",
                payload={"objective": objective, "subtask_count": len(subtasks)},
            )
        )

        return self._plan

    async def execute_plan(
        self,
        plan: OrchestrationPlan | None = None,
        *,
        on_progress: Any = None,
    ) -> OrchestrationPlan:
        """Execute all sub-tasks in the plan.

        Args:
            plan: The plan to execute. Uses self._plan if not provided.
            on_progress: Optional callback(task: SubTask) called after each task.

        Returns:
            The plan with updated task statuses.
        """
        plan = plan or self._plan
        if plan is None:
            raise ValueError("No plan to execute. Call decompose() first.")

        if plan.strategy == "parallel":
            await self._execute_parallel(plan, on_progress)
        else:
            await self._execute_sequential(plan, on_progress)

        return plan

    async def _execute_sequential(
        self, plan: OrchestrationPlan, on_progress: Any
    ) -> None:
        """Execute sub-tasks one at a time."""
        worker = self._get_or_create_worker("worker-0")

        for task in plan.subtasks:
            if task.status != TaskStatus.PENDING:
                continue

            self._log_message(
                WorkerMessage(
                    sender="orchestrator",
                    recipient=worker.worker_id,
                    kind="task",
                    payload={"task_id": task.task_id, "description": task.description},
                )
            )

            await worker.execute(task)

            self._log_message(
                WorkerMessage(
                    sender=worker.worker_id,
                    recipient="orchestrator",
                    kind="result",
                    payload={
                        "task_id": task.task_id,
                        "status": task.status.value,
                        "result": task.result[:500],
                    },
                )
            )

            if on_progress:
                on_progress(task)

            # Stop on failure in sequential mode.
            if task.status == TaskStatus.FAILED:
                for remaining in plan.subtasks:
                    if remaining.status == TaskStatus.PENDING:
                        remaining.status = TaskStatus.CANCELLED
                break

    async def _execute_parallel(
        self, plan: OrchestrationPlan, on_progress: Any
    ) -> None:
        """Execute independent sub-tasks concurrently."""
        semaphore = asyncio.Semaphore(self._max_workers)
        pending_tasks = [t for t in plan.subtasks if t.status == TaskStatus.PENDING]

        async def run_with_limit(task: SubTask, worker: WorkerAgent) -> None:
            async with semaphore:
                await worker.execute(task)
                if on_progress:
                    on_progress(task)

        workers = [
            self._get_or_create_worker(f"worker-{i}")
            for i in range(min(len(pending_tasks), self._max_workers))
        ]

        coros = [
            run_with_limit(task, workers[i % len(workers)])
            for i, task in enumerate(pending_tasks)
        ]
        await asyncio.gather(*coros, return_exceptions=True)

    def _get_or_create_worker(self, worker_id: str) -> WorkerAgent:
        if worker_id not in self._workers:
            self._workers[worker_id] = WorkerAgent(worker_id, self._session)
        return self._workers[worker_id]

    def status(self) -> dict[str, Any]:
        """Return current orchestration status."""
        if self._plan is None:
            return {"status": "idle", "plan": None}
        return {
            "status": "completed" if self._plan.is_done else "running",
            "objective": self._plan.objective,
            "strategy": self._plan.strategy,
            "total": len(self._plan.subtasks),
            "completed": self._plan.completed_count,
            "failed": self._plan.failed_count,
            "workers": len(self._workers),
            "messages": len(self._message_log),
        }


__all__ = [
    "OrchestrationPlan",
    "Orchestrator",
    "SubTask",
    "TaskStatus",
    "WorkerAgent",
    "WorkerMessage",
]
