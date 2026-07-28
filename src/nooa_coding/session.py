"""Formal local AgentSession API with persistence, streaming, and recovery."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import re
import sqlite3
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from nooa.interactive import AgentMessage
from nooa.storage import SQLiteStorageManager
from nooa.unifiedllm import UnifiedLLM
from pydantic import BaseModel, Field

from .agent import CodingAgent, CodingTaskResult
from .config import CodingSettings, load_settings
from .events import SessionEvent, SessionEventKind
from .llm import build_llm
from .policy import ApprovalManager, PermissionPolicy, PolicyShellTools
from .workspace import Checkpoint, DiffResult, WorkspaceInfo, WorkspaceManager

SessionStatus = Literal["idle", "running", "cancelling", "cancelled", "failed"]
_SESSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


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

        self.approvals = ApprovalManager(self._policy_event)
        policy = PermissionPolicy(settings.permissions, self.approvals)
        shell = PolicyShellTools(
            metadata.workspace.path,
            policy,
            settings.limits,
            self._tool_event,
        )
        self.llm = llm or build_llm(settings.models, on_failover=self._model_failover)
        self.storage = SQLiteStorageManager(self.database_path)
        self.agent = CodingAgent(
            llm=self.llm,
            repo=metadata.workspace.path,
            settings=settings,
            shell=shell,
            event_sink=self._tool_event,
            storage=self.storage,
        )
        self.agent.event_manager.register_event_type(AgentMessage)
        self._unsubscribe = self.agent.event_manager.on("*", self._on_nooa_event)

        if restore:
            self.storage.restore_latest_snapshot(self.agent)
        self._persist_metadata()

    @property
    def session_id(self) -> str:
        return self.metadata.session_id

    @property
    def workspace(self) -> Path:
        return Path(self.metadata.workspace.path)

    @property
    def status(self) -> SessionStatus:
        return self.metadata.status

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

    def _on_nooa_event(self, event: Any) -> None:
        try:
            payload = event.model_dump(mode="json")
        except Exception:
            payload = {"text": str(event)}
        kind: SessionEventKind = "message" if isinstance(event, AgentMessage) else "agent"
        self._emit(kind, getattr(event, "event_type", type(event).__name__), payload)

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
        async with self._run_lock:
            if self._active_task is not None:
                raise RuntimeError("session already has an active turn")
            run_id = uuid.uuid4().hex
            self._set_status("running")
            self._save_snapshot()
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
                raise
            else:
                self._set_status("idle")
                self._emit(
                    "session",
                    "turn_finished",
                    {"run_id": run_id, "result": result.model_dump(mode="json")},
                )
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

    async def compact(self, *, preserve_recent: int | None = None) -> str | None:
        if self._active_task is not None:
            raise RuntimeError("cannot compact while a turn is running")
        tag = await self.agent.compact_history(preserve_recent=preserve_recent)
        self._save_snapshot()
        self._emit("session", "history_compacted", {"summary_tag": tag})
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


__all__ = [
    "AgentSession",
    "AgentSessionManager",
    "SessionMetadata",
    "SessionStatus",
    "SessionSummary",
]
