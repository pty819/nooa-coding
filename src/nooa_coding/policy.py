"""Host-enforced approval policy for file and shell tools."""

from __future__ import annotations

import asyncio
import fnmatch
import re
import shlex
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

from nooa import Skill
from nooa.agentdoc import hidden, spec
from nooa.tools.shell_tools import FileWrite, Match, ShellResult, ShellTools
from pydantic import BaseModel

from .config import LimitSettings, PermissionMode, PermissionSettings

ApprovalKind = Literal["file_read", "file_write", "shell", "mcp"]
EventSink = Callable[[str, dict[str, Any]], None]


class ApprovalRequest(BaseModel):
    request_id: str
    kind: ApprovalKind
    resource: str
    reason: str


class ApprovalManager:
    """Own pending approval futures so a UI can answer while a tool is blocked."""

    def __init__(self, event_sink: EventSink) -> None:
        self._event_sink = event_sink
        self._pending: dict[str, tuple[ApprovalRequest, asyncio.Future[bool]]] = {}

    def pending(self) -> list[ApprovalRequest]:
        return [item[0] for item in self._pending.values()]

    async def request(self, kind: ApprovalKind, resource: str, reason: str) -> None:
        request = ApprovalRequest(
            request_id=uuid.uuid4().hex[:12],
            kind=kind,
            resource=resource,
            reason=reason,
        )
        future = asyncio.get_running_loop().create_future()
        self._pending[request.request_id] = (request, future)
        self._event_sink("approval_requested", request.model_dump())
        try:
            allowed = await future
        finally:
            self._pending.pop(request.request_id, None)
        self._event_sink(
            "approval_resolved",
            {**request.model_dump(), "allowed": allowed},
        )
        if not allowed:
            raise PermissionError(f"approval denied for {kind}: {resource}")

    def decide(self, request_id: str, *, allow: bool) -> ApprovalRequest:
        try:
            request, future = self._pending[request_id]
        except KeyError as exc:
            raise KeyError(f"unknown or resolved approval request: {request_id}") from exc
        if not future.done():
            future.set_result(allow)
        return request

    def cancel_all(self) -> None:
        for _, future in self._pending.values():
            if not future.done():
                future.cancel()


class PermissionPolicy:
    def __init__(self, settings: PermissionSettings, approvals: ApprovalManager) -> None:
        self.settings = settings
        self.approvals = approvals

    @staticmethod
    async def _enforce_mode(
        mode: PermissionMode,
        approvals: ApprovalManager,
        kind: ApprovalKind,
        resource: str,
        reason: str,
    ) -> None:
        if mode == "deny":
            raise PermissionError(f"policy denies {kind}: {resource}")
        if mode == "ask":
            await approvals.request(kind, resource, reason)

    async def file_read(self, path: str) -> None:
        await self._enforce_mode(
            self.settings.file_read,
            self.approvals,
            "file_read",
            path,
            "The agent wants to read this file.",
        )

    async def file_write(self, path: str) -> None:
        await self._enforce_mode(
            self.settings.file_write,
            self.approvals,
            "file_write",
            path,
            "The agent wants to create or modify this file.",
        )

    @staticmethod
    def read_only_shell_allowed(command: str) -> bool:
        """Allow only commands whose syntax cannot intentionally mutate the worktree."""
        normalized = " ".join(command.strip().split())
        if re.search(r"(?:&&|\|\||[;\n|<>`]|\$\()", normalized):
            return False
        try:
            tokens = shlex.split(normalized)
        except ValueError:
            return False
        if not tokens:
            return False
        if any(
            token.startswith("/") or ".." in Path(token).parts
            for token in tokens[1:]
            if not token.startswith("-")
        ):
            return False
        if tokens[0] in {"pwd", "ls"}:
            return True
        if tokens[0] == "rg":
            return not any(token.startswith("--pre") for token in tokens[1:])
        if tokens[0] == "find":
            mutating = {"-delete", "-exec", "-execdir", "-fprint", "-fprint0", "-ok", "-okdir"}
            return not any(token in mutating for token in tokens[1:])
        if tokens[0] == "git" and len(tokens) >= 2:
            if tokens[1] not in {"diff", "log", "show", "status"}:
                return False
            dangerous = ("--ext-diff", "--output", "--exec-path")
            return not any(token.startswith(dangerous) for token in tokens[2:])
        return False

    def shell_mode(self, command: str) -> PermissionMode:
        """Classify a command without triggering an approval request."""
        normalized = " ".join(command.strip().split())
        segments = [part.strip() for part in re.split(r"(?:&&|\|\||[;\n|])", normalized)]
        if any(
            fnmatch.fnmatchcase(candidate, pattern)
            for pattern in self.settings.deny_shell
            for candidate in (normalized, *segments)
        ):
            return "deny"
        has_shell_control = bool(re.search(r"(?:&&|\|\||[;\n|<>`]|\$\()", normalized))
        try:
            tokens = shlex.split(normalized)
        except ValueError:
            tokens = []
        escapes_workspace = any(
            token.startswith("/") or ".." in Path(token).parts
            for token in tokens[1:]
            if not token.startswith("-")
        )
        if (
            not has_shell_control
            and not escapes_workspace
            and any(
                fnmatch.fnmatchcase(normalized, pattern) for pattern in self.settings.allow_shell
            )
        ):
            return "allow"
        return self.settings.shell

    async def shell(self, command: str) -> None:
        mode = self.shell_mode(command)
        if mode == "deny":
            raise PermissionError(f"policy denies shell command: {command}")
        await self._enforce_mode(
            mode,
            self.approvals,
            "shell",
            command,
            "The agent wants to execute this shell command.",
        )


class PolicyShellTools(Skill):
    """Workspace-confined ShellTools with host approval and resource limits."""

    def __init__(
        self,
        cwd: str | Path,
        policy: PermissionPolicy,
        limits: LimitSettings,
        event_sink: EventSink,
    ) -> None:
        self.cwd = Path(cwd).resolve()
        self._shell = ShellTools(cwd=str(self.cwd))
        self._policy = policy
        self._limits = limits
        self._event_sink = event_sink
        self._operation_lock = asyncio.Lock()
        self._read_only_depth = 0
        super().__init__()

    @property
    def _session(self) -> Any:
        return self._shell.session

    @hidden
    async def close(self) -> None:
        await self._shell.close()

    @hidden
    @asynccontextmanager
    async def _read_only_scope(self):
        """Prevent a repository-inspection turn from crossing into mutation."""
        self._read_only_depth += 1
        try:
            yield
        finally:
            self._read_only_depth -= 1

    async def run(
        self,
        command: Annotated[str, spec(description="Shell command to execute")],
        *,
        stdin: Annotated[str | None, spec(description="Optional stdin text")] = None,
        timeout: Annotated[float | None, spec(description="Maximum seconds")] = None,
    ) -> ShellResult:
        """Run a policy-approved command inside the isolated worktree."""
        if stdin is not None and len(stdin) > self._limits.max_stdin_chars:
            raise ValueError("stdin exceeds configured max_stdin_chars")
        if self._read_only_depth and not self._policy.read_only_shell_allowed(command):
            raise PermissionError(
                "inspection mode only permits configured read-only shell commands"
            )
        await self._policy.shell(command)
        effective_timeout = min(
            timeout or self._limits.command_timeout, self._limits.command_timeout
        )
        async with self._operation_lock:
            return await self._execute(command, stdin=stdin, timeout=effective_timeout)

    async def _run_trusted(self, command: str, *, timeout: float) -> ShellResult:
        """Run an application-configured verification command without prompting."""
        effective_timeout = min(timeout, self._limits.verification_timeout)
        async with self._operation_lock:
            return await self._execute(
                command,
                stdin=None,
                timeout=effective_timeout,
                announce=False,
            )

    async def _execute(
        self,
        command: str,
        *,
        stdin: str | None,
        timeout: float,
        announce: bool = True,
    ) -> ShellResult:
        if announce:
            self._event_sink("command_started", {"command": command, "timeout": timeout})
        try:
            result = await self._shell.run(command, stdin=stdin, timeout=timeout)
        finally:
            if self._shell.cwd.resolve() != self.cwd:
                await self._shell.run(f"cd {shlex.quote(str(self.cwd))}", timeout=5)
        stdout = result.stdout
        stderr = result.stderr
        if len(stdout) > self._limits.max_output_chars:
            stdout = stdout[: self._limits.max_output_chars] + "\n... (output truncated)"
        if len(stderr) > self._limits.max_output_chars:
            stderr = stderr[: self._limits.max_output_chars] + "\n... (stderr truncated)"
        limited = ShellResult(stdout, stderr, result.returncode, matches=result.matches)
        if announce:
            self._event_sink(
                "command_finished",
                {
                    "command": command,
                    "returncode": limited.returncode,
                    "stdout": limited.stdout,
                    "stderr": limited.stderr,
                },
            )
        return limited

    async def read(
        self,
        path: Annotated[str, spec(description="File path relative to the worktree")],
        lines: Annotated[tuple[int, int] | None, spec(description="Optional line range")] = None,
    ) -> Match:
        """Read a policy-approved file inside the isolated worktree."""
        await self._policy.file_read(path)
        async with self._operation_lock:
            return await self._shell.read(path, lines=lines)

    async def replace(
        self,
        target: Annotated[Any, spec(description="Match or file path")],
        old_or_new: Annotated[str, spec(description="Replacement or old text")] = "",
        new: Annotated[str | None, spec(description="New text for path form")] = None,
    ) -> FileWrite:
        """Edit a file after host approval."""
        path = target.path if isinstance(target, Match) else str(target)
        if self._read_only_depth:
            raise PermissionError("inspection mode cannot modify files")
        await self._policy.file_write(path)
        async with self._operation_lock:
            result = await self._shell.replace(target, old_or_new, new)
        self._event_sink("file_changed", {"path": result.path, "operation": "replace"})
        return result

    async def write_file(
        self,
        path: Annotated[str, spec(description="File path relative to the worktree")],
        content: Annotated[str, spec(description="Full file contents")],
    ) -> FileWrite:
        """Create or overwrite a file after host approval."""
        if len(content) > self._limits.max_stdin_chars:
            raise ValueError("file content exceeds configured max_stdin_chars")
        if self._read_only_depth:
            raise PermissionError("inspection mode cannot modify files")
        await self._policy.file_write(path)
        async with self._operation_lock:
            result = await self._shell.write_file(path, content)
        self._event_sink("file_changed", {"path": result.path, "operation": "write"})
        return result


__all__ = [
    "ApprovalManager",
    "ApprovalRequest",
    "PermissionPolicy",
    "PolicyShellTools",
]
