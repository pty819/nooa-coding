"""Claude Code / Codex compatible hooks system with declarative JSON configuration.

Supports 10 lifecycle events:
PreToolUse, PostToolUse, Stop, Notification, SessionStart,
PermissionRequest, PreCompact, PostCompact, SubagentStart, SubagentStop.

Hooks can execute shell commands with environment variable injection and matcher filtering.

Configuration format (settings.json "hooks" field or .nooa-coding/hooks.json)::

    {
      "hooks": {
        "PreToolUse": [
          {
            "matcher": "Bash",
            "hooks": [
              {"type": "command", "command": "echo $TOOL_INPUT", "timeout": 10}
            ]
          }
        ],
        "PostToolUse": [
          {
            "matcher": "Edit|Write",
            "hooks": [
              {"type": "command", "command": "ruff check $FILE_PATH", "timeout": 30}
            ]
          }
        ]
      }
    }
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import logging
import os
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ─── Event Types ─────────────────────────────────────────────────────────────


class HookEvent(StrEnum):
    """Lifecycle events that can trigger hooks."""

    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    STOP = "Stop"
    NOTIFICATION = "Notification"
    SESSION_START = "SessionStart"
    PERMISSION_REQUEST = "PermissionRequest"
    PRE_COMPACT = "PreCompact"
    POST_COMPACT = "PostCompact"
    SUBAGENT_START = "SubagentStart"
    SUBAGENT_STOP = "SubagentStop"


# ─── Configuration Models ────────────────────────────────────────────────────


class HookAction(BaseModel):
    """A single hook action to execute."""

    type: str = "command"  # "command" is the primary supported type
    command: str = ""
    timeout: int = 30
    status_message: str = Field(default="", alias="statusMessage")

    model_config = {"populate_by_name": True}


class HookRule(BaseModel):
    """A matcher + list of hook actions for one event."""

    matcher: str = "*"  # Glob pattern matched against tool name
    hooks: list[HookAction] = Field(default_factory=list)


class HooksConfig(BaseModel):
    """Top-level hooks configuration."""

    hooks: dict[str, list[HookRule]] = Field(default_factory=dict)

    def rules_for(self, event: HookEvent) -> list[HookRule]:
        """Get all rules registered for an event."""
        return self.hooks.get(event.value, [])


# ─── Hook Result ─────────────────────────────────────────────────────────────


class HookResult(BaseModel):
    """Result of executing a single hook action."""

    command: str
    returncode: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    blocked: bool = False  # True if this hook blocked the tool call
    block_reason: str = ""


class PreToolUseResult(BaseModel):
    """Aggregate result of all PreToolUse hooks for one tool call."""

    allowed: bool = True
    block_reason: str = ""
    hook_results: list[HookResult] = Field(default_factory=list)


# ─── Environment Builder ─────────────────────────────────────────────────────


def _build_env(
    event: HookEvent,
    tool_name: str = "",
    tool_input: str = "",
    file_path: str = "",
    tool_output: str = "",
    session_id: str = "",
    workspace: str = "",
) -> dict[str, str]:
    """Build environment variables passed to hook commands."""
    env = os.environ.copy()
    env.update({
        "HOOK_EVENT": event.value,
        "TOOL_NAME": tool_name,
        "TOOL_INPUT": tool_input[:10000],  # Cap to avoid env overflow
        "FILE_PATH": file_path,
        "TOOL_OUTPUT": tool_output[:10000],
        "SESSION_ID": session_id,
        "WORKSPACE": workspace,
    })
    return env


# ─── Hook Runner ─────────────────────────────────────────────────────────────


class HookRunner:
    """Execute lifecycle hooks based on declarative configuration.

    Usage::

        runner = HookRunner(config, workspace="/path/to/worktree")
        # Before tool execution:
        pre = await runner.trigger_pre_tool_use("Bash", tool_input="rm -rf /")
        if not pre.allowed:
            raise PermissionError(pre.block_reason)
        # After tool execution:
        await runner.trigger_post_tool_use("Bash", tool_input="...", tool_output="...")
    """

    def __init__(
        self,
        config: HooksConfig | None = None,
        *,
        workspace: str | Path = "",
        session_id: str = "",
        event_sink: Any = None,
    ) -> None:
        self._config = config or HooksConfig()
        self._workspace = str(workspace)
        self._session_id = session_id
        self._event_sink = event_sink

    @property
    def config(self) -> HooksConfig:
        return self._config

    def update_config(self, config: HooksConfig) -> None:
        """Hot-reload hook configuration."""
        self._config = config

    # ─── PreToolUse ──────────────────────────────────────────────────────

    async def trigger_pre_tool_use(
        self,
        tool_name: str,
        *,
        tool_input: str = "",
        file_path: str = "",
    ) -> PreToolUseResult:
        """Run all matching PreToolUse hooks. Can block the tool call."""
        rules = self._matching_rules(HookEvent.PRE_TOOL_USE, tool_name)
        if not rules:
            return PreToolUseResult(allowed=True)

        results: list[HookResult] = []
        for rule in rules:
            for action in rule.hooks:
                result = await self._execute_action(
                    action,
                    event=HookEvent.PRE_TOOL_USE,
                    tool_name=tool_name,
                    tool_input=tool_input,
                    file_path=file_path,
                )
                results.append(result)
                # Non-zero exit from PreToolUse hook = block.
                if result.returncode != 0:
                    reason = result.stderr.strip() or result.stdout.strip()
                    if not reason:
                        reason = f"Hook blocked {tool_name} (exit {result.returncode})"
                    if self._event_sink:
                        self._event_sink("hook_blocked", {
                            "tool": tool_name,
                            "command": action.command,
                            "reason": reason,
                        })
                    return PreToolUseResult(
                        allowed=False,
                        block_reason=reason,
                        hook_results=results,
                    )

        return PreToolUseResult(allowed=True, hook_results=results)

    # ─── PostToolUse ─────────────────────────────────────────────────────

    async def trigger_post_tool_use(
        self,
        tool_name: str,
        *,
        tool_input: str = "",
        tool_output: str = "",
        file_path: str = "",
    ) -> list[HookResult]:
        """Run all matching PostToolUse hooks (observe only, cannot block)."""
        rules = self._matching_rules(HookEvent.POST_TOOL_USE, tool_name)
        if not rules:
            return []

        results: list[HookResult] = []
        for rule in rules:
            for action in rule.hooks:
                result = await self._execute_action(
                    action,
                    event=HookEvent.POST_TOOL_USE,
                    tool_name=tool_name,
                    tool_input=tool_input,
                    tool_output=tool_output,
                    file_path=file_path,
                )
                results.append(result)
        return results

    # ─── Stop ────────────────────────────────────────────────────────────

    async def trigger_stop(self, *, summary: str = "") -> list[HookResult]:
        """Run Stop hooks when the agent finishes a response."""
        rules = self._matching_rules(HookEvent.STOP, "*")
        results: list[HookResult] = []
        for rule in rules:
            for action in rule.hooks:
                result = await self._execute_action(
                    action,
                    event=HookEvent.STOP,
                    tool_input=summary,
                )
                results.append(result)
        return results

    # ─── Notification ────────────────────────────────────────────────────

    async def trigger_notification(self, message: str) -> list[HookResult]:
        """Run Notification hooks."""
        rules = self._matching_rules(HookEvent.NOTIFICATION, "*")
        results: list[HookResult] = []
        for rule in rules:
            for action in rule.hooks:
                result = await self._execute_action(
                    action,
                    event=HookEvent.NOTIFICATION,
                    tool_input=message,
                )
                results.append(result)
        return results

    # ─── SessionStart ────────────────────────────────────────────────────

    async def trigger_session_start(self) -> list[HookResult]:
        """Run SessionStart hooks."""
        rules = self._matching_rules(HookEvent.SESSION_START, "*")
        results: list[HookResult] = []
        for rule in rules:
            for action in rule.hooks:
                result = await self._execute_action(
                    action,
                    event=HookEvent.SESSION_START,
                )
                results.append(result)
        return results

    # ─── PermissionRequest ─────────────────────────────────────────────

    async def trigger_permission_request(
        self,
        kind: str,
        resource: str,
        reason: str = "",
    ) -> list[HookResult]:
        """Run PermissionRequest hooks when an approval is requested."""
        rules = self._matching_rules(HookEvent.PERMISSION_REQUEST, kind)
        results: list[HookResult] = []
        for rule in rules:
            for action in rule.hooks:
                result = await self._execute_action(
                    action,
                    event=HookEvent.PERMISSION_REQUEST,
                    tool_name=kind,
                    tool_input=f"{resource}: {reason}",
                    file_path=resource if kind in ("file_write", "file_read") else "",
                )
                results.append(result)
        return results

    # ─── PreCompact / PostCompact ──────────────────────────────────────

    async def trigger_pre_compact(self, *, context_summary: str = "") -> list[HookResult]:
        """Run PreCompact hooks before context compaction."""
        rules = self._matching_rules(HookEvent.PRE_COMPACT, "*")
        results: list[HookResult] = []
        for rule in rules:
            for action in rule.hooks:
                result = await self._execute_action(
                    action,
                    event=HookEvent.PRE_COMPACT,
                    tool_input=context_summary,
                )
                results.append(result)
        return results

    async def trigger_post_compact(self, *, summary: str = "") -> list[HookResult]:
        """Run PostCompact hooks after context compaction."""
        rules = self._matching_rules(HookEvent.POST_COMPACT, "*")
        results: list[HookResult] = []
        for rule in rules:
            for action in rule.hooks:
                result = await self._execute_action(
                    action,
                    event=HookEvent.POST_COMPACT,
                    tool_output=summary,
                )
                results.append(result)
        return results

    # ─── SubagentStart / SubagentStop ──────────────────────────────────

    async def trigger_subagent_start(
        self, worker_id: str, task: str = ""
    ) -> list[HookResult]:
        """Run SubagentStart hooks when a worker agent begins."""
        rules = self._matching_rules(HookEvent.SUBAGENT_START, worker_id)
        results: list[HookResult] = []
        for rule in rules:
            for action in rule.hooks:
                result = await self._execute_action(
                    action,
                    event=HookEvent.SUBAGENT_START,
                    tool_name=worker_id,
                    tool_input=task,
                )
                results.append(result)
        return results

    async def trigger_subagent_stop(
        self, worker_id: str, *, result: str = "", status: str = ""
    ) -> list[HookResult]:
        """Run SubagentStop hooks when a worker agent finishes."""
        rules = self._matching_rules(HookEvent.SUBAGENT_STOP, worker_id)
        results: list[HookResult] = []
        for rule in rules:
            for action in rule.hooks:
                hook_result = await self._execute_action(
                    action,
                    event=HookEvent.SUBAGENT_STOP,
                    tool_name=worker_id,
                    tool_output=f"{status}: {result}",
                )
                results.append(hook_result)
        return results

    # ─── Internal ────────────────────────────────────────────────────────

    def _matching_rules(self, event: HookEvent, tool_name: str) -> list[HookRule]:
        """Filter rules by event and matcher pattern."""
        rules = self._config.rules_for(event)
        matched: list[HookRule] = []
        for rule in rules:
            if self._matches(rule.matcher, tool_name):
                matched.append(rule)
        return matched

    @staticmethod
    def _matches(pattern: str, value: str) -> bool:
        """Check if a value matches a glob/pipe-separated pattern."""
        if pattern in ("*", ""):
            return True
        # Support pipe-separated alternatives: "Bash|Edit|Write"
        for alt in pattern.split("|"):
            alt = alt.strip()
            if fnmatch.fnmatchcase(value, alt):
                return True
        return False

    async def _execute_action(
        self,
        action: HookAction,
        *,
        event: HookEvent,
        tool_name: str = "",
        tool_input: str = "",
        tool_output: str = "",
        file_path: str = "",
    ) -> HookResult:
        """Execute a single hook action as a subprocess."""
        if action.type != "command" or not action.command:
            return HookResult(command=action.command, returncode=0)

        env = _build_env(
            event,
            tool_name=tool_name,
            tool_input=tool_input,
            file_path=file_path,
            tool_output=tool_output,
            session_id=self._session_id,
            workspace=self._workspace,
        )

        if self._event_sink:
            self._event_sink("hook_started", {
                "event": event.value,
                "command": action.command,
                "status_message": action.status_message,
            })

        try:
            proc = await asyncio.create_subprocess_shell(
                action.command,
                cwd=self._workspace or None,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=action.timeout
            )
            result = HookResult(
                command=action.command,
                returncode=proc.returncode or 0,
                stdout=stdout.decode(errors="replace")[:5000],
                stderr=stderr.decode(errors="replace")[:5000],
            )
        except TimeoutError:
            result = HookResult(
                command=action.command,
                returncode=124,
                timed_out=True,
                stderr=f"Hook timed out after {action.timeout}s",
            )
        except Exception as exc:
            result = HookResult(
                command=action.command,
                returncode=126,
                stderr=f"Hook execution error: {exc}",
            )

        if self._event_sink:
            self._event_sink("hook_finished", {
                "event": event.value,
                "command": action.command,
                "returncode": result.returncode,
                "timed_out": result.timed_out,
            })

        return result


# ─── Configuration Loading ───────────────────────────────────────────────────


def load_hooks_config(
    workspace: str | Path = "",
    *,
    settings_hooks: dict[str, Any] | None = None,
) -> HooksConfig:
    """Discover and load hooks configuration from multiple sources.

    Priority (later overrides earlier):
    1. Built-in defaults (empty)
    2. ~/.config/nooa-coding/hooks.json
    3. <workspace>/.nooa-coding/hooks.json
    4. settings.yaml "hooks" field (passed as settings_hooks)
    """
    merged: dict[str, list[dict[str, Any]]] = {}

    sources: list[Path] = []
    user_config = Path.home() / ".config" / "nooa-coding" / "hooks.json"
    if user_config.is_file():
        sources.append(user_config)
    if workspace:
        project_config = Path(workspace) / ".nooa-coding" / "hooks.json"
        if project_config.is_file():
            sources.append(project_config)

    for source in sources:
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
            hooks_data = data.get("hooks", data)
            for event_name, rules in hooks_data.items():
                if event_name not in merged:
                    merged[event_name] = []
                if isinstance(rules, list):
                    merged[event_name].extend(rules)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load hooks from %s: %s", source, exc)

    # Settings-provided hooks override file-based ones.
    if settings_hooks:
        for event_name, rules in settings_hooks.items():
            if isinstance(rules, list):
                merged[event_name] = rules

    return HooksConfig(hooks=merged)


__all__ = [
    "HookAction",
    "HookEvent",
    "HookResult",
    "HookRule",
    "HookRunner",
    "HooksConfig",
    "PreToolUseResult",
    "load_hooks_config",
]
