"""Application-owned MCP client registration and policy around NOOA's public API."""

from __future__ import annotations

import asyncio
import fnmatch
import inspect
import json
import re
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from time import monotonic
from typing import Any, Literal

from nooa.agentdoc import spec
from nooa.mcp import MCPManager
from pydantic import BaseModel, Field

from .config import MCPSettings, PermissionMode
from .policy import ApprovalManager, EventSink

MCPServerState = Literal["disabled", "connected", "failed"]


class MCPServerStatus(BaseModel):
    name: str
    status: MCPServerState
    source: str
    attribute: str | None = None
    tools: list[str] = Field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True)
class _ConfiguredServer:
    name: str
    source: str
    config: dict[str, Any]


def _safe_attribute(server_name: str) -> str:
    normalized = re.sub(r"\W+", "_", server_name, flags=re.UNICODE).strip("_")
    if not normalized:
        normalized = "server"
    if normalized[0].isdigit():
        normalized = f"server_{normalized}"
    return f"mcp_{normalized}"


def _safe_error(exc: BaseException) -> str:
    message = f"{type(exc).__name__}: {exc}"
    message = re.sub(r"(?i)(authorization:\s*(?:bearer|basic)\s+)[^\s,;]+", r"\1***", message)
    message = re.sub(r"(?i)(token|api[_-]?key|secret)=([^\s&,;]+)", r"\1=***", message)
    return message[:2_000]


class _MCPServerProxy:
    """Base for dynamically generated, policy-controlled MCP server capabilities."""

    __nosnapshot__ = True

    def __init__(self, runtime: MCPRuntime, server_name: str, target: Any) -> None:
        self._runtime = runtime
        self._server_name = server_name
        self._target = target


def _make_proxy(runtime: MCPRuntime, server_name: str, target: Any) -> tuple[Any, list[str]]:
    methods: dict[str, Any] = {}
    tool_names: list[str] = []
    target_type = type(target)
    for name, unbound in inspect.getmembers(target_type, inspect.iscoroutinefunction):
        if name.startswith("_"):
            continue
        bound = getattr(target, name, None)
        if not callable(bound):
            continue
        tool_names.append(name)

        def build(tool_name: str, original: Any, original_bound: Any):
            @wraps(original)
            async def invoke(self: _MCPServerProxy, *args: Any, **kwargs: Any) -> Any:
                return await self._runtime.call(
                    self._server_name,
                    tool_name,
                    original_bound,
                    args,
                    kwargs,
                )

            return invoke

        methods[name] = build(name, unbound, bound)

    class_name = "".join(part.capitalize() for part in _safe_attribute(server_name).split("_"))
    proxy_type = type(
        f"{class_name}Tools",
        (_MCPServerProxy,),
        {
            **methods,
            "__doc__": f"Policy-controlled tools from external MCP server {server_name!r}.",
            "__module__": __name__,
        },
    )
    return proxy_type(runtime, server_name, target), sorted(tool_names)


class MCPRuntime:
    """Discover external MCP configs, inject tools, and enforce host policy."""

    __nosnapshot__ = True

    def __init__(
        self,
        repo: str | Path,
        settings: MCPSettings,
        approvals: ApprovalManager,
        event_sink: EventSink,
    ) -> None:
        self.repo = Path(repo).expanduser().resolve()
        self.settings = settings
        self.approvals = approvals
        self._event_sink = event_sink
        self._configured: dict[str, _ConfiguredServer] = {}
        self._statuses: dict[str, MCPServerStatus] = {}
        self._owned_attributes: dict[str, str] = {}
        self._agent: Any | None = None
        self._read_only_depth = 0
        self.config_errors: list[str] = []
        self._discover()

    def _config_path(self, value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else self.repo / path

    def _load_file(self, path: Path) -> dict[str, dict[str, Any]]:
        if not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot parse MCP config {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"MCP config root must be an object: {path}")
        servers = payload.get("mcpServers", {})
        if not isinstance(servers, dict):
            raise ValueError(f"mcpServers must be an object: {path}")
        result: dict[str, dict[str, Any]] = {}
        for name, config in servers.items():
            if not isinstance(name, str) or not name.strip() or not isinstance(config, dict):
                raise ValueError(f"invalid MCP server entry in {path}: {name!r}")
            result[name] = dict(config)
        return result

    def _discover(self) -> None:
        self._configured.clear()
        self.config_errors.clear()
        for value in self.settings.config_files:
            path = self._config_path(value)
            try:
                servers = self._load_file(path)
            except ValueError as exc:
                self.config_errors.append(str(exc))
                continue
            for name, config in servers.items():
                self._configured[name] = _ConfiguredServer(name, str(path), config)
        for name, config in self.settings.servers.items():
            if not name.strip() or not isinstance(config, dict):
                self.config_errors.append(f"invalid inline MCP server entry: {name!r}")
                continue
            self._configured[name] = _ConfiguredServer(name, "settings", dict(config))
        if self.settings.enabled and self.settings.fail_on_error and self.config_errors:
            raise ValueError("; ".join(self.config_errors))

    def _selected(self, name: str) -> bool:
        enabled = any(
            fnmatch.fnmatchcase(name, pattern) for pattern in self.settings.enabled_servers
        )
        disabled = any(
            fnmatch.fnmatchcase(name, pattern) for pattern in self.settings.disabled_servers
        )
        return self.settings.enabled and enabled and not disabled

    def install(self, agent: Any) -> None:
        self._agent = agent
        for name, server in self._configured.items():
            if self._selected(name):
                self._connect(server)
            else:
                self._statuses[name] = MCPServerStatus(
                    name=name,
                    status="disabled",
                    source=server.source,
                )
        agent.context_manager.set_dynamic("mcp_tools", "self._mcp.prompt_status()")

    def _connect(self, server: _ConfiguredServer) -> MCPServerStatus:
        if self._agent is None:
            raise RuntimeError("MCP runtime is not attached to an agent")
        attribute = _safe_attribute(server.name)
        owner = self._owned_attributes.get(server.name)
        if hasattr(self._agent, attribute) and owner != attribute:
            status = MCPServerStatus(
                name=server.name,
                status="failed",
                source=server.source,
                attribute=attribute,
                error=f"agent attribute collision: {attribute}",
            )
            self._statuses[server.name] = status
            self._event_sink(
                "mcp_server_failed",
                {"server": server.name, "error": status.error},
            )
            if self.settings.fail_on_error:
                raise ValueError(status.error)
            return status

        self._event_sink("mcp_server_connecting", {"server": server.name, "source": server.source})
        try:
            target = MCPManager.create_from_server(
                server.name,
                servers={server.name: server.config},
            )
            proxy, tools = _make_proxy(self, server.name, target)
            if not tools:
                raise ValueError("server exposed no callable tools")
            setattr(self._agent, attribute, proxy)
            spec(self._agent, attribute, hidden=False)
            self._owned_attributes[server.name] = attribute
            status = MCPServerStatus(
                name=server.name,
                status="connected",
                source=server.source,
                attribute=attribute,
                tools=tools,
            )
            self._event_sink(
                "mcp_server_connected",
                {"server": server.name, "attribute": attribute, "tools": tools},
            )
        except Exception as exc:
            status = MCPServerStatus(
                name=server.name,
                status="failed",
                source=server.source,
                attribute=attribute,
                error=_safe_error(exc),
            )
            self._event_sink(
                "mcp_server_failed",
                {"server": server.name, "error": status.error},
            )
            if self.settings.fail_on_error:
                raise RuntimeError(f"MCP server {server.name!r} failed: {status.error}") from exc
        self._statuses[server.name] = status
        return status

    def _matches(self, patterns: tuple[str, ...], identifier: str, tool_name: str) -> bool:
        return any(
            fnmatch.fnmatchcase(candidate, pattern)
            for pattern in patterns
            for candidate in (identifier, tool_name)
        )

    async def _authorize(self, server_name: str, tool_name: str) -> None:
        identifier = f"{server_name}.{tool_name}"
        policy = self.settings.permissions
        if self._matches(policy.deny, identifier, tool_name):
            raise PermissionError(f"policy denies MCP tool: {identifier}")
        if self._read_only_depth and not self._matches(policy.read_only, identifier, tool_name):
            raise PermissionError(f"inspection mode denies MCP tool: {identifier}")
        mode: PermissionMode = (
            "allow" if self._matches(policy.allow, identifier, tool_name) else policy.default
        )
        if mode == "deny":
            raise PermissionError(f"policy denies MCP tool: {identifier}")
        if mode == "ask":
            await self.approvals.request(
                "mcp",
                identifier,
                "The agent wants to call this external MCP tool.",
            )

    async def call(
        self,
        server_name: str,
        tool_name: str,
        method: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        await self._authorize(server_name, tool_name)
        started = monotonic()
        self._event_sink(
            "mcp_call_started",
            {
                "server": server_name,
                "tool": tool_name,
                "positional_arguments": len(args),
                "argument_names": sorted(kwargs),
            },
        )
        try:
            result = await asyncio.wait_for(
                method(*args, **kwargs),
                timeout=self.settings.call_timeout,
            )
            limited, output_chars, truncated = self._limit_output(result)
        except Exception as exc:
            self._event_sink(
                "mcp_call_failed",
                {"server": server_name, "tool": tool_name, "error": _safe_error(exc)},
            )
            raise
        self._event_sink(
            "mcp_call_finished",
            {
                "server": server_name,
                "tool": tool_name,
                "duration_ms": round((monotonic() - started) * 1_000, 1),
                "output_chars": output_chars,
                "truncated": truncated,
            },
        )
        return limited

    def _limit_output(self, result: Any) -> tuple[Any, int, bool]:
        if isinstance(result, str):
            rendered = result
        elif isinstance(result, bytes):
            rendered = result.decode(errors="replace")
        else:
            try:
                rendered = json.dumps(result, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                rendered = repr(result)
        size = len(rendered)
        if size <= self.settings.max_output_chars:
            return result, size, False
        suffix = "\n... (MCP output truncated by nooa-coding)"
        return rendered[: self.settings.max_output_chars] + suffix, size, True

    def read_only(self):
        runtime = self

        class Scope:
            async def __aenter__(self) -> None:
                runtime._read_only_depth += 1

            async def __aexit__(self, *_: Any) -> None:
                runtime._read_only_depth -= 1

        return Scope()

    def statuses(self) -> list[MCPServerStatus]:
        return [self._statuses[name].model_copy(deep=True) for name in sorted(self._statuses)]

    def tools(self, server_name: str | None = None) -> dict[str, list[str]]:
        if server_name is not None:
            if server_name not in self._statuses:
                raise KeyError(f"unknown MCP server: {server_name}")
            return {server_name: list(self._statuses[server_name].tools)}
        return {status.name: list(status.tools) for status in self.statuses()}

    def prompt_status(self) -> str:
        connected = [status for status in self.statuses() if status.status == "connected"]
        if not connected:
            return "No external MCP servers are connected."
        lines = ["External MCP capabilities:"]
        for status in connected:
            methods = ", ".join(status.tools)
            lines.append(f"- self.{status.attribute}: {methods}")
        return "\n".join(lines)

    def _remove(self, server_name: str) -> None:
        if self._agent is None:
            return
        attribute = self._owned_attributes.pop(server_name, None)
        if attribute and hasattr(self._agent, attribute):
            spec(self._agent, attribute, hidden=True)
            delattr(self._agent, attribute)

    def enable(self, server_name: str) -> MCPServerStatus:
        try:
            server = self._configured[server_name]
        except KeyError as exc:
            raise KeyError(f"unknown MCP server: {server_name}") from exc
        self._remove(server_name)
        return self._connect(server)

    def disable(self, server_name: str) -> MCPServerStatus:
        try:
            server = self._configured[server_name]
        except KeyError as exc:
            raise KeyError(f"unknown MCP server: {server_name}") from exc
        self._remove(server_name)
        status = MCPServerStatus(name=server_name, status="disabled", source=server.source)
        self._statuses[server_name] = status
        self._event_sink("mcp_server_disabled", {"server": server_name})
        return status

    def reload(self, server_name: str | None = None) -> list[MCPServerStatus]:
        if server_name is not None:
            self._remove(server_name)
            self._statuses.pop(server_name, None)
        else:
            for name in list(self._owned_attributes):
                self._remove(name)
        self._discover()
        if server_name is not None:
            return [self.enable(server_name)]
        self._statuses.clear()
        if self._agent is not None:
            self.install(self._agent)
        return self.statuses()

    def close(self) -> None:
        for name in list(self._owned_attributes):
            self._remove(name)


__all__ = ["MCPRuntime", "MCPServerStatus", "MCPServerState"]
