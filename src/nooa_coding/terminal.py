"""Human-oriented Rich rendering for the interactive terminal client."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console, Group, RenderableType
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .agent import CodingTaskResult
from .events import SessionEvent, TokenUsage

if TYPE_CHECKING:
    from .mcp import MCPServerStatus
    from .policy import ApprovalRequest
    from .session import AgentSession


_ANSI = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
_VISIBLE_ANSI = re.compile(r"(?:\?|\\x1b)\[[0-9;?]*[ -/]*[@-~]")


def clean_terminal_text(value: object) -> str:
    """Remove provider-supplied or already-mangled terminal control sequences."""
    text = str(value)
    return _VISIBLE_ANSI.sub("", _ANSI.sub("", text))


class TerminalRenderer:
    """Keep durable internal events separate from the user's conversational UI."""

    def __init__(self, console: Console) -> None:
        self.console = console

    def banner(self, session_id: str, model: str, workspace: Path) -> None:
        details = Table.grid(padding=(0, 1))
        details.add_column(style="dim", no_wrap=True)
        details.add_column()
        details.add_row("Session", session_id)
        details.add_row("Model", model)
        details.add_row("Worktree", str(workspace))
        self.console.print(Panel(details, title="NOOA Coding Agent", border_style="cyan"))
        self.console.print("Type a request, or use [bold]/help[/bold] for commands.")

    def help(self) -> None:
        rows = (
            ("/help", "Show this command list"),
            ("/diff", "Show worktree status and patch"),
            ("/checkpoint [label]", "Save a recoverable checkpoint"),
            ("/rollback CHECKPOINT", "Restore files and agent state"),
            ("/compact", "Compress older conversation history"),
            ("/fork [SESSION_ID]", "Fork the conversation and worktree"),
            ("/approvals", "List pending tool approvals"),
            ("/approve REQUEST_ID", "Approve one pending operation"),
            ("/deny REQUEST_ID", "Deny one pending operation"),
            ("/replay [N]", "Show recent durable events"),
            ("/reload", "Reload AGENTS.md and skills"),
            ("/mcp list", "Show external MCP server status"),
            ("/mcp tools [SERVER]", "List injected MCP tools"),
            ("/mcp enable|disable SERVER", "Connect or disconnect an MCP server"),
            ("/mcp reload [SERVER]", "Reload MCP configuration and tools"),
            ("/cost", "Show token usage and estimated cost"),
            ("/status", "Show session status panel"),
            ("/clear", "Clear conversation history"),
            ("/model [NAME]", "Show or switch the active model"),
            ("/permissions [MODE]", "Show or switch permission mode (allow/ask/default)"),
            ("/plan TASK", "Generate an implementation plan without executing"),
            ("/goal OBJECTIVE", "Set a goal; agent auto-continues until achieved"),
            ("/goal clear", "Clear the active goal"),
            ("/review", "LLM-powered code review of the current diff"),
            ("/search QUERY", "Web search and return results"),
            ("/export", "Export session events to JSONL"),
            ("/workspace", "Print the isolated worktree path"),
            ("/exit", "Save and exit"),
        )
        table = Table(title="Commands", box=None, show_header=False, pad_edge=False)
        table.add_column(style="bold cyan", no_wrap=True)
        table.add_column()
        for command, description in rows:
            table.add_row(command, description)
        self.console.print(table)
        self.console.print(
            Text(
                "While a turn runs: /cancel, /approve ID, /deny ID, or type a follow-up.",
                style="dim",
            )
        )
        self.console.print(
            Text(
                "Input: Ctrl+J newline, Ctrl+G external editor, Tab completion.",
                style="dim",
            )
        )

    def event(self, event: SessionEvent) -> None:
        if event.kind == "thinking":
            if event.name == "llm_call_started":
                method = event.data.get("method", "")
                turn = event.data.get("turn", 0)
                label = f" {method}" if method else ""
                self.console.print(
                    Text(f"⟳ thinking{label} (turn {turn})…", style="dim italic")
                )
            elif event.name == "llm_call_ended":
                success = event.data.get("success", True)
                if not success:
                    self.console.print(Text("✗ LLM call failed", style="red"))
            return
        if event.kind == "usage":
            # Token usage events are tracked silently; shown via /cost.
            return
        if event.kind == "message":
            content = event.data.get("content")
            if content:
                self.console.print(Markdown(clean_terminal_text(content)))
            return
        if event.name == "approval_requested":
            request_id = clean_terminal_text(event.data.get("request_id", ""))
            kind = clean_terminal_text(event.data.get("kind", "operation"))
            resource = clean_terminal_text(event.data.get("resource", ""))
            body = Group(
                Text(f"{kind}: {resource}"),
                Text(f"Use /approve {request_id} or /deny {request_id}", style="dim"),
            )
            self.console.print(
                Panel(body, title=f"Approval required · {request_id}", border_style="yellow")
            )
            return
        if event.name == "approval_resolved":
            decision = "approved" if event.data.get("allowed") else "denied"
            self.console.print(Text(f"Operation {decision}.", style="dim"))
            return
        if event.name == "command_started":
            command = clean_terminal_text(event.data.get("command", ""))
            self.console.print(Text(f"$ {command}", style="cyan"))
            return
        if event.name == "command_finished":
            code = int(event.data.get("returncode", 1))
            if code == 0:
                self.console.print(Text("✓ command finished", style="dim green"))
            else:
                stderr = clean_terminal_text(event.data.get("stderr", "")).strip()
                tail = "\n".join(stderr.splitlines()[-8:])
                self.console.print(
                    Panel(
                        Text(tail or f"Command exited with status {code}"),
                        title=f"Command failed · exit {code}",
                        border_style="red",
                    )
                )
            return
        if event.name == "file_changed":
            path = clean_terminal_text(event.data.get("path", ""))
            self.console.print(Text(f"Changed {path}", style="green"))
            return
        if event.name == "mcp_server_connected":
            server = clean_terminal_text(event.data.get("server", ""))
            tools = len(event.data.get("tools", []))
            self.console.print(
                Text(f"Connected MCP server {server} ({tools} tools)", style="green")
            )
            return
        if event.name == "mcp_server_failed":
            server = clean_terminal_text(event.data.get("server", ""))
            error = clean_terminal_text(event.data.get("error", ""))
            self.console.print(
                Panel(error, title=f"MCP server failed · {server}", border_style="yellow")
            )
            return
        if event.name == "mcp_call_started":
            server = clean_terminal_text(event.data.get("server", ""))
            tool = clean_terminal_text(event.data.get("tool", ""))
            self.console.print(Text(f"MCP {server}.{tool}", style="magenta"))
            return
        if event.name == "mcp_call_finished":
            elapsed = event.data.get("duration_ms", 0)
            truncated = " · output truncated" if event.data.get("truncated") else ""
            self.console.print(
                Text(f"✓ MCP call finished in {elapsed} ms{truncated}", style="dim green")
            )
            return
        if event.name == "mcp_call_failed":
            server = clean_terminal_text(event.data.get("server", ""))
            tool = clean_terminal_text(event.data.get("tool", ""))
            error = clean_terminal_text(event.data.get("error", ""))
            self.console.print(
                Panel(error, title=f"MCP call failed · {server}.{tool}", border_style="red")
            )
            return
        if event.name == "cell_guard_blocked":
            reason = clean_terminal_text(event.data.get("reason", ""))
            self.console.print(Text(f"Generated action blocked: {reason}", style="yellow"))
            return
        if event.kind == "model_failover":
            source = clean_terminal_text(event.data.get("from", ""))
            target = clean_terminal_text(event.data.get("to", ""))
            self.console.print(
                Panel(f"{source} → {target}", title="Model failover", border_style="yellow")
            )
            return
        if event.kind == "error":
            message = clean_terminal_text(event.data.get("message", event.name))
            self.console.print(Panel(message, title="Turn failed", border_style="red"))

    def result(self, result: CodingTaskResult) -> None:
        summary = clean_terminal_text(result.summary)
        if result.mode == "conversation":
            self.console.print(Markdown(summary))
            return

        color = {
            "completed": "green",
            "inspected": "cyan",
            "blocked": "yellow",
            "verification_failed": "red",
        }.get(result.status, "white")
        sections: list[RenderableType] = [Markdown(summary)]
        if result.changed_files:
            sections.extend(
                [
                    Text("Changed files", style="bold"),
                    Text("\n".join(f"• {path}" for path in result.changed_files)),
                ]
            )
        if result.verifications:
            checks = Table(box=None, show_header=False, pad_edge=False)
            checks.add_column(width=2)
            checks.add_column()
            for check in result.verifications:
                checks.add_row("✓" if check.passed else "✗", check.command)
            sections.extend([Text("Verification", style="bold"), checks])
        evidence = clean_terminal_text(result.evidence).strip()
        if evidence:
            sections.extend([Text("Evidence", style="bold"), Text(evidence)])
        self.console.print(
            Panel(
                Group(*sections),
                title=f"{result.mode} · {result.status}",
                border_style=color,
            )
        )

    def approvals(self, pending: list[ApprovalRequest]) -> None:
        if not pending:
            self.console.print(Text("No pending approvals.", style="dim"))
            return
        table = Table("ID", "Kind", "Resource", title="Pending approvals")
        for item in pending:
            table.add_row(item.request_id, item.kind, clean_terminal_text(item.resource))
        self.console.print(table)

    def token_usage(self, usage: TokenUsage) -> None:
        """Render cumulative token usage and cost."""
        if usage.llm_calls == 0:
            self.console.print(Text("No LLM calls yet.", style="dim"))
            return
        table = Table(title="Token Usage", box=None, show_header=False, pad_edge=False)
        table.add_column(style="bold", no_wrap=True)
        table.add_column(justify="right")
        table.add_row("LLM calls", str(usage.llm_calls))
        table.add_row("Prompt tokens", f"{usage.total_prompt_tokens:,}")
        table.add_row("Completion tokens", f"{usage.total_completion_tokens:,}")
        if usage.total_cached_tokens:
            table.add_row("Cached tokens", f"{usage.total_cached_tokens:,}")
        if usage.total_reasoning_tokens:
            table.add_row("Reasoning tokens", f"{usage.total_reasoning_tokens:,}")
        table.add_row("Total tokens", f"{usage.total_tokens:,}")
        if usage.total_cost_usd > 0:
            table.add_row("Est. cost", f"${usage.total_cost_usd:.4f}")
        self.console.print(table)

    def status(self, session: AgentSession) -> None:
        """Render a comprehensive status panel."""
        usage = session.token_usage
        details = Table.grid(padding=(0, 2))
        details.add_column(style="bold", no_wrap=True)
        details.add_column()
        details.add_row("Session", session.session_id)
        details.add_row("Model", session.current_model)
        details.add_row("Status", session.status)
        details.add_row("Worktree", str(session.workspace))
        details.add_row("LLM calls", str(usage.llm_calls))
        details.add_row("Tokens", f"{usage.total_tokens:,}")
        if usage.total_cost_usd > 0:
            details.add_row("Est. cost", f"${usage.total_cost_usd:.4f}")
        pending = session.approvals.pending()
        details.add_row("Pending approvals", str(len(pending)))
        mcp_connected = sum(
            1 for s in session.mcp_status() if s.status == "connected"
        )
        details.add_row("MCP servers", str(mcp_connected))
        self.console.print(Panel(details, title="Session Status", border_style="cyan"))

    def mcp_status(self, statuses: list[MCPServerStatus], errors: list[str]) -> None:
        if statuses:
            table = Table("Server", "Status", "Agent attribute", "Tools", "Source", title="MCP")
            for item in statuses:
                table.add_row(
                    item.name,
                    item.status,
                    item.attribute or "—",
                    str(len(item.tools)),
                    item.source,
                )
                if item.error:
                    table.add_row("", Text(item.error, style="yellow"), "", "", "")
            self.console.print(table)
        else:
            self.console.print(Text("No MCP servers are configured.", style="dim"))
        for error in errors:
            self.console.print(Text(error, style="yellow"))

    def mcp_tools(self, tools: dict[str, list[str]]) -> None:
        if not tools:
            self.console.print(Text("No MCP tools are connected.", style="dim"))
            return
        table = Table("Server", "Tools", title="External MCP tools")
        for server, names in tools.items():
            table.add_row(server, "\n".join(names) if names else "—")
        self.console.print(table)


__all__ = ["TerminalRenderer", "clean_terminal_text"]
