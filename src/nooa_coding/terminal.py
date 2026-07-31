"""Human-oriented Rich rendering for the interactive terminal client."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console, Group, RenderableType
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from .agent import CodingTaskResult
from .events import SessionEvent, TokenUsage

if TYPE_CHECKING:
    from .mcp import MCPServerStatus
    from .policy import ApprovalRequest
    from .session import AgentSession

_VERSION = "0.1.0"

_LOGO = r"""
[bold cyan]  ╔══════════════════════════════════════╗
  ║[/bold cyan]  [bold white]NOOA[/bold white] [dim]Coding Agent[/dim]  [green]v{version}[/green]          [bold cyan]║
  ║[/bold cyan]  [dim]NVIDIA OO-Agents · local · isolated[/dim] [bold cyan]║
  ╚══════════════════════════════════════╝[/bold cyan]
"""


_ANSI = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
_VISIBLE_ANSI = re.compile(r"(?:\?|\\x1b)\[[0-9;?]*[ -/]*[@-~]")


def clean_terminal_text(value: object) -> str:
    """Remove provider-supplied or already-mangled terminal control sequences."""
    text = str(value)
    return _VISIBLE_ANSI.sub("", _ANSI.sub("", text))


def _elapsed(start: float) -> str:
    """Format elapsed seconds as human-readable string."""
    delta = time.monotonic() - start
    if delta < 60:
        return f"{delta:.1f}s"
    minutes = int(delta // 60)
    seconds = delta % 60
    return f"{minutes}m {seconds:.0f}s"


class ThinkingSpinner:
    """Animated spinner shown while the agent is working."""

    def __init__(self, console: Console) -> None:
        self._console = console
        self._start: float = 0
        self._label = "Thinking"
        self._active = False

    def start(self, label: str = "Thinking") -> None:
        self._start = time.monotonic()
        self._label = label
        self._active = True
        self._console.print(
            Text(f"  ◐ {label}…", style="dim italic"),
        )

    def update(self, label: str) -> None:
        if self._active and label != self._label:
            self._label = label
            self._console.print(
                Text(f"  ◑ {label} ({_elapsed(self._start)})…", style="dim italic"),
            )

    def stop(self, final_label: str = "") -> None:
        if self._active:
            self._active = False
            elapsed = _elapsed(self._start)
            msg = final_label or self._label
            self._console.print(
                Text(f"  ● {msg} · {elapsed}", style="dim"),
            )


class TerminalRenderer:
    """Keep durable internal events separate from the user's conversational UI."""

    def __init__(self, console: Console) -> None:
        self.console = console
        self.spinner = ThinkingSpinner(console)
        self._turn_count = 0

    def banner(self, session_id: str, model: str, workspace: Path) -> None:
        self.console.print(_LOGO.format(version=_VERSION))
        details = Table.grid(padding=(0, 2))
        details.add_column(style="bold blue", no_wrap=True, justify="right")
        details.add_column(style="white")
        details.add_row("session", session_id)
        details.add_row("model", f"[green]{model}[/green]")
        details.add_row("worktree", f"[dim]{workspace}[/dim]")
        self.console.print(Panel(details, border_style="blue", padding=(0, 2)))
        self.console.print(
            Text("  Type a request or ", style="dim")
            + Text("/help", style="bold cyan")
            + Text(" for commands · ", style="dim")
            + Text("Ctrl+G", style="bold")
            + Text(" editor · ", style="dim")
            + Text("Tab", style="bold")
            + Text(" complete", style="dim"),
        )
        self.console.print()

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
            ("/suggest TASK", "Generate a diff suggestion without applying"),
            ("/goal OBJECTIVE", "Set a goal; agent auto-continues until achieved"),
            ("/goal clear", "Clear the active goal"),
            ("/review", "LLM-powered code review of the current diff"),
            ("/search QUERY", "Web search and return results"),
            ("/export", "Export session events to JSONL"),
            ("/init [--force]", "Analyze repo and generate AGENTS.md"),
            ("/pr [--draft]", "Push branch and create a pull request"),
            ("/orchestrate OBJ", "Decompose and execute a complex task"),
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

    def user_input(self, text: str) -> None:
        """Render the user's input with a visual label."""
        self._turn_count += 1
        self.console.print()
        self.console.print(Rule(style="blue"))
        # Show truncated input as a header.
        preview = text.split("\n")[0][:80]
        if len(text.split("\n")[0]) > 80:
            preview += "…"
        self.console.print(
            Text("  ❯ ", style="bold blue") + Text(preview, style="white")
        )
        self.console.print()

    def event(self, event: SessionEvent) -> None:
        if event.kind == "thinking":
            if event.name == "llm_call_started":
                method = event.data.get("method", "")
                turn = event.data.get("turn", 0)
                label = f" {method}" if method else ""
                self.spinner.start(f"Thinking{label} (turn {turn})")
            elif event.name == "llm_call_ended":
                success = event.data.get("success", True)
                if success:
                    self.spinner.stop("Done thinking")
                else:
                    self.spinner.stop()
                    self.console.print(Text("  ✗ LLM call failed", style="bold red"))
            return
        if event.kind == "usage":
            return
        if event.kind == "message":
            content = event.data.get("content")
            if content:
                self.spinner.stop()
                self.console.print(Markdown(clean_terminal_text(content)))
            return
        if event.name == "approval_requested":
            request_id = clean_terminal_text(event.data.get("request_id", ""))
            kind = clean_terminal_text(event.data.get("kind", "operation"))
            resource = clean_terminal_text(event.data.get("resource", ""))
            body = Group(
                Text(f"  {kind}: ", style="bold") + Text(resource),
                Text(f"  /approve {request_id}", style="green")
                + Text(" or ", style="dim")
                + Text(f"/deny {request_id}", style="red"),
            )
            self.console.print(
                Panel(body, title=f"⚠ Approval · {request_id}", border_style="yellow")
            )
            return
        if event.name == "approval_resolved":
            allowed = event.data.get("allowed")
            icon = "✓" if allowed else "✗"
            word = "approved" if allowed else "denied"
            style = "dim green" if allowed else "dim red"
            self.console.print(Text(f"  {icon} {word}", style=style))
            return
        if event.name == "command_started":
            command = clean_terminal_text(event.data.get("command", ""))
            self.spinner.update(f"Running: {command[:60]}")
            self.console.print(
                Text("  $ ", style="bold cyan") + Text(command, style="cyan")
            )
            return
        if event.name == "command_finished":
            code = int(event.data.get("returncode", 1))
            if code == 0:
                self.console.print(Text("  ✓ ok", style="dim green"))
            else:
                stderr = clean_terminal_text(event.data.get("stderr", "")).strip()
                tail = "\n".join(stderr.splitlines()[-6:])
                self.console.print(
                    Panel(
                        Text(tail or f"exit {code}", style="red"),
                        title=f"✗ exit {code}",
                        border_style="red",
                        padding=(0, 1),
                    )
                )
            return
        if event.name == "file_changed":
            path = clean_terminal_text(event.data.get("path", ""))
            operation = event.data.get("operation", "edit")
            icon = "✎" if operation == "replace" else "＋"
            self.console.print(
                Text(f"  {icon} ", style="green") + Text(path, style="underline green")
            )
            diff = event.data.get("diff", "")
            if diff:
                for line in diff.splitlines()[:8]:
                    if line.startswith("+ "):
                        self.console.print(Text(f"    {line}", style="green"))
                    elif line.startswith("- "):
                        self.console.print(Text(f"    {line}", style="red"))
                    else:
                        self.console.print(Text(f"    {line}", style="dim"))
            return
        if event.name == "mcp_server_connected":
            server = clean_terminal_text(event.data.get("server", ""))
            tools = len(event.data.get("tools", []))
            self.console.print(
                Text(f"  ⚡ MCP {server}", style="magenta")
                + Text(f" ({tools} tools)", style="dim")
            )
            return
        if event.name == "mcp_server_failed":
            server = clean_terminal_text(event.data.get("server", ""))
            error = clean_terminal_text(event.data.get("error", ""))
            self.console.print(
                Panel(error, title=f"⚡ MCP failed · {server}", border_style="yellow")
            )
            return
        if event.name == "mcp_call_started":
            server = clean_terminal_text(event.data.get("server", ""))
            tool = clean_terminal_text(event.data.get("tool", ""))
            self.spinner.update(f"MCP {server}.{tool}")
            self.console.print(
                Text(f"  ⚡ {server}", style="magenta") + Text(f".{tool}", style="dim magenta")
            )
            return
        if event.name == "mcp_call_finished":
            elapsed = event.data.get("duration_ms", 0)
            truncated = " · truncated" if event.data.get("truncated") else ""
            self.console.print(
                Text(f"  ✓ {elapsed}ms{truncated}", style="dim green")
            )
            return
        if event.name == "mcp_call_failed":
            server = clean_terminal_text(event.data.get("server", ""))
            tool = clean_terminal_text(event.data.get("tool", ""))
            error = clean_terminal_text(event.data.get("error", ""))
            self.console.print(
                Panel(error, title=f"✗ {server}.{tool}", border_style="red")
            )
            return
        if event.name == "cell_guard_blocked":
            reason = clean_terminal_text(event.data.get("reason", ""))
            self.console.print(
                Text("  ⛔ ", style="bold yellow") + Text(reason, style="yellow")
            )
            return
        if event.kind == "model_failover":
            source = clean_terminal_text(event.data.get("from", ""))
            target = clean_terminal_text(event.data.get("to", ""))
            self.console.print(
                Panel(
                    Text(f"  {source} → {target}", style="yellow"),
                    title="⚡ Model failover",
                    border_style="yellow",
                )
            )
            return
        if event.kind == "error":
            self.spinner.stop()
            message = clean_terminal_text(
                event.data.get("error") or event.data.get("message") or event.name
            )
            self.console.print(
                Panel(message, title="✗ Error", border_style="red")
            )

    def result(self, result: CodingTaskResult) -> None:
        self.spinner.stop()
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
        icon = {
            "completed": "✓",
            "inspected": "◎",
            "blocked": "⚠",
            "verification_failed": "✗",
        }.get(result.status, "•")

        sections: list[RenderableType] = [Markdown(summary)]
        if result.changed_files:
            file_list = Text()
            for path in result.changed_files:
                file_list.append("  ✎ ", style="green")
                file_list.append(path + "\n", style="underline")
            sections.extend([Text("Files", style="bold"), file_list])
        if result.verifications:
            checks = Table(box=None, show_header=False, pad_edge=False)
            checks.add_column(width=3)
            checks.add_column()
            for check in result.verifications:
                if check.passed:
                    checks.add_row("[green]✓[/green]", check.command)
                else:
                    checks.add_row("[red]✗[/red]", f"[red]{check.command}[/red]")
            sections.extend([Text("Verification", style="bold"), checks])
        evidence = clean_terminal_text(result.evidence).strip()
        if evidence:
            # Truncate long evidence.
            if len(evidence) > 500:
                evidence = evidence[:500] + "…"
            sections.extend([Text("Evidence", style="bold"), Text(evidence, style="dim")])

        title = f"{icon} {result.mode} · {result.status}"
        self.console.print(
            Panel(
                Group(*sections),
                title=title,
                border_style=color,
                padding=(1, 2),
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
            self.console.print(Text("  No LLM calls yet.", style="dim"))
            return
        parts = [
            f"[bold]{usage.llm_calls}[/bold] calls",
            f"[cyan]{usage.total_prompt_tokens:,}[/cyan] prompt",
            f"[green]{usage.total_completion_tokens:,}[/green] completion",
        ]
        if usage.total_cached_tokens:
            parts.append(f"[dim]{usage.total_cached_tokens:,} cached[/dim]")
        if usage.total_reasoning_tokens:
            parts.append(f"[magenta]{usage.total_reasoning_tokens:,} reasoning[/magenta]")
        parts.append(f"[bold]{usage.total_tokens:,}[/bold] total")
        if usage.total_cost_usd > 0:
            parts.append(f"[yellow]${usage.total_cost_usd:.4f}[/yellow]")
        self.console.print(Text("  ⬡ ", style="blue") + Text.from_markup(" · ".join(parts)))

    def status(self, session: AgentSession) -> None:
        """Render a comprehensive status panel."""
        usage = session.token_usage
        details = Table.grid(padding=(0, 2))
        details.add_column(style="bold blue", no_wrap=True, justify="right")
        details.add_column()
        details.add_row("session", session.session_id)
        details.add_row("model", f"[green]{session.current_model}[/green]")
        details.add_row("status", session.status)
        details.add_row("worktree", f"[dim]{session.workspace}[/dim]")
        details.add_row("llm calls", str(usage.llm_calls))
        details.add_row("tokens", f"{usage.total_tokens:,}")
        if usage.total_cost_usd > 0:
            details.add_row("cost", f"[yellow]${usage.total_cost_usd:.4f}[/yellow]")
        pending = session.approvals.pending()
        details.add_row("approvals", f"[yellow]{len(pending)}[/yellow]" if pending else "0")
        mcp_connected = sum(
            1 for s in session.mcp_status() if s.status == "connected"
        )
        details.add_row("mcp", f"{mcp_connected} connected")
        goal = session.goal
        if goal and goal.status == "active":
            details.add_row("goal", f"[cyan]{goal.objective[:40]}[/cyan]")
            details.add_row("", f"[dim]{goal.turns_used}/{goal.turn_budget} turns[/dim]")
        self.console.print(Panel(details, title="◎ Status", border_style="blue", padding=(1, 2)))

    def goal_progress(self, objective: str, turns_used: int, turn_budget: int, evaluation: str) -> None:
        """Render goal loop progress as a visual bar."""
        filled = min(turns_used, turn_budget)
        bar_len = 20
        done_chars = int(bar_len * filled / turn_budget) if turn_budget else 0
        bar = "█" * done_chars + "░" * (bar_len - done_chars)
        self.console.print(
            Text("  ◎ Goal ", style="bold cyan")
            + Text(f"[{bar}] ", style="cyan")
            + Text(f"{filled}/{turn_budget}", style="dim")
        )
        if evaluation:
            preview = evaluation[:120]
            self.console.print(Text(f"    {preview}", style="dim italic"))

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


__all__ = ["TerminalRenderer", "ThinkingSpinner", "clean_terminal_text"]
