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
from .events import SessionEvent

if TYPE_CHECKING:
    from .policy import ApprovalRequest


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

    def event(self, event: SessionEvent) -> None:
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


__all__ = ["TerminalRenderer", "clean_terminal_text"]
