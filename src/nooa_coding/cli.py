"""Interactive terminal client for the local AgentSession API."""

from __future__ import annotations

import asyncio
import json
import re
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import click
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.text import Text

from .agent import CodingTaskResult
from .config import CodingSettings, ModelEndpoint, load_settings
from .events import SessionEvent
from .session import AgentSession, AgentSessionManager
from .terminal import TerminalRenderer

console = Console()
renderer = TerminalRenderer(console)

_SLASH_COMMANDS = [
    "/help",
    "/diff",
    "/checkpoint",
    "/rollback",
    "/compact",
    "/fork",
    "/approvals",
    "/approve",
    "/deny",
    "/replay",
    "/reload",
    "/mcp",
    "/cost",
    "/status",
    "/clear",
    "/model",
    "/permissions",
    "/review",
    "/plan",
    "/suggest",
    "/goal",
    "/search",
    "/export",
    "/workspace",
    "/init",
    "/pr",
    "/orchestrate",
    "/lessons",
    "/report",
    "/exit",
]


class SlashCompleter(Completer):
    """Complete slash commands and @file mentions."""

    def __init__(self, workspace: Path | None = None) -> None:
        self._workspace = workspace

    def get_completions(self, document, complete_event):  # noqa: ANN001, ANN201
        text = document.text_before_cursor
        # Slash command completion at start of input.
        if text.startswith("/"):
            for command in _SLASH_COMMANDS:
                if command.startswith(text) and command != text:
                    yield Completion(command, start_position=-len(text))
            return
        # @file mention completion: find the last @token in the input.
        at_idx = text.rfind("@")
        if at_idx >= 0:
            # Only trigger if @ is at start or preceded by whitespace.
            if at_idx == 0 or text[at_idx - 1] in (" ", "\n", "\t"):
                query = text[at_idx + 1:]
                # Don't complete if there's a space after @ (already typed path).
                if " " not in query or query.endswith("/"):
                    yield from self._file_completions(query, at_idx, text)

    def _file_completions(self, query: str, at_idx: int, full_text: str):  # noqa: ANN001, ANN201
        """Fuzzy-match files in the workspace."""
        if self._workspace is None or not self._workspace.is_dir():
            return
        prefix_len = len(full_text) - at_idx  # includes the @
        matches: list[str] = []
        try:
            for path in sorted(self._workspace.rglob("*")):
                if len(matches) >= 20:
                    break
                rel = str(path.relative_to(self._workspace))
                # Skip hidden dirs and common noise.
                if any(part.startswith(".") for part in path.relative_to(self._workspace).parts):
                    continue
                if "__pycache__" in rel or "node_modules" in rel:
                    continue
                if query.lower() in rel.lower():
                    suffix = "/" if path.is_dir() else ""
                    matches.append(rel + suffix)
        except OSError:
            return
        for match in matches:
            display = f"@{match}"
            yield Completion(display, start_position=-prefix_len)


def _build_key_bindings() -> KeyBindings:
    """Multi-line input: Ctrl+J inserts newline, Enter sends, Ctrl+G opens editor."""
    bindings = KeyBindings()

    @bindings.add("c-j")
    def _newline(event):  # noqa: ANN001, ANN202
        event.current_buffer.insert_text("\n")

    @bindings.add("c-g")
    def _open_editor(event):  # noqa: ANN001, ANN202
        """Open $EDITOR for multi-line composition."""
        import os
        import subprocess
        import tempfile

        buf = event.current_buffer
        editor = os.environ.get("EDITOR", "vi")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as tmp:
            tmp.write(buf.text)
            tmp_path = tmp.name
        try:
            subprocess.call([editor, tmp_path])
            with open(tmp_path, encoding="utf-8") as f:
                buf.text = f.read().rstrip("\n")
        finally:
            os.unlink(tmp_path)

    return bindings


class AsyncPrompt(Protocol):
    async def prompt_async(self, message: str) -> str: ...


def _expand_at_mentions(text: str, workspace: Path) -> str:
    """Expand @file references by inlining file content."""

    def _replace(match: re.Match) -> str:  # noqa: ANN001
        rel_path = match.group(1)
        target = workspace / rel_path
        if target.is_file():
            try:
                content = target.read_text(encoding="utf-8")
                # Truncate very large files.
                if len(content) > 50_000:
                    content = content[:50_000] + "\n... (truncated)"
                return f"\n--- @{rel_path} ---\n```\n{content}\n```\n"
            except (OSError, UnicodeDecodeError):
                return match.group(0)
        elif target.is_dir():
            entries = sorted(p.name for p in target.iterdir())[:50]
            listing = "\n".join(entries)
            return f"\n--- @{rel_path}/ (directory) ---\n{listing}\n"
        return match.group(0)

    return re.sub(r"@([\w./\-]+)", _replace, text)


class ApprovalController(Protocol):
    def approve(self, request_id: str) -> None: ...

    def deny(self, request_id: str) -> None: ...


async def _next_event(stream: object) -> SessionEvent:
    return await anext(stream)  # type: ignore[arg-type]


def _render_event(event: SessionEvent) -> None:
    renderer.event(event)


def _resolve_approval(session: ApprovalController, request_id: str, *, allow: bool) -> bool:
    """Resolve an approval without letting stale UI input terminate the client."""
    try:
        if allow:
            session.approve(request_id)
        else:
            session.deny(request_id)
    except KeyError:
        console.print(Text(f"Approval {request_id} is no longer pending.", style="yellow"))
        return False
    return True


async def _run_single_turn(
    session: AgentSession,
    text: str,
    prompt: AsyncPrompt,
    *,
    allow_controls: bool,
) -> list[str]:
    cursor = session.replay()[-1].sequence if session.replay() else 0
    turn = session.start(text)
    stream = session.stream(after_sequence=cursor)
    event_task: asyncio.Task[SessionEvent] | None = asyncio.create_task(_next_event(stream))
    control_task: asyncio.Task[str] | None = None
    if allow_controls:
        control_task = asyncio.create_task(
            prompt.prompt_async(HTML("<dim>  ⋮ /cancel · /approve · /deny</dim> > "))
        )
    pending_followups: list[str] = []
    try:
        while not turn.done():
            waiters: set[asyncio.Task[object]] = {turn}  # type: ignore[arg-type]
            if event_task is not None:
                waiters.add(event_task)  # type: ignore[arg-type]
            if control_task is not None:
                waiters.add(control_task)  # type: ignore[arg-type]
            done, _ = await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
            if event_task in done:
                try:
                    event = event_task.result()
                except StopAsyncIteration:
                    event_task = None
                else:
                    cursor = event.sequence
                    _render_event(event)
                    event_task = asyncio.create_task(_next_event(stream))
            if turn in done:
                break
            if control_task is not None and control_task in done:
                try:
                    command = control_task.result().strip()
                except (EOFError, KeyboardInterrupt):
                    await session.cancel()
                    break
                if command == "/cancel":
                    await session.cancel()
                    break
                elif command.lower() in {"y", "yes", "n", "no"}:
                    pending = session.approvals.pending()
                    if len(pending) != 1:
                        console.print(
                            Text(
                                "Use /approve ID or /deny ID when approval is ambiguous.",
                                style="yellow",
                            )
                        )
                    elif command.lower() in {"y", "yes"}:
                        _resolve_approval(session, pending[0].request_id, allow=True)
                    else:
                        _resolve_approval(session, pending[0].request_id, allow=False)
                elif command.startswith("/approve "):
                    _resolve_approval(session, command.split(maxsplit=1)[1], allow=True)
                elif command.startswith("/deny "):
                    _resolve_approval(session, command.split(maxsplit=1)[1], allow=False)
                elif command in {"/approve", "/deny"}:
                    console.print(Text(f"Usage: {command} REQUEST_ID", style="yellow"))
                elif command.startswith("/"):
                    console.print(
                        Text(
                            "Unknown running command. Use /cancel, /approve, or /deny.",
                            style="yellow",
                        )
                    )
                elif command:
                    pending_followups.append(command)
                    console.print(Text("Follow-up queued for the next turn.", style="dim"))
                if not turn.done():
                    control_task = asyncio.create_task(
                        prompt.prompt_async(HTML("<dim>  ⋮ /cancel · /approve · /deny</dim> > "))
                    )
        try:
            result = await turn
        except asyncio.CancelledError:
            console.print(Text("Turn cancelled.", style="yellow"))
        else:
            renderer.result(result)
    finally:
        for task in (event_task, control_task):
            if task is not None and not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

    return pending_followups


async def _run_turn(
    session: AgentSession,
    text: str,
    prompt: AsyncPrompt,
    *,
    allow_controls: bool,
) -> None:
    queued = [text]
    while queued:
        current = queued.pop(0)
        queued.extend(
            await _run_single_turn(session, current, prompt, allow_controls=allow_controls)
        )


def _help() -> None:
    renderer.help()


def _auto_approve(settings: CodingSettings) -> CodingSettings:
    permissions = settings.permissions.model_copy(
        update={"file_read": "allow", "file_write": "allow", "shell": "allow"}
    )
    mcp_permissions = settings.mcp.permissions.model_copy(update={"default": "allow"})
    mcp = settings.mcp.model_copy(update={"permissions": mcp_permissions})
    return settings.model_copy(update={"permissions": permissions, "mcp": mcp})


def _handle_mcp_command(session: AgentSession, text: str) -> None:
    parts = text.split()
    action = parts[1] if len(parts) > 1 else "list"
    argument = parts[2] if len(parts) > 2 else None
    try:
        if action == "list" and argument is None:
            renderer.mcp_status(session.mcp_status(), session.mcp_config_errors())
        elif action == "tools" and len(parts) <= 3:
            renderer.mcp_tools(session.mcp_tools(argument))
        elif action == "enable" and argument and len(parts) == 3:
            renderer.mcp_status([session.mcp_enable(argument)], [])
        elif action == "disable" and argument and len(parts) == 3:
            renderer.mcp_status([session.mcp_disable(argument)], [])
        elif action == "reload" and len(parts) <= 3:
            renderer.mcp_status(session.mcp_reload(argument), session.mcp_config_errors())
        else:
            console.print(
                Text(
                    "Usage: /mcp list | tools [SERVER] | enable SERVER | "
                    "disable SERVER | reload [SERVER]",
                    style="yellow",
                )
            )
    except (KeyError, RuntimeError, ValueError) as exc:
        message = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
        console.print(Text(f"MCP: {message}", style="yellow"))


async def _decompose_objective(session: AgentSession, objective: str) -> list:
    """Use LLM to decompose an objective into TaskPackages for sub-agents."""
    from .multi_agent import TaskPackage

    decompose_prompt = (
        "You are a task orchestrator. Decompose the following objective into "
        "2-5 concrete, independent sub-tasks that can be executed in parallel "
        "by isolated worker agents.\n\n"
        "Rules:\n"
        "- Each sub-task must be self-contained and actionable.\n"
        "- Sub-tasks must NOT depend on each other (they run in isolation).\n"
        "- Keep each sub-task focused on ONE logical change.\n\n"
        "Respond in this EXACT format (one task per line):\n"
        "TASK: <description>\n"
        "TASK: <description>\n"
        "...\n\n"
        f"OBJECTIVE: {objective}"
    )
    messages = [{"role": "user", "content": decompose_prompt}]
    response = await session.llm.acall(messages)
    if hasattr(session, "_track_direct_llm_call"):
        session._track_direct_llm_call(response)  # noqa: SLF001
    content = response.content or ""

    tasks: list[TaskPackage] = []
    for line in content.splitlines():
        line = line.strip()
        if line.upper().startswith("TASK:"):
            desc = line[5:].strip()
            if desc:
                tasks.append(
                    TaskPackage(
                        objective=desc,
                        context_summary=f"Part of: {objective}",
                        base_commit="HEAD",
                        timeout_seconds=session.settings.subagent.timeout_seconds,
                        token_budget=session.settings.subagent.token_budget,
                    )
                )
    if not tasks:
        # Fallback: single task.
        tasks = [TaskPackage(objective=objective, base_commit="HEAD")]
    return tasks


async def _interactive(manager: AgentSessionManager, session: AgentSession) -> None:
    terminal = PromptSession[str](
        completer=SlashCompleter(workspace=session.workspace),
        key_bindings=_build_key_bindings(),
        multiline=False,  # Enter sends; Ctrl+J inserts newline via bindings
        enable_history_search=True,
    )
    renderer.banner(session.session_id, session.current_model, session.workspace)

    def _prompt_msg() -> HTML:
        model_short = session.current_model.split("/")[-1][:20]
        return HTML(f"<b><ansiblue>{model_short}</ansiblue></b> <dim>❯</dim> ")

    def _bottom_toolbar() -> HTML:
        goal_hint = ""
        if session.goal and session.goal.status == "active":
            goal_hint = f" | <ansicyan>◎ goal {session.goal.turns_used}/{session.goal.turn_budget}</ansicyan>"
        return HTML(
            " <dim>Ctrl+J</dim> newline"
            " · <dim>Ctrl+G</dim> editor"
            " · <dim>Tab</dim> complete"
            " · <dim>/help</dim> commands"
            f"{goal_hint}"
        )

    with patch_stdout(raw=True):
        while True:
            try:
                text = (await terminal.prompt_async(_prompt_msg(), bottom_toolbar=_bottom_toolbar)).strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text:
                continue
            if text == "/exit":
                break
            if text == "/help":
                _help()
            elif text == "/diff":
                diff = session.diff()
                console.print(Syntax(diff.status + diff.stat + diff.patch, "diff"))
            elif text.startswith("/checkpoint"):
                label = text.partition(" ")[2].strip() or "manual"
                checkpoint = session.checkpoint(label)
                console.print(
                    Text(
                        f"Checkpoint {checkpoint.checkpoint_id} saved ({checkpoint.label}).",
                        style="green",
                    )
                )
            elif text.startswith("/rollback "):
                checkpoint = session.rollback(text.split(maxsplit=1)[1])
                console.print(
                    Text(f"Restored checkpoint {checkpoint.checkpoint_id}.", style="green")
                )
            elif text == "/compact":
                console.print(f"summary tag: {await session.compact()}")
            elif text.startswith("/fork"):
                requested = text.partition(" ")[2].strip() or None
                child = await session.fork(requested)
                await session.close()
                session = child
                renderer.banner(session.session_id, session.current_model, session.workspace)
            elif text == "/approvals":
                renderer.approvals(session.approvals.pending())
            elif text.startswith("/approve "):
                _resolve_approval(session, text.split(maxsplit=1)[1], allow=True)
            elif text.startswith("/deny "):
                _resolve_approval(session, text.split(maxsplit=1)[1], allow=False)
            elif text.startswith("/replay"):
                raw = text.partition(" ")[2].strip()
                count = int(raw) if raw else 20
                for event in session.replay()[-count:]:
                    _render_event(event)
            elif text == "/reload":
                console.print(session.agent.reload_project_resources())
            elif text == "/mcp" or text.startswith("/mcp "):
                _handle_mcp_command(session, text)
            elif text == "/workspace":
                console.print(str(session.workspace))
            elif text == "/init" or text.startswith("/init "):
                from .init_project import init_project

                force = "--force" in text
                try:
                    content = init_project(session.workspace, force=force)
                    console.print(Text("✓ Generated AGENTS.md", style="green"))
                    console.print(Text(content[:500], style="dim"))
                except FileExistsError:
                    console.print(
                        Text("AGENTS.md already exists. Use /init --force to overwrite.", style="yellow")
                    )
            elif text == "/pr" or text.startswith("/pr "):
                from .pr import prepare_and_create_pr

                draft = "--draft" in text
                console.print(Text("⟳ Preparing pull request…", style="dim italic"))
                pr_result = await prepare_and_create_pr(
                    session.workspace, draft=draft, llm=session.llm
                )
                if pr_result.success:
                    console.print(Text("✓ PR created", style="green"))
                    if pr_result.pr_url:
                        console.print(Text(f"  {pr_result.pr_url}", style="underline cyan"))
                else:
                    console.print(Text(f"✗ {pr_result.error}", style="red"))
            elif text.startswith("/orchestrate"):
                objective = text.partition(" ")[2].strip()
                if not objective:
                    console.print(
                        Text("Usage: /orchestrate OBJECTIVE", style="yellow")
                    )
                else:
                    console.print(
                        Text(f"⟳ Decomposing: {objective[:80]}…", style="dim italic")
                    )
                    # Use LLM to decompose into task packages.
                    tasks = await _decompose_objective(session, objective)
                    console.print(
                        Text(
                            f"  → {len(tasks)} sub-tasks (parallel isolated sessions)",
                            style="cyan",
                        )
                    )
                    for i, tp in enumerate(tasks, 1):
                        console.print(Text(f"    {i}. {tp.objective[:70]}", style="dim"))
                    console.print(Text("  Spawning sub-agents…", style="dim italic"))

                    def _on_progress(report):  # noqa: ANN001, ANN201
                        icon = "✓" if report.status == "completed" else "✗"
                        style = "green" if report.status == "completed" else "red"
                        console.print(
                            Text(f"    {icon} [{report.task_id}] {report.summary[:50] or report.error[:50]}", style=style)
                        )

                    reports = await session.delegate_tasks(tasks, on_progress=_on_progress)
                    completed = sum(1 for r in reports if r.status == "completed")
                    failed = sum(1 for r in reports if r.status in ("failed", "timeout"))
                    console.print(
                        Text(
                            f"  Done: {completed}/{len(reports)} completed, {failed} failed",
                            style="green" if failed == 0 else "yellow",
                        )
                    )
                    # Merge results back into main worktree.
                    if completed > 0:
                        console.print(Text("  Merging worktrees…", style="dim italic"))
                        merge_result = session.merge_subagent_results(reports)
                        if merge_result.success:
                            console.print(
                                Text(f"  ✓ Merged {len(merge_result.merged)} task(s)", style="green")
                            )
                        else:
                            console.print(
                                Text(f"  ⚠ Conflicts: {merge_result.conflicts}", style="yellow")
                            )
            elif text == "/lessons" or text.startswith("/lessons "):
                store = session.lesson_store
                subcmd = text.partition(" ")[2].strip()
                if subcmd == "stats":
                    stats = store.stats()
                    console.print(
                        Text(
                            f"Lessons: {stats['total']} total, by category: {stats['by_category']}",
                            style="cyan",
                        )
                    )
                elif subcmd.startswith("delete "):
                    lesson_id = subcmd[7:].strip()
                    if store.delete(lesson_id):
                        console.print(Text(f"Deleted lesson {lesson_id}", style="green"))
                    else:
                        console.print(Text(f"Lesson not found: {lesson_id}", style="yellow"))
                else:
                    lessons = store.recent(limit=10)
                    if not lessons:
                        console.print(Text("No lessons stored yet.", style="dim"))
                    else:
                        for lesson in lessons:
                            console.print(
                                Text(f"  [{lesson.category}] ", style="cyan")
                                + Text(lesson.title, style="bold")
                                + Text(f" ({lesson.lesson_id})", style="dim")
                            )
                            console.print(Text(f"    {lesson.content[:100]}", style="dim"))
            elif text == "/report" or text.startswith("/report "):
                from .report import ReportMetadata, SessionReport, write_report

                fmt_arg = text.partition(" ")[2].strip()
                formats = fmt_arg.split(",") if fmt_arg else ["json", "html"]
                usage = session.token_usage
                result = session.agent.last_result
                report = SessionReport(
                    metadata=ReportMetadata(
                        session_id=session.session_id,
                        generated_at=datetime.now(UTC).isoformat(),
                    ),
                    summary=result.summary if result else "No result yet",
                    status=result.status if result else "idle",
                    changed_files=result.changed_files if result else [],
                    verifications=[
                        v.model_dump() for v in (result.verifications if result else [])
                    ],
                    token_usage={
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                        "total_tokens": usage.total_tokens,
                        "cost_usd": f"{usage.total_cost_usd:.4f}",
                    },
                    events_count=session._sequence,
                )
                output_dir = Path.cwd() / "nooa-reports"
                written = write_report(report, output_dir, formats=formats)
                for path in written:
                    console.print(Text(f"✓ {path}", style="green"))
            elif text == "/cost":
                renderer.token_usage(session.token_usage)
            elif text == "/status":
                renderer.status(session)
            elif text == "/clear":
                session.clear_history()
                console.print(Text("Conversation history cleared.", style="dim"))
            elif text.startswith("/model"):
                requested_model = text.partition(" ")[2].strip()
                if not requested_model:
                    models_list = session.available_models
                    current = session.current_model
                    console.print(Text(f"Current model: {current}", style="cyan"))
                    if len(models_list) > 1:
                        console.print(
                            Text(f"Available: {', '.join(models_list)}", style="dim")
                        )
                        console.print(
                            Text("Use /model NAME to switch.", style="dim")
                        )
                else:
                    try:
                        new_model = session.switch_model(requested_model)
                        console.print(
                            Text(f"Switched to model: {new_model}", style="green")
                        )
                    except (ValueError, RuntimeError) as exc:
                        console.print(Text(str(exc), style="yellow"))
            elif text.startswith("/permissions"):
                mode = text.partition(" ")[2].strip()
                if not mode:
                    perms = session.settings.permissions
                    console.print(
                        Text(
                            f"file_read={perms.file_read} "
                            f"file_write={perms.file_write} "
                            f"shell={perms.shell}",
                            style="cyan",
                        )
                    )
                    console.print(
                        Text("Use /permissions allow|ask|default to switch.", style="dim")
                    )
                else:
                    try:
                        session.switch_permissions(mode)
                        console.print(
                            Text(f"Permission mode set to: {mode}", style="green")
                        )
                    except ValueError as exc:
                        console.print(Text(str(exc), style="yellow"))
            elif text == "/export":
                export_path = Path(f"nooa-session-{session.session_id}.jsonl")
                events = session.replay()
                with export_path.open("w", encoding="utf-8") as f:
                    for ev in events:
                        f.write(ev.model_dump_json() + "\n")
                console.print(
                    Text(f"Exported {len(events)} events to {export_path}", style="green")
                )
            elif text == "/review":
                console.print(Text("⟳ Reviewing current diff…", style="dim italic"))
                try:
                    async for token in session.stream_direct(
                        "Review the following diff:\n" + session.diff().patch[:30000]
                    ):
                        console.print(token, end="", highlight=False)
                    console.print()
                except Exception:
                    review_text = await session.review()
                    console.print(Markdown(review_text))
            elif text.startswith("/plan"):
                plan_task = text.partition(" ")[2].strip()
                if not plan_task:
                    console.print(
                        Text("Usage: /plan <task description>", style="yellow")
                    )
                else:
                    console.print(Text("⟳ Generating plan…", style="dim italic"))
                    try:
                        async for token in session.stream_direct(
                            "Create a detailed implementation plan for: " + plan_task
                        ):
                            console.print(token, end="", highlight=False)
                        console.print()
                    except Exception:
                        plan_text = await session.plan(plan_task)
                        console.print(Markdown(plan_text))
                    console.print(
                        Text(
                            "\nSend the task text to execute, or refine with /plan again.",
                            style="dim",
                        )
                    )
            elif text.startswith("/suggest"):
                suggest_task = text.partition(" ")[2].strip()
                if not suggest_task:
                    console.print(Text("Usage: /suggest <what to change>", style="yellow"))
                else:
                    console.print(Text("⟳ Generating suggestion…", style="dim italic"))
                    suggestion = await session.suggest(suggest_task)
                    console.print(Syntax(suggestion, "diff", word_wrap=True))
                    console.print(
                        Text(
                            "\nApply manually or send the task text to execute.",
                            style="dim",
                        )
                    )
            elif text.startswith("/search"):
                query = text.partition(" ")[2].strip()
                if not query:
                    console.print(Text("Usage: /search <query>", style="yellow"))
                else:
                    console.print(Text(f"⟳ Searching: {query}…", style="dim italic"))
                    results = await session.web_search(query)
                    console.print(Text(results))
            elif text.startswith("/goal"):
                arg = text.partition(" ")[2].strip()
                if not arg:
                    # Show current goal status.
                    goal = session.goal
                    if goal is None:
                        console.print(Text("No active goal. Use /goal <objective> to set one.", style="dim"))
                    else:
                        console.print(Text(
                            f"Goal: {goal.objective}\n"
                            f"Status: {goal.status} | Turns: {goal.turns_used}/{goal.turn_budget}\n"
                            f"Last evaluation: {goal.last_evaluation[:200]}",
                            style="cyan",
                        ))
                elif arg == "clear":
                    session.clear_goal()
                    console.print(Text("Goal cleared.", style="dim"))
                else:
                    # Parse optional --budget N
                    budget = 10
                    objective = arg
                    if "--budget" in arg:
                        parts = arg.rsplit("--budget", 1)
                        objective = parts[0].strip()
                        try:
                            budget = int(parts[1].strip())
                        except ValueError:
                            pass
                    goal = session.set_goal(objective, turn_budget=budget)
                    console.print(Text(
                        f"Goal set: {goal.objective}\n"
                        f"Turn budget: {goal.turn_budget}. Agent will auto-continue until achieved.",
                        style="green",
                    ))
            elif text.startswith("/"):
                console.print(Text("Unknown command. Use /help.", style="red"))
            else:
                expanded = _expand_at_mentions(text, session.workspace)
                renderer.user_input(text)
                await _run_turn(session, expanded, terminal, allow_controls=True)
                # Goal-driven reconcile loop: auto-continue until achieved.
                while session.goal is not None and session.goal.status == "active":
                    last_events = session.replay()
                    turn_results = [
                        e for e in last_events if e.name == "turn_finished"
                    ]
                    if not turn_results:
                        break
                    last_data = turn_results[-1].data.get("result", {})
                    last_result = CodingTaskResult.model_validate(last_data)
                    console.print(Text(
                        f"  ◐ Evaluating goal ({session.goal.turns_used + 1}/{session.goal.turn_budget})…",
                        style="dim italic",
                    ))
                    achieved, evaluation = await session.evaluate_goal(last_result)
                    if achieved:
                        renderer.goal_progress(
                            session.goal.objective,
                            session.goal.turns_used,
                            session.goal.turn_budget,
                            "✓ Achieved!",
                        )
                        console.print(Text(
                            f"  ✓ Goal achieved: {evaluation}",
                            style="bold green",
                        ))
                        break
                    if session.goal.status != "active":
                        console.print(Text(
                            f"  Goal loop ended: {evaluation}",
                            style="yellow",
                        ))
                        break
                    # Show progress bar.
                    renderer.goal_progress(
                        session.goal.objective,
                        session.goal.turns_used,
                        session.goal.turn_budget,
                        evaluation,
                    )
                    # Extract next action from evaluation.
                    next_action = evaluation
                    if "NOT_ACHIEVED:" in evaluation:
                        next_action = evaluation.split("NOT_ACHIEVED:", 1)[1].strip()
                    continuation = (
                        f"Continue working toward the goal: {session.goal.objective}\n\n"
                        f"Evaluator feedback: {next_action}"
                    )
                    await _run_turn(session, continuation, terminal, allow_controls=True)
    await session.close()


def _run_doctor(repo: Path, settings: CodingSettings, settings_file: Path | None) -> None:
    """Run environment diagnostics and print a health report."""
    import shutil
    import subprocess

    from rich.table import Table as RichTable

    table = RichTable(title="nooa-code doctor", show_header=True)
    table.add_column("Check", style="bold")
    table.add_column("Status")
    table.add_column("Detail", style="dim")

    # Git
    git_path = shutil.which("git")
    if git_path:
        try:
            ver = subprocess.run(["git", "--version"], capture_output=True, text=True, check=True)
            table.add_row("git", "✓", ver.stdout.strip())
        except subprocess.CalledProcessError:
            table.add_row("git", "✗", "git found but --version failed")
    else:
        table.add_row("git", "✗", "not found in PATH")

    # Repo is a git repository
    git_dir = repo / ".git"
    if git_dir.exists():
        table.add_row("repository", "✓", str(repo))
    else:
        table.add_row("repository", "✗", f"{repo} is not a git repo")

    # Models configured
    if settings.models:
        names = ", ".join(m.name for m in settings.models)
        table.add_row("models", "✓", names)
    else:
        table.add_row("models", "✗", "no models configured")

    # Settings file
    from .config import settings_paths

    found = settings_paths(repo, settings_file)
    if found:
        table.add_row("settings", "✓", ", ".join(str(p) for p in found))
    else:
        table.add_row("settings", "⚠", "no settings file found (using defaults)")

    # Sessions directory
    sessions_path = settings.sessions_path()
    table.add_row("sessions_dir", "✓" if sessions_path.is_dir() else "⚠", str(sessions_path))

    # Worktrees directory
    worktrees_path = settings.worktrees_path()
    table.add_row("worktrees_dir", "✓" if worktrees_path.is_dir() else "⚠", str(worktrees_path))

    # Verification commands
    if settings.verification_commands:
        table.add_row("verification", "✓", f"{len(settings.verification_commands)} command(s)")
    else:
        table.add_row("verification", "⚠", "none configured")

    # MCP
    if settings.mcp.enabled:
        table.add_row("mcp", "✓", f"servers: {settings.mcp.enabled_servers}")
    else:
        table.add_row("mcp", "⚠", "disabled")

    console.print(table)


@click.command()
@click.option(
    "--repo",
    default=".",
    show_default=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option("--settings", "settings_file", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--model", "models", multiple=True, help="Model names in failover order.")
@click.option("--api-base", default=None, help="API base for command-line model overrides.")
@click.option("--session", "session_id", help="Create or resume this session id.")
@click.option("--resume", is_flag=True, help="Require an existing session.")
@click.option("--task", help="Run one task and exit instead of opening the TUI.")
@click.option(
    "--yes",
    is_flag=True,
    help="Auto-approve non-denied file, shell, and MCP operations.",
)
@click.option("--json", "output_json", is_flag=True, help="Output result as JSON (implies --task).")
@click.option("--image", "images", multiple=True, type=click.Path(exists=True, dir_okay=False, path_type=Path), help="Attach image file(s) for multimodal input.")
@click.option("--verbose", is_flag=True, help="Enable verbose debug logging.")
@click.option("--doctor", is_flag=True, help="Run environment diagnostics and exit.")
@click.option("--list-sessions", is_flag=True, help="List project sessions and exit.")
def main(
    repo: Path,
    settings_file: Path | None,
    models: tuple[str, ...],
    api_base: str | None,
    session_id: str | None,
    resume: bool,
    task: str | None,
    yes: bool,
    output_json: bool,
    images: tuple[Path, ...],
    verbose: bool,
    doctor: bool,
    list_sessions: bool,
) -> None:
    """Run the local, worktree-isolated NOOA Coding Agent.

    Pipe stdin to inject additional context:

        cat error.log | nooa-code --task "diagnose this" --yes
    """
    try:
        if verbose:
            import logging

            logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s: %(message)s")

        settings = load_settings(repo, settings_file)

        if doctor:
            _run_doctor(repo, settings, settings_file)
            return

        if models:
            settings = settings.model_copy(
                update={
                    "models": tuple(ModelEndpoint(name=name, api_base=api_base) for name in models)
                }
            )
        if yes or output_json:
            settings = _auto_approve(settings)
        manager = AgentSessionManager(repo, settings)
        if list_sessions:
            click.echo(
                json.dumps(
                    [item.model_dump() for item in manager.list()], indent=2, ensure_ascii=False
                )
            )
            return
        if not settings.models:
            raise click.UsageError("configure coding_agent.models or pass --model")
        if resume:
            if not session_id:
                raise click.UsageError("--resume requires --session")
            session = manager.resume(session_id)
        elif session_id and manager.session_dir(session_id).exists():
            session = manager.resume(session_id)
        else:
            session = manager.create(session_id)

        # Read piped stdin as additional context.
        stdin_context = ""
        import sys

        if not sys.stdin.isatty():
            stdin_context = sys.stdin.read().strip()

        effective_task = task
        if stdin_context and effective_task:
            effective_task = f"{effective_task}\n\n--- Piped context ---\n{stdin_context}"
        elif stdin_context and not effective_task:
            effective_task = stdin_context

        # Attach image context for multimodal input.
        if images and effective_task:
            image_parts: list[str] = []
            max_image_bytes = 4 * 1024 * 1024  # 4 MB per image
            for img_path in images:
                try:
                    raw = img_path.read_bytes()
                    if len(raw) > max_image_bytes:
                        image_parts.append(
                            f"[Image: {img_path.name} ({len(raw)} bytes, too large to inline)]"
                        )
                    else:
                        import base64

                        data = base64.b64encode(raw).decode()
                        suffix = img_path.suffix.lstrip(".").lower() or "png"
                        mime = {"jpg": "jpeg", "svg": "svg+xml"}.get(suffix, suffix)
                        image_parts.append(
                            f"![{img_path.name}](data:image/{mime};base64,{data})"
                        )
                except OSError:
                    image_parts.append(f"[Image: {img_path.name} (unreadable)]")
            effective_task = f"{effective_task}\n\n--- Attached images ---\n" + "\n".join(image_parts)

        async def run() -> None:
            if output_json and effective_task:
                result = await session.prompt(effective_task)
                click.echo(result.model_dump_json(indent=2))
                await session.close()
            elif effective_task:
                terminal = PromptSession[str]()
                with patch_stdout(raw=True):
                    await _run_turn(session, effective_task, terminal, allow_controls=not yes)
                await session.close()
            else:
                await _interactive(manager, session)

        asyncio.run(run())
    except click.ClickException:
        raise
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


if __name__ == "__main__":
    main()
