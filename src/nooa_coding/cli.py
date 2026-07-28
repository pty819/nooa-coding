"""Interactive terminal client for the local AgentSession API."""

from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from pathlib import Path

import click
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console
from rich.syntax import Syntax
from rich.text import Text

from .config import ModelEndpoint, load_settings
from .events import SessionEvent
from .session import AgentSession, AgentSessionManager
from .terminal import TerminalRenderer

console = Console()
renderer = TerminalRenderer(console)


async def _next_event(stream: object) -> SessionEvent:
    return await anext(stream)  # type: ignore[arg-type]


def _render_event(event: SessionEvent) -> None:
    renderer.event(event)


async def _run_single_turn(
    session: AgentSession,
    text: str,
    prompt: PromptSession[str],
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
            prompt.prompt_async("[running: /cancel, /approve ID, /deny ID] > ")
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
            if control_task is not None and control_task in done:
                command = control_task.result().strip()
                if command == "/cancel":
                    await session.cancel()
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
                        session.approve(pending[0].request_id)
                    else:
                        session.deny(pending[0].request_id)
                elif command.startswith("/approve "):
                    session.approve(command.split(maxsplit=1)[1])
                elif command.startswith("/deny "):
                    session.deny(command.split(maxsplit=1)[1])
                elif command:
                    pending_followups.append(command)
                    console.print(Text("Follow-up queued for the next turn.", style="dim"))
                if not turn.done():
                    control_task = asyncio.create_task(
                        prompt.prompt_async("[running: /cancel, /approve ID, /deny ID] > ")
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
    prompt: PromptSession[str],
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


async def _interactive(manager: AgentSessionManager, session: AgentSession) -> None:
    terminal = PromptSession[str]()
    renderer.banner(session.session_id, session.current_model, session.workspace)
    with patch_stdout(raw=True):
        while True:
            try:
                text = (await terminal.prompt_async("nooa-code> ")).strip()
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
                session.approve(text.split(maxsplit=1)[1])
            elif text.startswith("/deny "):
                session.deny(text.split(maxsplit=1)[1])
            elif text.startswith("/replay"):
                raw = text.partition(" ")[2].strip()
                count = int(raw) if raw else 20
                for event in session.replay()[-count:]:
                    _render_event(event)
            elif text == "/reload":
                console.print(session.agent.reload_project_resources())
            elif text == "/workspace":
                console.print(str(session.workspace))
            elif text.startswith("/"):
                console.print(Text("Unknown command. Use /help.", style="red"))
            else:
                await _run_turn(session, text, terminal, allow_controls=True)
    await session.close()


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
@click.option("--yes", is_flag=True, help="Auto-approve non-denied file and shell operations.")
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
    list_sessions: bool,
) -> None:
    """Run the local, worktree-isolated NOOA Coding Agent."""
    try:
        settings = load_settings(repo, settings_file)
        if models:
            settings = settings.model_copy(
                update={
                    "models": tuple(ModelEndpoint(name=name, api_base=api_base) for name in models)
                }
            )
        if yes:
            permissions = settings.permissions.model_copy(
                update={"file_read": "allow", "file_write": "allow", "shell": "allow"}
            )
            settings = settings.model_copy(update={"permissions": permissions})
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

        async def run() -> None:
            if task:
                terminal = PromptSession[str]()
                with patch_stdout(raw=True):
                    await _run_turn(session, task, terminal, allow_controls=not yes)
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
