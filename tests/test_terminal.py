from __future__ import annotations

from io import StringIO
from pathlib import Path

from rich.console import Console

from nooa_coding.agent import CodingTaskResult, VerificationResult
from nooa_coding.events import SessionEvent
from nooa_coding.mcp import MCPServerStatus
from nooa_coding.terminal import TerminalRenderer, ThinkingSpinner, clean_terminal_text


def _renderer() -> tuple[TerminalRenderer, StringIO]:
    stream = StringIO()
    return TerminalRenderer(Console(file=stream, force_terminal=False, width=100)), stream


def test_terminal_text_removes_real_and_mangled_ansi_sequences() -> None:
    value = "\x1b[1mNOOA\x1b[0m and ?[1;36;40mGLM?[0m"
    assert clean_terminal_text(value) == "NOOA and GLM"


def test_renderer_hides_internal_lifecycle_events_and_cleans_messages() -> None:
    renderer, stream = _renderer()
    renderer.event(SessionEvent(sequence=1, session_id="s", kind="session", name="turn_started"))
    assert stream.getvalue() == ""

    renderer.event(
        SessionEvent(
            sequence=2,
            session_id="s",
            kind="message",
            name="message",
            data={"content": "\x1b[1manswer\x1b[0m ?[31mclean?[0m"},
        )
    )
    output = stream.getvalue()
    assert "answer clean" in output
    assert "?[" not in output
    assert "\x1b" not in output


def test_renderer_outputs_a_human_result_instead_of_json() -> None:
    renderer, stream = _renderer()
    renderer.result(
        CodingTaskResult(
            mode="change",
            status="completed",
            summary="Implemented the fix.",
            changed_files=["src/app.py"],
            evidence="Focused checks passed.",
            verifications=[VerificationResult(command="uv run pytest", passed=True, returncode=0)],
            model="fixture/model",
        )
    )
    output = stream.getvalue()
    assert "change · completed" in output
    assert "Implemented the fix" in output
    assert "src/app.py" in output
    assert '"status"' not in output


def test_help_renders_optional_arguments_literally() -> None:
    renderer, stream = _renderer()
    renderer.help()
    output = stream.getvalue()
    assert "/fork [SESSION_ID]" in output
    assert "/mcp tools [SERVER]" in output
    assert "**[**" not in output


def test_renderer_shows_mcp_status_and_call_events_without_arguments() -> None:
    renderer, stream = _renderer()
    renderer.mcp_status(
        [
            MCPServerStatus(
                name="docs",
                status="connected",
                source=".mcp.json",
                attribute="mcp_docs",
                tools=["search"],
            )
        ],
        [],
    )
    renderer.event(
        SessionEvent(
            sequence=1,
            session_id="s",
            kind="tool",
            name="mcp_call_started",
            data={"server": "docs", "tool": "search", "argument_names": ["secret"]},
        )
    )
    output = stream.getvalue()
    assert "mcp_docs" in output
    assert "docs" in output and "search" in output
    assert "secret" not in output


def test_banner_shows_logo_and_session_info() -> None:
    renderer, stream = _renderer()
    renderer.banner("test-session", "openai/gpt-5", Path("/tmp/wt"))
    output = stream.getvalue()
    assert "NOOA" in output
    assert "test-session" in output
    assert "gpt-5" in output
    assert "/tmp/wt" in output


def test_user_input_renders_separator_and_preview() -> None:
    renderer, stream = _renderer()
    renderer.user_input("implement the login feature")
    output = stream.getvalue()
    assert "implement the login feature" in output
    # Should contain a rule/separator.
    assert "─" in output or "━" in output or "❯" in output


def test_thinking_spinner_lifecycle() -> None:
    stream = StringIO()
    console = Console(file=stream, force_terminal=False, width=100)
    spinner = ThinkingSpinner(console)
    spinner.start("Thinking")
    spinner.update("Running: pytest")
    spinner.stop("Done")
    output = stream.getvalue()
    assert "Thinking" in output
    assert "Running: pytest" in output
    assert "Done" in output


def test_goal_progress_renders_bar() -> None:
    renderer, stream = _renderer()
    renderer.goal_progress("all tests pass", 3, 10, "still failing")
    output = stream.getvalue()
    assert "Goal" in output
    assert "3/10" in output
    assert "█" in output
    assert "░" in output
    assert "still failing" in output
