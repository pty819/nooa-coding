from __future__ import annotations

from io import StringIO

from rich.console import Console

from nooa_coding.agent import CodingTaskResult, VerificationResult
from nooa_coding.events import SessionEvent
from nooa_coding.terminal import TerminalRenderer, clean_terminal_text


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
    assert "**[**" not in output
