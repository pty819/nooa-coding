from __future__ import annotations

import asyncio
from contextlib import contextmanager
from io import StringIO
from pathlib import Path

import pytest
from click.testing import CliRunner
from conftest import coding_response, fake_llm
from rich.console import Console

from nooa_coding.cli import _resolve_approval, _run_single_turn, main
from nooa_coding.config import PermissionSettings
from nooa_coding.session import AgentSessionManager


def test_cli_help_lists_interactive_product() -> None:
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "worktree-isolated" in result.output
    assert "--list-sessions" in result.output


def test_one_shot_cli_uses_formal_session(tmp_path: Path, git_repo: Path, monkeypatch) -> None:
    settings = tmp_path / "settings.yaml"
    settings.write_text(
        f"""coding_agent:
  models:
    - name: fixture/model
  sessions_dir: {tmp_path / "sessions"}
  worktrees_dir: {tmp_path / "worktrees"}
  verification_commands:
    - python3 -c 'print("verified")'
  memory:
    enabled: false
  compaction:
    enabled: false
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "nooa_coding.session.build_llm", lambda *args, **kwargs: fake_llm(coding_response())
    )
    patch_calls = []

    @contextmanager
    def capture_patch_stdout(**kwargs):
        patch_calls.append(kwargs)
        yield

    monkeypatch.setattr("nooa_coding.cli.patch_stdout", capture_patch_stdout)

    result = CliRunner().invoke(
        main,
        [
            "--repo",
            str(git_repo),
            "--settings",
            str(settings),
            "--session",
            "cli-task",
            "--task",
            "write the result",
            "--yes",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "completed" in result.output
    assert "turn_started" not in result.output
    assert '"status"' not in result.output
    assert "?[" not in result.output
    assert patch_calls == [{"raw": True}]
    listing = CliRunner().invoke(
        main,
        [
            "--repo",
            str(git_repo),
            "--settings",
            str(settings),
            "--list-sessions",
        ],
    )
    assert listing.exit_code == 0
    assert "cli-task" in listing.output


@pytest.mark.asyncio
async def test_cancel_does_not_reopen_running_prompt(git_repo: Path, settings) -> None:
    asking = settings.model_copy(
        update={"permissions": PermissionSettings(file_write="ask", shell="allow")}
    )
    manager = AgentSessionManager(git_repo, asking)
    session = manager.create("cli-cancel", llm=fake_llm(coding_response()))

    class CancelWhenApprovalAppears:
        calls = 0

        async def prompt_async(self, message: str) -> str:
            del message
            self.calls += 1
            if self.calls > 1:
                raise AssertionError("running prompt reopened after cancellation")
            for _ in range(200):
                if session.approvals.pending():
                    return "/cancel"
                await asyncio.sleep(0.005)
            raise AssertionError("approval was not requested")

    prompt = CancelWhenApprovalAppears()
    try:
        queued = await _run_single_turn(session, "write a file", prompt, allow_controls=True)
        assert queued == []
        assert prompt.calls == 1
        assert session.status == "cancelled"
        assert session.approvals.pending() == []
    finally:
        await session.close()


def test_stale_approval_is_a_warning_not_an_exception(monkeypatch) -> None:
    class NoPendingApproval:
        @staticmethod
        def approve(request_id: str) -> None:
            del request_id
            raise KeyError("already resolved")

        @staticmethod
        def deny(request_id: str) -> None:
            del request_id
            raise KeyError("already resolved")

    output = StringIO()
    monkeypatch.setattr("nooa_coding.cli.console", Console(file=output, force_terminal=False))

    resolved = _resolve_approval(NoPendingApproval(), "expired-request", allow=False)

    assert resolved is False
    assert "no longer pending" in output.getvalue()
