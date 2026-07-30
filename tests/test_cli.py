from __future__ import annotations

import asyncio
from contextlib import contextmanager
from io import StringIO
from pathlib import Path

import pytest
from click.testing import CliRunner
from conftest import coding_response, fake_llm
from rich.console import Console

from nooa_coding.cli import (
    SlashCompleter,
    _auto_approve,
    _expand_at_mentions,
    _handle_mcp_command,
    _resolve_approval,
    _run_single_turn,
    main,
)
from nooa_coding.config import MCPPermissionSettings, MCPSettings, PermissionSettings
from nooa_coding.mcp import MCPServerStatus
from nooa_coding.session import AgentSessionManager


def test_cli_help_lists_interactive_product() -> None:
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "worktree-isolated" in result.output
    assert "--list-sessions" in result.output
    assert "MCP operations" in result.output


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


def test_mcp_cli_routes_management_commands(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []

    class RecordingRenderer:
        @staticmethod
        def mcp_status(statuses, errors) -> None:
            calls.append(("status", (statuses, errors)))

        @staticmethod
        def mcp_tools(tools) -> None:
            calls.append(("tools", tools))

    class StubSession:
        @staticmethod
        def mcp_status():
            return [MCPServerStatus(name="docs", status="disabled", source="settings")]

        @staticmethod
        def mcp_config_errors():
            return []

        @staticmethod
        def mcp_tools(server):
            return {server or "docs": ["search"]}

        @staticmethod
        def mcp_enable(server):
            return MCPServerStatus(name=server, status="connected", source="settings")

        @staticmethod
        def mcp_disable(server):
            return MCPServerStatus(name=server, status="disabled", source="settings")

        @staticmethod
        def mcp_reload(server):
            return [MCPServerStatus(name=server or "docs", status="connected", source="settings")]

    monkeypatch.setattr("nooa_coding.cli.renderer", RecordingRenderer())
    session = StubSession()

    for command in (
        "/mcp list",
        "/mcp tools docs",
        "/mcp enable docs",
        "/mcp disable docs",
        "/mcp reload docs",
    ):
        _handle_mcp_command(session, command)  # type: ignore[arg-type]

    assert [kind for kind, _ in calls] == ["status", "tools", "status", "status", "status"]


def test_json_output_mode_returns_structured_result(tmp_path: Path, git_repo: Path, monkeypatch) -> None:
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
    result = CliRunner().invoke(
        main,
        [
            "--repo",
            str(git_repo),
            "--settings",
            str(settings),
            "--session",
            "json-test",
            "--task",
            "write the result",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    import json as json_mod

    parsed = json_mod.loads(result.output)
    assert parsed["status"] == "completed"
    assert parsed["mode"] == "change"
    assert "summary" in parsed


def test_pipe_stdin_injects_context(tmp_path: Path, git_repo: Path, monkeypatch) -> None:
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
    # Simulate piped stdin via CliRunner input parameter.
    # Use a task that triggers the code/change route (not inspect).
    result = CliRunner().invoke(
        main,
        [
            "--repo",
            str(git_repo),
            "--settings",
            str(settings),
            "--session",
            "pipe-test",
            "--task",
            "implement the fix described below",
            "--json",
        ],
        input="ERROR: something broke\n",
    )
    assert result.exit_code == 0, result.output
    import json as json_mod

    parsed = json_mod.loads(result.output)
    assert parsed["status"] == "completed"


def test_expand_at_mentions_inlines_file_content(tmp_path: Path) -> None:
    """@file references are expanded to inline file content."""
    (tmp_path / "hello.py").write_text("print('hi')\n", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "data.txt").write_text("some data\n", encoding="utf-8")

    result = _expand_at_mentions("look at @hello.py please", tmp_path)
    assert "print('hi')" in result
    assert "@hello.py" in result

    # Directory listing.
    result2 = _expand_at_mentions("check @sub dir", tmp_path)
    assert "data.txt" in result2

    # Non-existent file is left as-is.
    result3 = _expand_at_mentions("see @missing.txt", tmp_path)
    assert "@missing.txt" in result3
    assert "---" not in result3


def test_slash_completer_provides_at_mentions(tmp_path: Path) -> None:
    """SlashCompleter yields file completions for @ queries."""
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "utils.py").write_text("y = 2\n", encoding="utf-8")

    completer = SlashCompleter(workspace=tmp_path)

    class FakeDoc:
        def __init__(self, text: str) -> None:
            self.text_before_cursor = text

    completions = list(completer.get_completions(FakeDoc("@ma"), None))
    assert any("main.py" in c.text for c in completions)

    # Slash commands still work.
    completions2 = list(completer.get_completions(FakeDoc("/he"), None))
    assert any("/help" in c.text for c in completions2)


def test_doctor_flag_prints_diagnostics(tmp_path: Path, git_repo: Path) -> None:
    """--doctor prints a health report and exits."""
    settings = tmp_path / "settings.yaml"
    settings.write_text(
        f"""coding_agent:
  models:
    - name: fixture/model
  sessions_dir: {tmp_path / "sessions"}
  worktrees_dir: {tmp_path / "worktrees"}
  memory:
    enabled: false
  compaction:
    enabled: false
""",
        encoding="utf-8",
    )
    result = CliRunner().invoke(
        main,
        ["--repo", str(git_repo), "--settings", str(settings), "--doctor"],
    )
    assert result.exit_code == 0
    assert "doctor" in result.output
    assert "git" in result.output
    assert "fixture/model" in result.output


def test_yes_mode_allows_mcp_by_default_without_removing_denies(settings) -> None:
    mcp = MCPSettings(
        enabled=True,
        permissions=MCPPermissionSettings(default="ask", deny=("*.delete",)),
    )
    configured = settings.model_copy(update={"mcp": mcp})
    approved = _auto_approve(configured)

    assert approved.permissions.file_read == "allow"
    assert approved.permissions.file_write == "allow"
    assert approved.permissions.shell == "allow"
    assert approved.mcp.permissions.default == "allow"
    assert approved.mcp.permissions.deny == ("*.delete",)
