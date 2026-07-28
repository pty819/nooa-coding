from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from click.testing import CliRunner
from conftest import coding_response, fake_llm

from nooa_coding.cli import main


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
