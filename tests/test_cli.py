from __future__ import annotations

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
