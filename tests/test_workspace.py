from __future__ import annotations

from pathlib import Path

from nooa_coding.workspace import WorkspaceManager


def test_worktree_checkpoint_diff_and_rollback(git_repo: Path, tmp_path: Path) -> None:
    manager = WorkspaceManager(git_repo, tmp_path / "worktrees")
    info = manager.create("session-one")
    workspace = Path(info.path)
    assert workspace != git_repo
    assert info.branch == "nooa-coding/session-one"

    (workspace / "feature.txt").write_text("one\n", encoding="utf-8")
    diff = manager.diff(workspace)
    assert "?? feature.txt" in diff.status
    assert "feature.txt | new file" in diff.stat
    assert "+one" in diff.patch

    checkpoint = manager.checkpoint(workspace, "feature")
    (workspace / "feature.txt").write_text("two\n", encoding="utf-8")
    manager.rollback(workspace, checkpoint)

    assert (workspace / "feature.txt").read_text(encoding="utf-8") == "one\n"
