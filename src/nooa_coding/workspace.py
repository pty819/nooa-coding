"""Dedicated Git worktrees and recoverable coding checkpoints."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel


class WorkspaceInfo(BaseModel):
    base_repo: str
    path: str
    branch: str
    start_ref: str
    isolated: bool = True


class Checkpoint(BaseModel):
    checkpoint_id: str
    commit: str
    label: str
    created_at: str
    snapshot_id: str | None = None


class DiffResult(BaseModel):
    status: str
    stat: str
    patch: str


def _run_git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if check and result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {message}")
    return result


class WorkspaceManager:
    """Create one branch-backed worktree per coding session."""

    def __init__(self, base_repo: str | Path, root: str | Path) -> None:
        self.base_repo = Path(base_repo).expanduser().resolve()
        if (
            not (self.base_repo / ".git").exists()
            and _run_git(self.base_repo, "rev-parse", "--git-dir", check=False).returncode
        ):
            raise ValueError(f"worktree isolation requires a Git repository: {self.base_repo}")
        digest = hashlib.sha256(str(self.base_repo).encode()).hexdigest()[:12]
        self.root = Path(root).expanduser().resolve() / digest

    @staticmethod
    def _safe_id(session_id: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "-", session_id).strip("-.")
        if not safe:
            raise ValueError("session id has no safe worktree name")
        return safe

    def path_for(self, session_id: str) -> Path:
        return self.root / self._safe_id(session_id)

    def create(self, session_id: str, *, start_ref: str = "HEAD") -> WorkspaceInfo:
        path = self.path_for(session_id)
        branch = f"nooa-coding/{self._safe_id(session_id)}"
        if path.exists():
            raise FileExistsError(f"worktree already exists: {path}")
        branch_exists = (
            _run_git(
                self.base_repo, "show-ref", "--verify", f"refs/heads/{branch}", check=False
            ).returncode
            == 0
        )
        if branch_exists:
            raise FileExistsError(f"session branch already exists: {branch}")
        path.parent.mkdir(parents=True, exist_ok=True)
        _run_git(self.base_repo, "worktree", "add", "-b", branch, str(path), start_ref)
        return WorkspaceInfo(
            base_repo=str(self.base_repo),
            path=str(path),
            branch=branch,
            start_ref=_run_git(path, "rev-parse", "HEAD").stdout.strip(),
        )

    def inspect(self, session_id: str) -> WorkspaceInfo:
        path = self.path_for(session_id)
        if not path.is_dir():
            raise FileNotFoundError(f"session worktree is missing: {path}")
        branch = _run_git(path, "branch", "--show-current").stdout.strip()
        return WorkspaceInfo(
            base_repo=str(self.base_repo),
            path=str(path),
            branch=branch,
            start_ref=_run_git(path, "rev-list", "--max-parents=0", "HEAD").stdout.strip(),
        )

    @staticmethod
    def diff(workspace: str | Path, *, max_chars: int = 200_000) -> DiffResult:
        root = Path(workspace)
        status = _run_git(root, "status", "--short").stdout
        stat = _run_git(root, "diff", "--stat", "HEAD", "--").stdout
        patch = _run_git(root, "diff", "HEAD", "--").stdout
        untracked = _run_git(root, "ls-files", "--others", "--exclude-standard", "-z").stdout.split(
            "\0"
        )
        new_files: list[str] = []
        for relative in (item for item in untracked if item):
            path = root / relative
            if not path.is_file():
                continue
            new_files.append(f" {relative} | new file")
            remaining = max_chars - len(patch)
            if remaining <= 0:
                break
            generated = _run_git(
                root,
                "diff",
                "--no-index",
                "--",
                "/dev/null",
                relative,
                check=False,
            ).stdout
            patch += generated[:remaining]
        if new_files:
            stat = "\n".join(new_files) + ("\n" + stat if stat else "\n")
        if len(patch) > max_chars:
            patch = patch[:max_chars] + "\n... (diff truncated)"
        return DiffResult(status=status, stat=stat, patch=patch)

    @staticmethod
    def checkpoint(workspace: str | Path, label: str) -> Checkpoint:
        root = Path(workspace)
        _run_git(root, "add", "-A")
        _run_git(
            root,
            "-c",
            "user.name=NOOA Coding Agent",
            "-c",
            "user.email=nooa-coding@localhost",
            "commit",
            "--allow-empty",
            "-m",
            f"checkpoint: {label}",
        )
        commit = _run_git(root, "rev-parse", "HEAD").stdout.strip()
        return Checkpoint(
            checkpoint_id=commit[:12],
            commit=commit,
            label=label,
            created_at=datetime.now(UTC).isoformat(),
        )

    @staticmethod
    def rollback(workspace: str | Path, checkpoint: Checkpoint) -> None:
        root = Path(workspace)
        exists = _run_git(root, "cat-file", "-e", f"{checkpoint.commit}^{{commit}}", check=False)
        if exists.returncode != 0:
            raise ValueError(f"checkpoint commit is unavailable: {checkpoint.checkpoint_id}")
        _run_git(root, "reset", "--hard", checkpoint.commit)
        _run_git(root, "clean", "-fd")

    @staticmethod
    def write_metadata(path: Path, info: WorkspaceInfo) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(info.model_dump(), indent=2) + "\n", encoding="utf-8")


__all__ = ["Checkpoint", "DiffResult", "WorkspaceInfo", "WorkspaceManager"]
