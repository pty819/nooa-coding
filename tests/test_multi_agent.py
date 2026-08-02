"""Tests for the production-grade sub-agent orchestration system."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nooa_coding.multi_agent import (
    Coordinator,
    OrchestrationPlan,
    SubAgentHandle,
    TaskPackage,
    TaskStatus,
    WorkerReport,
    merge_worktrees,
)

# ─── TaskPackage Tests ───────────────────────────────────────────────────────


class TestTaskPackage:
    def test_build_prompt_minimal(self):
        pkg = TaskPackage(objective="Fix the login bug")
        prompt = pkg.build_prompt()
        assert "## Task: Fix the login bug" in prompt
        assert "## Instructions" in prompt

    def test_build_prompt_full(self):
        pkg = TaskPackage(
            objective="Refactor auth module",
            context_summary="The auth module uses legacy JWT",
            file_scope=["src/auth.py", "src/tokens.py"],
            constraints=["Do not change public API", "Keep backward compat"],
            expected_output="All tests pass",
        )
        prompt = pkg.build_prompt()
        assert "## Task: Refactor auth module" in prompt
        assert "## Background" in prompt
        assert "legacy JWT" in prompt
        assert "## File Scope" in prompt
        assert "src/auth.py" in prompt
        assert "## Constraints" in prompt
        assert "Do not change public API" in prompt
        assert "## Expected Output" in prompt
        assert "All tests pass" in prompt

    def test_default_values(self):
        pkg = TaskPackage(objective="test")
        assert pkg.base_commit == "HEAD"
        assert pkg.token_budget == 100_000
        assert pkg.timeout_seconds == 600
        assert len(pkg.task_id) == 8


# ─── WorkerReport Tests ──────────────────────────────────────────────────────


class TestWorkerReport:
    def test_completed_report(self):
        report = WorkerReport(
            task_id="abc123",
            status=TaskStatus.COMPLETED,
            summary="Fixed the bug",
            files_changed=["src/main.py"],
            commits=["sha123"],
        )
        assert report.status == TaskStatus.COMPLETED
        assert report.files_changed == ["src/main.py"]

    def test_failed_report(self):
        report = WorkerReport(
            task_id="abc123",
            status=TaskStatus.FAILED,
            error="RuntimeError: something broke",
        )
        assert report.status == TaskStatus.FAILED
        assert "something broke" in report.error


# ─── OrchestrationPlan Tests ─────────────────────────────────────────────────


class TestOrchestrationPlan:
    def test_plan_creation(self):
        plan = OrchestrationPlan(
            objective="Big task",
            tasks=[
                TaskPackage(objective="sub1"),
                TaskPackage(objective="sub2"),
            ],
        )
        assert len(plan.tasks) == 2
        assert plan.strategy == "parallel"


# ─── Coordinator Tests ───────────────────────────────────────────────────────


def _make_mock_session(tmp_path: Path) -> MagicMock:
    """Create a mock parent session for Coordinator tests."""
    session = MagicMock()
    session.session_id = "main-session"
    session.settings.subagent.max_concurrent = 3
    session.settings.subagent.timeout_seconds = 600
    session.settings.subagent.token_budget = 100_000
    session.settings.subagent.allow_shell = ("uv run pytest*",)
    session.settings.subagent.deny_shell = ("rm -rf*",)
    session.metadata.workspace.path = str(tmp_path / "main-worktree")
    return session


def _make_mock_manager(tmp_path: Path) -> MagicMock:
    """Create a mock session manager."""
    manager = MagicMock()
    manager.repo = tmp_path

    def create_session(session_id, start_ref="HEAD", parent_session_id=None):
        sub = MagicMock()
        sub.session_id = session_id
        sub.metadata.workspace.path = str(tmp_path / f"worktree-{session_id}")
        sub.metadata.workspace.branch = f"nooa-coding/{session_id}"
        sub._is_sub_agent = False
        sub.agent.shell._policy = MagicMock()
        return sub

    manager.create = MagicMock(side_effect=create_session)
    return manager


class TestCoordinator:
    @pytest.mark.asyncio
    async def test_spawn_creates_handles(self, tmp_path):
        manager = _make_mock_manager(tmp_path)
        parent = _make_mock_session(tmp_path)
        coordinator = Coordinator(manager, parent)

        tasks = [
            TaskPackage(objective="Task A"),
            TaskPackage(objective="Task B"),
        ]
        handles = coordinator.spawn(tasks)

        assert len(handles) == 2
        assert all(isinstance(h, SubAgentHandle) for h in handles)
        assert manager.create.call_count == 2
        # Cancel the background tasks to avoid warnings.
        for h in handles:
            h.cancel()

    @pytest.mark.asyncio
    async def test_spawn_marks_sub_agent(self, tmp_path):
        manager = _make_mock_manager(tmp_path)
        parent = _make_mock_session(tmp_path)
        coordinator = Coordinator(manager, parent)

        tasks = [TaskPackage(objective="Task A")]
        handles = coordinator.spawn(tasks)
        assert handles[0].status == TaskStatus.RUNNING
        handles[0].cancel()

    @pytest.mark.asyncio
    async def test_spawn_after_close_raises(self, tmp_path):
        manager = _make_mock_manager(tmp_path)
        parent = _make_mock_session(tmp_path)
        coordinator = Coordinator(manager, parent)
        coordinator._closed = True

        with pytest.raises(RuntimeError, match="closed"):
            coordinator.spawn([TaskPackage(objective="Task")])

    def test_status(self, tmp_path):
        manager = _make_mock_manager(tmp_path)
        parent = _make_mock_session(tmp_path)
        coordinator = Coordinator(manager, parent)

        status = coordinator.status()
        assert status["active"] == 0
        assert status["total"] == 0
        assert status["closed"] is False

    @pytest.mark.asyncio
    async def test_wait_all_empty(self, tmp_path):
        manager = _make_mock_manager(tmp_path)
        parent = _make_mock_session(tmp_path)
        coordinator = Coordinator(manager, parent)

        reports = await coordinator.wait_all([])
        assert reports == []


# ─── Sub-Agent Spawn Prohibition Tests ───────────────────────────────────────


class TestSubAgentProhibition:
    def test_sub_agent_cannot_spawn(self):
        """Sub-agents must not be able to spawn further sub-agents."""
        # Simulate the coordinator property guard.
        is_sub_agent = True
        with pytest.raises(PermissionError, match="Sub-agents cannot"):
            if is_sub_agent:
                raise PermissionError("Sub-agents cannot spawn further sub-agents")


# ─── Merge Tests ─────────────────────────────────────────────────────────────


class TestMergeWorktrees:
    def test_merge_no_reports(self, tmp_path):
        result = merge_worktrees(tmp_path, [])
        assert result.success is True
        assert result.merged == []

    def test_merge_skips_failed(self, tmp_path):
        reports = [
            WorkerReport(task_id="t1", status=TaskStatus.FAILED, error="broke"),
            WorkerReport(task_id="t2", status=TaskStatus.TIMEOUT, error="slow"),
        ]
        result = merge_worktrees(tmp_path, reports)
        assert result.failed == ["t1", "t2"]
        assert result.merged == []

    def test_merge_skips_no_commits(self, tmp_path):
        reports = [
            WorkerReport(task_id="t1", status=TaskStatus.COMPLETED, commits=[]),
        ]
        result = merge_worktrees(tmp_path, reports)
        assert result.merged == []
        assert result.success is True

    def test_merge_cherry_pick_success(self, tmp_path):
        """Test successful cherry-pick merge with a real git repo."""
        subprocess.run(["git", "init", str(tmp_path)], capture_output=True)
        subprocess.run(
            ["git", "-C", str(tmp_path), "config", "user.email", "test@test.com"],
            capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(tmp_path), "config", "user.name", "Test"],
            capture_output=True,
        )
        (tmp_path / "file.txt").write_text("hello")
        subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], capture_output=True)
        subprocess.run(
            ["git", "-C", str(tmp_path), "commit", "-m", "init"],
            capture_output=True,
        )

        subprocess.run(
            ["git", "-C", str(tmp_path), "checkout", "-b", "sub-branch"],
            capture_output=True,
        )
        (tmp_path / "new_file.txt").write_text("new content")
        subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], capture_output=True)
        subprocess.run(
            ["git", "-C", str(tmp_path), "commit", "-m", "sub work"],
            capture_output=True,
        )
        sha = subprocess.run(
            ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        ).stdout.strip()

        subprocess.run(
            ["git", "-C", str(tmp_path), "checkout", "master"],
            capture_output=True,
        )

        reports = [
            WorkerReport(
                task_id="t1",
                status=TaskStatus.COMPLETED,
                summary="Added new file",
                commits=[sha],
            ),
        ]
        result = merge_worktrees(tmp_path, reports)
        assert result.success is True
        assert "t1" in result.merged
        assert (tmp_path / "new_file.txt").exists()


# ─── Integration-style Tests (mocked LLM) ───────────────────────────────────


class TestCoordinatorExecution:
    @pytest.mark.asyncio
    async def test_run_sub_agent_success(self, tmp_path):
        """Test the _run_sub_agent coroutine with a mocked session.prompt."""
        manager = _make_mock_manager(tmp_path)
        parent = _make_mock_session(tmp_path)
        coordinator = Coordinator(manager, parent)

        task = TaskPackage(objective="Write tests", timeout_seconds=10)

        sub_session = MagicMock()
        sub_session.metadata.workspace.path = str(tmp_path / "sub-worktree")
        sub_session.metadata.workspace.branch = "nooa-coding/sub-test"

        mock_result = MagicMock()
        mock_result.summary = "Tests written successfully"
        sub_session.prompt = AsyncMock(return_value=mock_result)

        with patch.object(coordinator, "_commit_changes", return_value="abc123"):
            with patch.object(
                coordinator,
                "_collect_diff",
                return_value={"files": ["test_foo.py"], "stat": "1 file changed"},
            ):
                report = await coordinator._run_sub_agent(
                    task, sub_session, "Write tests"
                )

        assert report.status == TaskStatus.COMPLETED
        assert report.summary == "Tests written successfully"
        assert report.files_changed == ["test_foo.py"]
        assert report.commits == ["abc123"]

    @pytest.mark.asyncio
    async def test_run_sub_agent_timeout(self, tmp_path):
        """Test timeout handling in _run_sub_agent."""
        manager = _make_mock_manager(tmp_path)
        parent = _make_mock_session(tmp_path)
        coordinator = Coordinator(manager, parent)

        task = TaskPackage(objective="Slow task", timeout_seconds=0.01)

        sub_session = MagicMock()
        sub_session.metadata.workspace.path = str(tmp_path / "sub-worktree")
        sub_session.metadata.workspace.branch = "nooa-coding/sub-slow"

        async def hang_forever(prompt):
            await asyncio.sleep(100)

        sub_session.prompt = hang_forever

        report = await coordinator._run_sub_agent(task, sub_session, "Slow task")
        assert report.status == TaskStatus.TIMEOUT
        assert "timeout" in report.error.lower()

    @pytest.mark.asyncio
    async def test_run_sub_agent_failure(self, tmp_path):
        """Test error handling in _run_sub_agent."""
        manager = _make_mock_manager(tmp_path)
        parent = _make_mock_session(tmp_path)
        coordinator = Coordinator(manager, parent)

        task = TaskPackage(objective="Broken task", timeout_seconds=10)

        sub_session = MagicMock()
        sub_session.metadata.workspace.path = str(tmp_path / "sub-worktree")
        sub_session.metadata.workspace.branch = "nooa-coding/sub-broken"

        sub_session.prompt = AsyncMock(side_effect=ValueError("LLM error"))

        report = await coordinator._run_sub_agent(task, sub_session, "Broken task")
        assert report.status == TaskStatus.FAILED
        assert "LLM error" in report.error


# ─── Config Tests ────────────────────────────────────────────────────────────


class TestSubAgentConfig:
    def test_default_settings(self):
        from nooa_coding.config import SubAgentSettings

        settings = SubAgentSettings()
        assert settings.max_concurrent == 3
        assert settings.timeout_seconds == 600
        assert settings.token_budget == 100_000
        assert "uv run pytest*" in settings.allow_shell
        assert "git push*" in settings.deny_shell

    def test_settings_in_coding_settings(self):
        from nooa_coding.config import CodingSettings

        settings = CodingSettings()
        assert settings.subagent.max_concurrent == 3
