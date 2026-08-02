from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from nooa_coding.agent import _cell_policy_violation
from nooa_coding.config import LimitSettings, PermissionSettings
from nooa_coding.policy import ApprovalManager, PermissionPolicy, PolicyShellTools


@pytest.mark.asyncio
async def test_file_write_waits_for_explicit_approval(tmp_path: Path) -> None:
    events = []
    approvals = ApprovalManager(lambda name, data: events.append((name, data)))
    policy = PermissionPolicy(PermissionSettings(file_write="ask"), approvals)
    shell = PolicyShellTools(tmp_path, policy, LimitSettings(), lambda *_: None)
    task = asyncio.create_task(shell.write_file("approved.txt", "ok\n"))
    await asyncio.sleep(0)

    request = approvals.pending()[0]
    assert not task.done()
    approvals.decide(request.request_id, allow=True)
    await task
    await shell.close()

    assert (tmp_path / "approved.txt").read_text(encoding="utf-8") == "ok\n"
    assert [name for name, _ in events] == ["approval_requested", "approval_resolved"]


@pytest.mark.asyncio
async def test_deny_pattern_cannot_be_auto_approved(tmp_path: Path) -> None:
    approvals = ApprovalManager(lambda *_: None)
    policy = PermissionPolicy(PermissionSettings(file_write="allow", shell="allow"), approvals)
    shell = PolicyShellTools(tmp_path, policy, LimitSettings(), lambda *_: None)
    with pytest.raises(PermissionError, match="denies"):
        await shell.run("rm -rf unsafe")
    with pytest.raises(PermissionError, match="denies"):
        await shell.run("git status; rm -rf unsafe")
    await shell.close()


@pytest.mark.asyncio
async def test_command_timeout_and_output_cap(tmp_path: Path) -> None:
    approvals = ApprovalManager(lambda *_: None)
    policy = PermissionPolicy(PermissionSettings(shell="allow"), approvals)
    shell = PolicyShellTools(
        tmp_path,
        policy,
        LimitSettings(command_timeout=0.1, max_output_chars=20),
        lambda *_: None,
    )
    timed = await shell.run("sleep 2")
    output = await shell.run("python3 -c 'print(\"x\" * 100)'", timeout=1)
    await shell.close()

    assert timed.returncode == 124
    assert len(output.stdout) < 80
    assert "truncated" in output.stdout


def test_generated_cells_must_use_policy_controlled_tools() -> None:
    assert _cell_policy_violation("open('outside.txt', 'w').write('x')") is not None
    assert _cell_policy_violation("Path('outside.txt').write_text('x')") is not None
    assert _cell_policy_violation("await self.shell._shell.run('whoami')") is not None
    assert _cell_policy_violation("await self.shell.write_file('safe.txt', 'x')") is None
    assert _cell_policy_violation("return_result(_call.return_type(summary='ok'))") is None
    assert _cell_policy_violation("'a'.replace('a', 'b')") is None


@pytest.mark.asyncio
async def test_inspection_scope_blocks_file_and_shell_mutation(tmp_path: Path) -> None:
    approvals = ApprovalManager(lambda *_: None)
    policy = PermissionPolicy(PermissionSettings(file_write="allow", shell="allow"), approvals)
    shell = PolicyShellTools(tmp_path, policy, LimitSettings(), lambda *_: None)
    try:
        async with shell._read_only_scope():
            with pytest.raises(PermissionError, match="inspection mode"):
                await shell.write_file("blocked.txt", "no\n")
            with pytest.raises(PermissionError, match="read-only shell"):
                await shell.run('python3 -c \'open("blocked.txt", "w").write("no")\'')
            result = await shell.run("pwd")
            assert result.success
    finally:
        await shell.close()
    assert not (tmp_path / "blocked.txt").exists()


@pytest.mark.asyncio
async def test_auto_allow_does_not_cover_absolute_paths(tmp_path: Path) -> None:
    approvals = ApprovalManager(lambda *_: None)
    policy = PermissionPolicy(PermissionSettings(shell="ask"), approvals)
    task = asyncio.create_task(policy.shell("rg secret /tmp"))
    await asyncio.sleep(0)
    request = approvals.pending()[0]
    approvals.decide(request.request_id, allow=False)
    with pytest.raises(PermissionError, match="denied"):
        await task


@pytest.mark.asyncio
async def test_shell_cwd_is_restored_to_worktree(tmp_path: Path) -> None:
    approvals = ApprovalManager(lambda *_: None)
    policy = PermissionPolicy(PermissionSettings(file_write="allow", shell="allow"), approvals)
    shell = PolicyShellTools(tmp_path, policy, LimitSettings(), lambda *_: None)
    await shell.run("cd .. && pwd")
    try:
        assert shell._shell.cwd == tmp_path.resolve()
        with pytest.raises(ValueError, match="escapes"):
            await shell.write_file("../outside.txt", "no")
    finally:
        await shell.close()


@pytest.mark.asyncio
async def test_single_pending_approval_resolvable_without_id(tmp_path: Path) -> None:
    """When only one approval is pending, resolving pending()[0] works (no ID needed in UI)."""
    approvals = ApprovalManager(lambda *_: None)
    policy = PermissionPolicy(PermissionSettings(shell="ask"), approvals)
    task = asyncio.create_task(policy.shell("echo hi"))
    await asyncio.sleep(0)

    pending = approvals.pending()
    assert len(pending) == 1
    # Simulate bare /approve: resolve the single pending item by its ID.
    approvals.decide(pending[0].request_id, allow=True)
    await task  # Should not raise.


@pytest.mark.asyncio
async def test_yolo_mode_allows_all_except_deny_patterns(tmp_path: Path) -> None:
    """Yolo mode (all allow) still respects deny_shell patterns."""
    approvals = ApprovalManager(lambda *_: None)
    settings = PermissionSettings(
        file_read="allow", file_write="allow", shell="allow"
    )
    policy = PermissionPolicy(settings, approvals)
    shell = PolicyShellTools(tmp_path, policy, LimitSettings(), lambda *_: None)
    try:
        # Normal commands pass without approval.
        result = await shell.run("echo yolo")
        assert result.success
        assert not approvals.pending()
        # Deny patterns still blocked.
        with pytest.raises(PermissionError, match="denies"):
            await shell.run("rm -rf /")
    finally:
        await shell.close()


def test_allowlist_does_not_auto_allow_mutating_flags() -> None:
    """find -exec and similar must NOT be auto-allowed by the allowlist."""
    approvals = ApprovalManager(lambda *_: None)
    policy = PermissionPolicy(PermissionSettings(shell="ask"), approvals)
    # find . -exec matches 'find *' glob but has mutating flag.
    assert policy.shell_mode("find . -exec touch marker {} +") == "ask"
    assert policy.shell_mode("find . -delete") == "ask"
    # Safe find commands are still auto-allowed.
    assert policy.shell_mode("find . -name '*.py'") == "allow"
    # rg with --pre is not auto-allowed.
    assert policy.shell_mode("rg --pre cat pattern") == "ask"
    # Normal rg is fine.
    assert policy.shell_mode("rg pattern") == "allow"
