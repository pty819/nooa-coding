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
    assert _cell_policy_violation("'a'.replace('a', 'b')") is None


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
