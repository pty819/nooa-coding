from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest
from conftest import fake_llm, response_with_code, run_git
from nooa.agentdoc import doc
from nooa.unifiedllm import FakeLLMClient

from nooa_coding.config import MCPPermissionSettings, MCPSettings
from nooa_coding.mcp import MCPRuntime
from nooa_coding.policy import ApprovalManager
from nooa_coding.session import AgentSessionManager


def _commit_mcp_config(git_repo: Path, server: dict) -> None:
    (git_repo / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"fixture": server}}, indent=2),
        encoding="utf-8",
    )
    run_git(git_repo, "add", ".mcp.json")
    run_git(
        git_repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "add mcp fixture",
    )


def _stdio_server() -> dict:
    script = Path(__file__).parent / "assets" / "mcp_echo_server.py"
    return {
        "command": sys.executable,
        "args": [str(script.resolve())],
        "transport": "stdio",
    }


def _with_mcp(settings, *, permissions: MCPPermissionSettings, max_output_chars: int = 100_000):
    return settings.model_copy(
        update={
            "mcp": MCPSettings(
                enabled=True,
                config_files=(".mcp.json",),
                enabled_servers=("fixture",),
                call_timeout=10,
                max_output_chars=max_output_chars,
                permissions=permissions,
            )
        }
    )


@pytest.mark.asyncio
async def test_stdio_mcp_is_parsed_injected_and_called(git_repo: Path, settings) -> None:
    _commit_mcp_config(git_repo, _stdio_server())
    configured = _with_mcp(
        settings,
        permissions=MCPPermissionSettings(
            default="allow",
            read_only=("fixture.echo",),
        ),
        max_output_chars=24,
    )
    response = response_with_code(
        "value = await self.mcp_fixture.echo(text='hello')\n"
        "return_result(InspectionDraft(status='completed', summary=value, "
        "evidence='external MCP result observed'))"
    )
    session = AgentSessionManager(git_repo, configured).create("mcp-stdio", llm=fake_llm(response))
    try:
        status = session.mcp_status()[0]
        assert status.status == "connected"
        assert status.attribute == "mcp_fixture"
        assert status.tools == ["echo", "large_output"]
        rendered_api = doc(session.agent)
        assert "mcp_fixture" in rendered_api
        assert "echo" in rendered_api

        result = await session.prompt("inspect the external knowledge source")
        assert result.status == "inspected"
        assert result.summary == "external:hello"

        async with session.agent._mcp.read_only():
            with pytest.raises(PermissionError, match="inspection mode denies"):
                await session.agent.mcp_fixture.large_output(size=80)  # type: ignore[attr-defined]
        large = await session.agent.mcp_fixture.large_output(size=80)  # type: ignore[attr-defined]
        assert isinstance(large, str)
        assert len(large) < 100
        assert "MCP output truncated" in large
        names = [event.name for event in session.replay()]
        assert "mcp_server_connected" in names
        assert "mcp_call_started" in names
        assert "mcp_call_finished" in names

        assert session.mcp_disable("fixture").status == "disabled"
        assert not hasattr(session.agent, "mcp_fixture")
        assert session.mcp_enable("fixture").status == "connected"
        assert hasattr(session.agent, "mcp_fixture")
        assert session.mcp_reload("fixture")[0].status == "connected"

        (session.workspace / ".mcp.json").unlink()
        with pytest.raises(KeyError, match="unknown MCP server"):
            session.mcp_reload("fixture")
        assert session.mcp_status() == []
        assert not hasattr(session.agent, "mcp_fixture")
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_mcp_call_uses_the_existing_approval_flow(git_repo: Path, settings) -> None:
    _commit_mcp_config(git_repo, _stdio_server())
    configured = _with_mcp(
        settings,
        permissions=MCPPermissionSettings(
            default="ask",
            read_only=("fixture.echo",),
        ),
    )
    response = response_with_code(
        "value = await self.mcp_fixture.echo(text='approved')\n"
        "return_result(InspectionDraft(status='completed', summary=value, evidence='approved'))"
    )
    session = AgentSessionManager(git_repo, configured).create(
        "mcp-approval", llm=fake_llm(response)
    )
    try:
        turn = session.start("inspect the external MCP source")
        for _ in range(200):
            if session.approvals.pending():
                break
            await asyncio.sleep(0.01)
        request = session.approvals.pending()[0]
        assert request.kind == "mcp"
        assert request.resource == "fixture.echo"
        session.approve(request.request_id)
        assert (await turn).summary == "external:approved"
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_broken_mcp_server_isolated_from_the_session(git_repo: Path, settings) -> None:
    _commit_mcp_config(
        git_repo,
        {
            "command": "definitely-missing-nooa-coding-mcp-command",
            "transport": "stdio",
        },
    )
    configured = _with_mcp(
        settings,
        permissions=MCPPermissionSettings(default="allow"),
    )
    session = AgentSessionManager(git_repo, configured).create(
        "mcp-broken", llm=FakeLLMClient(scripted_responses=[])
    )
    try:
        status = session.mcp_status()[0]
        assert status.status == "failed"
        assert status.error
        assert not hasattr(session.agent, "mcp_fixture")
        result = await session.prompt("你是什么模型呢？")
        assert result.status == "answered"
        assert any(event.name == "mcp_server_failed" for event in session.replay())
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_mcp_runtime_enforces_timeout_and_deny_policy(tmp_path: Path) -> None:
    events: list[tuple[str, dict]] = []
    runtime = MCPRuntime(
        tmp_path,
        MCPSettings(
            enabled=True,
            config_files=(),
            call_timeout=0.01,
            permissions=MCPPermissionSettings(default="allow", deny=("blocked.*",)),
        ),
        ApprovalManager(lambda name, data: events.append((name, data))),
        lambda name, data: events.append((name, data)),
    )

    async def slow() -> str:
        await asyncio.sleep(1)
        return "late"

    with pytest.raises(TimeoutError):
        await runtime.call("fixture", "slow", slow, (), {})
    with pytest.raises(PermissionError, match="policy denies"):
        await runtime.call("blocked", "delete", slow, (), {})
    assert [name for name, _ in events] == ["mcp_call_started", "mcp_call_failed"]


def test_invalid_mcp_config_is_reported_or_can_fail_fast(tmp_path: Path) -> None:
    (tmp_path / ".mcp.json").write_text("{broken", encoding="utf-8")
    approvals = ApprovalManager(lambda *_: None)
    tolerant = MCPRuntime(
        tmp_path,
        MCPSettings(enabled=True, config_files=(".mcp.json",)),
        approvals,
        lambda *_: None,
    )
    assert "cannot parse MCP config" in tolerant.config_errors[0]

    with pytest.raises(ValueError, match="cannot parse MCP config"):
        MCPRuntime(
            tmp_path,
            MCPSettings(enabled=True, config_files=(".mcp.json",), fail_on_error=True),
            approvals,
            lambda *_: None,
        )
