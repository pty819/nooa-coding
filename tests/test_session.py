from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from conftest import coding_response, fake_llm, response_with_code
from nooa.agentdoc import doc
from nooa.events import Message, Summary
from nooa.unifiedllm import FakeLLMClient, LLMResponse

from nooa_coding.config import CompactionSettings, PermissionSettings
from nooa_coding.session import AgentSessionManager


@pytest.mark.asyncio
async def test_session_runs_in_worktree_streams_replays_and_resumes(
    git_repo: Path, settings
) -> None:
    manager = AgentSessionManager(git_repo, settings)
    session = manager.create("primary", llm=fake_llm(coding_response()))

    result = await session.prompt("create the fixture output")

    assert result.status == "completed"
    assert (session.workspace / "result.txt").read_text(encoding="utf-8") == "done\n"
    assert not (git_repo / "result.txt").exists()
    assert any(item.name == "turn_finished" for item in session.replay())
    historical = [item async for item in session.stream(follow=False)]
    assert [item.sequence for item in historical] == sorted({item.sequence for item in historical})
    workspace = session.workspace
    await session.close()

    resumed = manager.resume("primary", llm=FakeLLMClient(scripted_responses=[]))
    try:
        assert resumed.workspace == workspace
        assert resumed.agent.task == "create the fixture output"
        assert resumed.agent.last_result is not None
        assert resumed.agent.last_result.status == "completed"
    finally:
        await resumed.close()


@pytest.mark.asyncio
async def test_model_identity_comes_from_live_session_without_an_llm_call(
    git_repo: Path, settings
) -> None:
    manager = AgentSessionManager(git_repo, settings)
    session = manager.create("identity", llm=FakeLLMClient(scripted_responses=[]))
    try:
        result = await session.prompt("你是什么模型呢？")
        assert result.mode == "conversation"
        assert result.status == "answered"
        assert result.model == session.current_model
        assert session.current_model in result.summary
        assert all(event.name != "LLMComplete" for event in session.replay())
        agent_api = doc(session.agent)
        assert "notify" in agent_api
        assert "message(" not in agent_api
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_repository_question_routes_to_read_only_inspection(git_repo: Path, settings) -> None:
    response = response_with_code(
        "return_result(InspectionDraft(status='completed', "
        "summary='The function can be simplified.', "
        "evidence='README.md was inspected.'))"
    )
    manager = AgentSessionManager(git_repo, settings)
    session = manager.create("inspect", llm=fake_llm(response))
    try:
        result = await session.prompt("这个函数应该怎么优化？")
        assert result.mode == "inspect"
        assert result.status == "inspected"
        assert result.changed_files == []
        assert session.diff().status == ""
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_change_completion_requires_a_new_host_observed_change(
    git_repo: Path, settings
) -> None:
    manager = AgentSessionManager(git_repo, settings)
    session = manager.create("host-evidence", llm=fake_llm(coding_response(write_file=False)))
    (session.workspace / "preexisting.txt").write_text("already here\n", encoding="utf-8")
    try:
        result = await session.prompt("implement the requested change")
        assert result.status == "verification_failed"
        assert result.changed_files == ["preexisting.txt"]
        assert "no new worktree change" in result.evidence
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_session_approval_and_cancel(git_repo: Path, settings) -> None:
    asking = settings.model_copy(
        update={
            "permissions": PermissionSettings(file_write="ask", shell="allow"),
        }
    )
    manager = AgentSessionManager(git_repo, asking)
    approved = manager.create("approved", llm=fake_llm(coding_response()))
    turn = approved.start("write an approved file")
    for _ in range(100):
        if approved.approvals.pending():
            break
        await asyncio.sleep(0.01)
    request = approved.approvals.pending()[0]
    assert not turn.done()
    approved.approve(request.request_id)
    assert (await turn).status == "completed"
    await approved.close()

    cancelled = manager.create("cancelled", llm=fake_llm(coding_response()))
    turn = cancelled.start("write a file but stop")
    for _ in range(100):
        if cancelled.approvals.pending():
            break
        await asyncio.sleep(0.01)
    assert await cancelled.cancel() is True
    with pytest.raises(asyncio.CancelledError):
        await turn
    assert cancelled.status == "cancelled"
    assert not (cancelled.workspace / "result.txt").exists()
    await cancelled.close()


@pytest.mark.asyncio
async def test_explicit_compaction_collapses_old_events(git_repo: Path, settings) -> None:
    summary_response = LLMResponse(
        raw_response=None,
        content=json.dumps("Preserved decisions, evidence, and remaining work."),
        tool_calls=[],
        finish_reason="stop",
        assistant_message={"role": "assistant", "content": "summary"},
    )
    compacting = settings.model_copy(
        update={"compaction": CompactionSettings(enabled=False, preserve_recent=1)}
    )
    manager = AgentSessionManager(git_repo, compacting)
    session = manager.create("compact", llm=fake_llm(summary_response))
    session.agent.event_manager.add(Message(content="first investigation"))
    session.agent.event_manager.add(Message(content="second investigation"))
    session.agent.event_manager.add(Message(content="recent state"))

    tag = await session.compact(preserve_recent=1)

    assert tag is not None
    summary = session.agent.events[tag]
    assert isinstance(summary, Summary)
    assert summary.summary_text is not None
    assert "Preserved decisions" in summary.summary_text
    await session.close()


@pytest.mark.asyncio
async def test_checkpoint_rollback_and_fork(git_repo: Path, settings) -> None:
    manager = AgentSessionManager(git_repo, settings)
    session = manager.create("source", llm=fake_llm(coding_response()))
    await session.prompt("create a result")
    checkpoint = session.checkpoint("known-good")
    (session.workspace / "result.txt").write_text("broken\n", encoding="utf-8")

    session.rollback(checkpoint.checkpoint_id)

    assert (session.workspace / "result.txt").read_text(encoding="utf-8") == "done\n"
    child = await session.fork("child", llm=FakeLLMClient(scripted_responses=[]))
    try:
        assert child.metadata.parent_session_id == "source"
        assert child.workspace != session.workspace
        assert (child.workspace / "result.txt").read_text(encoding="utf-8") == "done\n"
        assert child.agent.last_result is not None
    finally:
        await child.close()
        await session.close()


@pytest.mark.asyncio
async def test_resume_marks_interrupted_turn_as_crash_recovered(git_repo: Path, settings) -> None:
    manager = AgentSessionManager(git_repo, settings)
    session = manager.create("crash", llm=FakeLLMClient(scripted_responses=[]))
    session._set_status("running")
    await session.close()

    recovered = manager.resume("crash", llm=FakeLLMClient(scripted_responses=[]))
    try:
        assert recovered.metadata.recovered_after_crash is True
        assert recovered.status == "idle"
        assert any(
            event.name == "session_resumed" and event.data["crash_recovery"]
            for event in recovered.replay()
        )
    finally:
        await recovered.close()


def test_session_listing_is_project_scoped(git_repo: Path, settings) -> None:
    manager = AgentSessionManager(git_repo, settings)
    session = manager.create("listed", llm=FakeLLMClient(scripted_responses=[]))
    summaries = manager.list()
    asyncio.run(session.close())

    assert [item.session_id for item in summaries] == ["listed"]
    assert summaries[0].checkpoint_count == 1


@pytest.mark.asyncio
async def test_project_memory_is_shared_across_sessions(git_repo: Path, settings) -> None:
    enabled = settings.model_copy(
        update={"memory": settings.memory.model_copy(update={"enabled": True})}
    )
    manager = AgentSessionManager(git_repo, enabled)
    first = manager.create("memory-one", llm=FakeLLMClient(scripted_responses=[]))
    memory_id = first.agent.remember(
        "Run parser checks with uv run pytest tests/parser.",
        type="skill",
        tags=["parser", "pytest"],
    )
    await first.close()

    second = manager.create("memory-two", llm=FakeLLMClient(scripted_responses=[]))
    try:
        recalled = second.agent.recall("parser pytest", k=3)
        assert any(item.id == memory_id for item in recalled)
        assert Path(second.settings.memory.path).is_file()
    finally:
        await second.close()


@pytest.mark.asyncio
async def test_agents_and_skills_are_loaded_from_worktree(git_repo: Path, settings) -> None:
    from conftest import run_git

    (git_repo / "AGENTS.md").write_text("Always run fixture checks.\n", encoding="utf-8")
    skill_dir = git_repo / ".codex" / "skills" / "fixture-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: fixture-skill\ndescription: Validate fixture behavior.\n---\nUse focused checks.\n",
        encoding="utf-8",
    )
    run_git(git_repo, "add", "AGENTS.md", ".codex/skills/fixture-skill/SKILL.md")
    run_git(
        git_repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "add agent resources",
    )
    manager = AgentSessionManager(git_repo, settings)
    session = manager.create("resources", llm=FakeLLMClient(scripted_responses=[]))
    try:
        assert "cmd.fixture-skill" in session.agent.skills.activated()
        rendered = str(session.agent.context_manager["project_instructions"])
        assert "Always run fixture checks" in rendered
    finally:
        await session.close()
