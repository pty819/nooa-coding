from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from conftest import coding_response, fake_llm, response_with_code
from nooa import build_prompt_data
from nooa.agentdoc import doc
from nooa.events import Message, Summary
from nooa.unifiedllm import FakeLLMClient, LLMResponse

from nooa_coding.agent import HistorySummary, _looks_like_shell_command
from nooa_coding.config import CompactionSettings, PermissionSettings
from nooa_coding.events import TokenUsage
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
async def test_runtime_system_prompt_explains_identity_capabilities_and_config(
    git_repo: Path, settings
) -> None:
    manager = AgentSessionManager(git_repo, settings)
    session = manager.create("system-context", llm=FakeLLMClient(scripted_responses=[]))
    try:
        prompt = await build_prompt_data(
            session.agent._inspect_repository,  # pyright: ignore[reportPrivateUsage]
            "inspect the runtime identity",
        )
        system = prompt.system_prompt or ""

        assert "NOOA Coding Agent runtime" in system
        assert f"<nooa-active-model>{session.current_model}</nooa-active-model>" in system
        assert str(session.workspace) in system
        assert "doc(self)" in system
        assert ".nooa-coding/settings.yaml" in system
        assert "AGENTS.md" in system
        assert ".mcp.json" in system
        assert ".codex/skills" in system
        assert "Global files under `~/.config/nooa-coding`" in system
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_lead_agent_has_team_tool_and_routing_guidance(
    git_repo: Path, settings
) -> None:
    manager = AgentSessionManager(git_repo, settings)
    session = manager.create("lead", llm=FakeLLMClient(scripted_responses=[]))
    try:
        # The lead session can autonomously delegate to specialist sub-agents.
        assert session.agent.team is not None
        system = session.agent._runtime_system_context()  # pyright: ignore[reportPrivateUsage]
        assert "Delegating to specialist sub-agents" in system
        assert "self.team.delegate" in system
        # The team tool must be discoverable via the agent's self-doc so the
        # model knows it can call it.
        agent_api = doc(session.agent)
        assert "team" in agent_api
        # The team tool's own doc (injected as the `team_tools` context block)
        # must expose the delegate/list_presets API to the model.
        from nooa_coding.tools import TeamTools

        team_doc = doc(TeamTools)
        assert "delegate" in team_doc
        assert "list_presets" in team_doc
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_sub_agent_session_has_no_team_tool(git_repo: Path, settings) -> None:
    manager = AgentSessionManager(git_repo, settings)
    session = manager.create(
        "worker", llm=FakeLLMClient(scripted_responses=[]), is_sub_agent=True
    )
    try:
        # Sub-agents must not get the team tool (prevents recursive spawning).
        assert session.agent.team is None
        system = session.agent._runtime_system_context()  # pyright: ignore[reportPrivateUsage]
        assert "Delegating to specialist sub-agents" not in system
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
        # The pre-turn auto-checkpoint commits preexisting.txt, so the turn
        # itself produces no *new* worktree change.
        assert result.changed_files == []
        assert "no new worktree change" in result.evidence
    finally:
        await session.close()


def test_shell_command_gate_rejects_prose() -> None:
    assert _looks_like_shell_command("uv run pytest tests/")
    assert _looks_like_shell_command("python3 -c 'print(\"verified\")'")
    assert _looks_like_shell_command("./.venv/bin/python bs_iv.py")
    assert not _looks_like_shell_command("")
    assert not _looks_like_shell_command("\n".join(["echo a", "echo b"]))
    assert not _looks_like_shell_command("运行 `.venv/bin/python bs_iv.py` 验证自测输出。")
    assert not _looks_like_shell_command("Run the self test and inspect the output.")


@pytest.mark.asyncio
async def test_natural_language_suggested_verification_is_rejected(
    git_repo: Path, settings
) -> None:
    no_verification = settings.model_copy(update={"verification_commands": ()})
    manager = AgentSessionManager(git_repo, no_verification)
    prose = "运行 `.venv/bin/python bs_iv.py` 验证自测输出。可进一步用 import bs_iv 测试自定义参数。"
    session = manager.create(
        "prose-rejected", llm=fake_llm(coding_response(verification=prose))
    )
    try:
        result = await session.prompt("implement the requested change")
        assert result.status == "verification_failed"
        assert "not an executable shell command" in result.evidence
        assert [v.command for v in result.verifications] == ["git diff --check"]
        assert all(prose not in v.command for v in result.verifications)
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


def test_history_summary_coerces_structured_model_output() -> None:
    assert HistorySummary.model_validate("plain prose").value == "plain prose"
    assert HistorySummary.model_validate({"value": "kept"}).value == "kept"
    coerced = HistorySummary.model_validate(
        {"summary": "Session covered the bs_iv solver", "refs": "bs_iv.py:163"}
    )
    assert "Session covered the bs_iv solver" in coerced.value
    assert "bs_iv.py:163" in coerced.value
    assert HistorySummary.model_validate({"value": {"nested": True}}).value.startswith("{")


@pytest.mark.asyncio
async def test_compact_accepts_dict_summary_from_model(git_repo: Path, settings) -> None:
    # glm-5.2-style output: a structured object instead of the requested prose.
    dict_response = LLMResponse(
        raw_response=None,
        content=json.dumps(
            {"summary": "Session covered the bs_iv solver", "refs": {"open": "bs_iv.py:163"}}
        ),
        tool_calls=[],
        finish_reason="stop",
        assistant_message={"role": "assistant", "content": "summary"},
    )
    compacting = settings.model_copy(
        update={"compaction": CompactionSettings(enabled=False, preserve_recent=1)}
    )
    manager = AgentSessionManager(git_repo, compacting)
    session = manager.create("compact-dict", llm=fake_llm(dict_response))
    session.agent.event_manager.add(Message(content="first investigation"))
    session.agent.event_manager.add(Message(content="second investigation"))
    session.agent.event_manager.add(Message(content="recent state"))

    tag = await session.compact(preserve_recent=1)

    assert tag is not None
    summary = session.agent.events[tag]
    assert isinstance(summary, Summary)
    assert summary.summary_text is not None
    assert "Session covered the bs_iv solver" in summary.summary_text
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


@pytest.mark.asyncio
async def test_token_usage_accumulates_from_llm_complete_events(
    git_repo: Path, settings
) -> None:
    """Token usage is tracked from LLMComplete runtime events."""
    manager = AgentSessionManager(git_repo, settings)
    session = manager.create("usage", llm=fake_llm(coding_response()))
    try:
        # Before any LLM call, usage is zero.
        assert session.token_usage.llm_calls == 0
        assert session.token_usage.total_tokens == 0

        await session.prompt("create the fixture output")

        # FakeLLMClient doesn't emit LLMComplete events with usage data,
        # so usage stays zero — but the property must not raise.
        usage = session.token_usage
        assert isinstance(usage, TokenUsage)
        assert usage.total_cost_usd >= 0
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_thinking_events_are_emitted_during_turn(git_repo: Path, settings) -> None:
    """LLMCallStart/End from NOOA runtime are forwarded as thinking events."""
    manager = AgentSessionManager(git_repo, settings)
    session = manager.create("thinking", llm=fake_llm(coding_response()))
    try:
        await session.prompt("create the fixture output")

        events = session.replay()
        # The FakeLLMClient triggers the runtime which emits LLMCallStart/End.
        # These should appear as 'thinking' kind events in the session trace.
        thinking_events = [e for e in events if e.kind == "thinking"]
        # At minimum, the codeact loop fires at least one LLM call.
        started = [e for e in thinking_events if e.name == "llm_call_started"]
        ended = [e for e in thinking_events if e.name == "llm_call_ended"]
        assert len(started) >= 1
        assert len(ended) >= 1
        # Each started event carries method and turn info.
        assert started[0].data.get("method") != ""
        assert started[0].data.get("turn", 0) >= 1
    finally:
        await session.close()


def test_token_usage_model_accumulates() -> None:
    """Unit test for TokenUsage model arithmetic."""
    usage = TokenUsage()
    usage.add(prompt_tokens=100, completion_tokens=50, cost_usd=0.01)
    usage.add(prompt_tokens=200, completion_tokens=80, cached_tokens=30, cost_usd=0.02)
    assert usage.llm_calls == 2
    assert usage.total_prompt_tokens == 300
    assert usage.total_completion_tokens == 130
    assert usage.total_cached_tokens == 30
    assert usage.total_tokens == 430
    assert abs(usage.total_cost_usd - 0.03) < 1e-9


@pytest.mark.asyncio
async def test_model_switch_changes_active_client(git_repo: Path, settings) -> None:
    """Runtime model switching updates the active LLM client."""
    from nooa_coding.llm import FailoverLLM

    client_a = FakeLLMClient(scripted_responses=[])
    client_a.model = "model-a"
    client_b = FakeLLMClient(scripted_responses=[])
    client_b.model = "model-b"
    llm = FailoverLLM([client_a, client_b])

    manager = AgentSessionManager(git_repo, settings)
    session = manager.create("model-switch", llm=llm)
    try:
        assert session.current_model == "model-a"
        assert session.available_models == ["model-a", "model-b"]

        session.switch_model("model-b")
        assert session.current_model == "model-b"
        assert llm.active_index == 1

        # Switching to unknown model raises ValueError.
        with pytest.raises(ValueError, match="unknown model"):
            session.switch_model("model-z")
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_permission_switch_updates_policy(git_repo: Path, settings) -> None:
    """Runtime permission switching updates the policy settings."""
    manager = AgentSessionManager(git_repo, settings)
    session = manager.create("perm-switch", llm=FakeLLMClient(scripted_responses=[]))
    try:
        # Default from fixture: file_write=allow, shell=allow.
        policy = session.agent.shell._policy  # noqa: SLF001
        assert policy.settings.file_write == "allow"

        session.switch_permissions("ask")
        assert policy.settings.file_write == "ask"
        assert policy.settings.shell == "ask"

        session.switch_permissions("allow")
        assert policy.settings.file_write == "allow"
        assert policy.settings.shell == "allow"

        session.switch_permissions("yolo")
        assert policy.settings.file_write == "allow"
        assert policy.settings.shell == "allow"

        with pytest.raises(ValueError, match="unknown permission mode"):
            session.switch_permissions("invalid")
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_permission_switch_auto_edit(git_repo: Path, settings) -> None:
    """auto-edit auto-approves file ops but still asks for shell."""
    manager = AgentSessionManager(git_repo, settings)
    session = manager.create("auto-edit", llm=FakeLLMClient(scripted_responses=[]))
    try:
        policy = session.agent.shell._policy  # noqa: SLF001
        session.switch_permissions("auto-edit")
        assert policy.settings.file_read == "allow"
        assert policy.settings.file_write == "allow"
        assert policy.settings.shell == "ask"
        # Aliases resolve to the same posture.
        session.switch_permissions("auto")
        assert policy.settings.shell == "ask"
        assert policy.settings.file_write == "allow"
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_undo_reverts_last_turn_change(git_repo: Path, settings) -> None:
    """Each turn is bracketed by an auto-checkpoint that /undo can revert."""
    manager = AgentSessionManager(git_repo, settings)
    session = manager.create("undo", llm=fake_llm(coding_response()))
    try:
        result = await session.prompt("create the fixture output")
        assert result.status == "completed"
        assert (session.workspace / "result.txt").exists()

        checkpoint = session.undo()
        assert checkpoint.label.startswith("auto-turn")
        assert not (session.workspace / "result.txt").exists()
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_undo_without_any_turn_raises(git_repo: Path, settings) -> None:
    manager = AgentSessionManager(git_repo, settings)
    session = manager.create("undo-empty", llm=FakeLLMClient(scripted_responses=[]))
    try:
        with pytest.raises(KeyError, match="nothing to undo"):
            session.undo()
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_context_window_resolution_and_current_tokens(
    git_repo: Path, settings
) -> None:
    manager = AgentSessionManager(git_repo, settings)
    session = manager.create("ctx", llm=fake_llm(coding_response()))
    try:
        assert session.current_context_tokens == 0
        with patch("nooa_coding.llm.resolve_context_window", return_value=99_000):
            assert session.context_window == 99_000
        # Resolved value is cached on the session.
        assert session.context_window == 99_000
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_review_returns_no_changes_message_on_clean_worktree(
    git_repo: Path, settings
) -> None:
    """Review on a clean worktree returns a friendly message without calling the LLM."""
    manager = AgentSessionManager(git_repo, settings)
    session = manager.create("review-clean", llm=FakeLLMClient(scripted_responses=[]))
    try:
        result = await session.review()
        assert "No changes to review" in result
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_plan_generates_output(git_repo: Path, settings) -> None:
    """Plan mode calls the LLM and returns content."""
    from nooa.unifiedllm import LLMResponse

    plan_response = LLMResponse(
        raw_response=None,
        content="# Plan\n1. Step one\n2. Step two",
        tool_calls=[],
        finish_reason="stop",
        assistant_message={"role": "assistant", "content": "plan"},
    )
    manager = AgentSessionManager(git_repo, settings)
    session = manager.create("plan-test", llm=fake_llm(plan_response))
    try:
        result = await session.plan("refactor the parser module")
        assert "Plan" in result
        assert "Step one" in result
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_delegate_parallel_runs_subtasks_in_forks(git_repo: Path, settings) -> None:
    """delegate_parallel spawns child sessions that run independently."""
    manager = AgentSessionManager(git_repo, settings)
    session = manager.create("parent", llm=fake_llm(coding_response()))
    try:
        results = await manager.delegate_parallel(
            session,
            ["create the fixture output", "create the fixture output"],
            llm_factory=lambda: fake_llm(coding_response()),
        )
        assert len(results) == 2
        assert all(r["status"] == "completed" for r in results)
        assert results[0]["session_id"] == "parent-sub0"
        assert results[1]["session_id"] == "parent-sub1"
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_goal_set_evaluate_and_achieve(git_repo: Path, settings) -> None:
    """Goal mode: set goal, run turn, evaluator confirms achieved."""
    from nooa.unifiedllm import LLMResponse

    # First response: coding task result. Second: evaluator says ACHIEVED.
    eval_response = LLMResponse(
        raw_response=None,
        content="ACHIEVED: all tests pass and lint is clean",
        tool_calls=[],
        finish_reason="stop",
        assistant_message={"role": "assistant", "content": "eval"},
    )
    manager = AgentSessionManager(git_repo, settings)
    session = manager.create(
        "goal-test", llm=fake_llm(coding_response(), eval_response)
    )
    try:
        # Set a goal.
        goal = session.set_goal("all tests pass", turn_budget=5)
        assert goal.status == "active"
        assert goal.turn_budget == 5
        assert session.goal is not None

        # Run a turn.
        result = await session.prompt("create the fixture output")
        assert result.status == "completed"

        # Evaluate the goal.
        achieved, evaluation = await session.evaluate_goal(result)
        assert achieved is True
        assert "ACHIEVED" in evaluation
        assert session.goal.status == "achieved"
        assert session.goal.turns_used == 1
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_goal_not_achieved_continues(git_repo: Path, settings) -> None:
    """Goal mode: evaluator says NOT_ACHIEVED, goal stays active."""
    from nooa.unifiedllm import LLMResponse

    eval_response = LLMResponse(
        raw_response=None,
        content="NOT_ACHIEVED: tests still failing in test_auth.py",
        tool_calls=[],
        finish_reason="stop",
        assistant_message={"role": "assistant", "content": "eval"},
    )
    manager = AgentSessionManager(git_repo, settings)
    session = manager.create(
        "goal-continue", llm=fake_llm(coding_response(), eval_response)
    )
    try:
        session.set_goal("all tests pass", turn_budget=5)
        result = await session.prompt("fix the tests")

        achieved, evaluation = await session.evaluate_goal(result)
        assert achieved is False
        assert "NOT_ACHIEVED" in evaluation
        assert session.goal.status == "active"
        assert session.goal.turns_used == 1
        assert session.goal.turns_remaining == 4
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_goal_budget_exhaustion(git_repo: Path, settings) -> None:
    """Goal mode: budget exhaustion stops the loop without calling LLM."""
    from nooa_coding.agent import CodingTaskResult

    manager = AgentSessionManager(git_repo, settings)
    session = manager.create("goal-exhaust", llm=FakeLLMClient(scripted_responses=[]))
    try:
        session.set_goal("impossible goal", turn_budget=2)
        # Simulate prior turns used up the budget.
        session.goal.turns_used = 2

        mock_result = CodingTaskResult(
            mode="change",
            status="completed",
            summary="did something",
            evidence="some evidence",
        )
        achieved, evaluation = await session.evaluate_goal(mock_result)
        assert achieved is False
        assert "exhausted" in evaluation.lower()
        assert session.goal.status == "exhausted"
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_goal_clear(git_repo: Path, settings) -> None:
    """Goal mode: clear removes the active goal."""
    manager = AgentSessionManager(git_repo, settings)
    session = manager.create("goal-clear", llm=FakeLLMClient(scripted_responses=[]))
    try:
        session.set_goal("some objective")
        assert session.goal is not None
        session.clear_goal()
        assert session.goal is None
    finally:
        await session.close()
