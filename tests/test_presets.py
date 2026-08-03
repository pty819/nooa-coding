"""Tests for preset sub-agent personas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nooa_coding.presets import (
    PRESET_AGENTS,
    SubAgentPreset,
    build_preset_task,
    get_preset,
    render_routing_guidance,
)

EXPECTED_PRESETS = {"search", "explore", "architect", "test", "executor", "pm"}
READ_ONLY_PRESETS = {"search", "explore", "architect", "pm"}
WRITABLE_PRESETS = {"test", "executor"}


def test_all_expected_presets_exist() -> None:
    assert EXPECTED_PRESETS.issubset(PRESET_AGENTS.keys())


def test_preset_read_only_posture() -> None:
    for name in READ_ONLY_PRESETS:
        assert PRESET_AGENTS[name].read_only is True, name
    for name in WRITABLE_PRESETS:
        assert PRESET_AGENTS[name].read_only is False, name


def test_presets_are_immutable() -> None:
    preset = PRESET_AGENTS["search"]
    with pytest.raises(ValidationError):
        preset.name = "hacked"  # type: ignore[misc]


def test_every_preset_has_role_and_description() -> None:
    for preset in PRESET_AGENTS.values():
        assert isinstance(preset, SubAgentPreset)
        assert preset.role_prompt.strip()
        assert preset.description.strip()
        assert preset.title.strip()


def test_get_preset_returns_match() -> None:
    assert get_preset("architect") is PRESET_AGENTS["architect"]


def test_get_preset_unknown_raises() -> None:
    with pytest.raises(KeyError, match="unknown sub-agent preset"):
        get_preset("does-not-exist")


def test_build_preset_task_carries_role_and_constraints() -> None:
    preset = get_preset("search")
    task = build_preset_task(preset, "find the auth handler", context_summary="ctx")
    assert task.objective == "find the auth handler"
    assert task.role == preset.role_prompt
    assert task.read_only is True
    assert task.context_summary == "ctx"
    assert list(task.constraints) == list(preset.constraints)


def test_build_preset_task_prompt_includes_role() -> None:
    preset = get_preset("executor")
    task = build_preset_task(preset, "implement the feature")
    prompt = task.build_prompt()
    assert "## Your Role" in prompt
    assert preset.role_prompt in prompt
    assert "implement the feature" in prompt


def test_build_preset_task_budget_overrides() -> None:
    preset = get_preset("test")
    task = build_preset_task(
        preset,
        "run suite",
        timeout_seconds=42,
        token_budget=1234,
    )
    assert task.timeout_seconds == 42
    assert task.token_budget == 1234
    assert task.read_only is False


def test_role_prompts_are_business_goal_system_prompts() -> None:
    """Each preset must carry a true role-specific system prompt grounded in a
    concrete business goal, methodology, and output contract — not a label."""
    for name, preset in PRESET_AGENTS.items():
        assert "You are" in preset.role_prompt, name
        assert "Business goal" in preset.role_prompt, name
        assert "Methodology" in preset.role_prompt, name
        assert "Output contract" in preset.role_prompt, name


def test_routing_guidance_lists_all_presets() -> None:
    guidance = render_routing_guidance()
    for name in EXPECTED_PRESETS:
        assert f"`{name}`" in guidance, name


def test_routing_guidance_documents_delegate_api_and_triggers() -> None:
    guidance = render_routing_guidance()
    assert "self.team.delegate" in guidance
    assert "Trigger guidance" in guidance
    assert "list_presets" in guidance
