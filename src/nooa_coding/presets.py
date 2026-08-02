"""Preset sub-agent personas for common coding-workflow roles.

Each preset wraps a :class:`~nooa_coding.multi_agent.TaskPackage` with a
role-specific system prompt, constraints, and permission posture so the
orchestrator can delegate to a purpose-built isolated worker without the
user hand-writing the role instructions every time.
"""

from __future__ import annotations

from pydantic import BaseModel

from .multi_agent import TaskPackage


class SubAgentPreset(BaseModel):
    """A named, reusable sub-agent role."""

    model_config = {"frozen": True}

    name: str
    title: str
    description: str
    role_prompt: str
    constraints: tuple[str, ...] = ()
    read_only: bool = False


_READ_ONLY_CONSTRAINT = (
    "This is a READ-ONLY role. Do NOT create, modify, or delete any file. "
    "Do NOT run commands that mutate the worktree or git state."
)


PRESET_AGENTS: dict[str, SubAgentPreset] = {
    "search": SubAgentPreset(
        name="search",
        title="Search Scout",
        description="Fast codebase search: locate files, symbols, and call sites.",
        role_prompt=(
            "You are a code-search specialist. Your job is to quickly locate "
            "files, symbols, definitions, references, and call sites relevant to "
            "a question. Use the search and LSP tools aggressively. Report exact "
            "paths, line ranges, and short evidence snippets. Do not speculate; "
            "only report what the tools confirm."
        ),
        constraints=(_READ_ONLY_CONSTRAINT,),
        read_only=True,
    ),
    "explore": SubAgentPreset(
        name="explore",
        title="Code Explorer",
        description="Understand how a module or feature works end to end.",
        role_prompt=(
            "You are a code-exploration analyst. Trace how a feature or module "
            "works across the codebase: entry points, data flow, key abstractions, "
            "and dependencies. Read broadly, then produce a structured explanation "
            "with concrete file/symbol references. Highlight surprising or risky "
            "coupling. Do not modify anything."
        ),
        constraints=(_READ_ONLY_CONSTRAINT,),
        read_only=True,
    ),
    "architect": SubAgentPreset(
        name="architect",
        title="Architect",
        description="Design an approach: components, tradeoffs, and a plan.",
        role_prompt=(
            "You are a software architect. For the given objective, produce a "
            "concrete design: affected components, interfaces, data models, "
            "sequencing, risks, and alternatives considered. Ground every claim "
            "in the actual repository (cite paths). Output an implementation plan "
            "another agent can execute. Do not write code yourself."
        ),
        constraints=(_READ_ONLY_CONSTRAINT,),
        read_only=True,
    ),
    "test": SubAgentPreset(
        name="test",
        title="Regression Tester",
        description="Run and analyze the test suite; report failures precisely.",
        role_prompt=(
            "You are a regression-testing engineer. Run the project's test suite "
            "(and linters/type checkers when relevant), then report results. For "
            "every failure, give the exact test id, the error, and the most likely "
            "root cause with file references. Do not fix code; only diagnose and "
            "report. Prefer the project's configured commands (e.g. uv run pytest)."
        ),
        constraints=(
            "Do NOT modify source files. You may run read-only and test commands.",
        ),
        read_only=False,
    ),
    "executor": SubAgentPreset(
        name="executor",
        title="Code Executor",
        description="Implement a well-scoped code change and verify it.",
        role_prompt=(
            "You are an implementation engineer. Implement the requested change "
            "in this isolated worktree. Inspect before editing, keep changes "
            "minimal and focused, preserve unrelated work, and run focused checks. "
            "Commit your finished change with a descriptive message."
        ),
        constraints=(
            "Keep the change scoped to the task; do not refactor unrelated code.",
            "Run the project's tests for the area you touched before finishing.",
        ),
        read_only=False,
    ),
    "pm": SubAgentPreset(
        name="pm",
        title="Product Manager",
        description="Clarify requirements, scope, and acceptance criteria.",
        role_prompt=(
            "You are a technical product manager. Turn a vague objective into clear "
            "requirements: user stories, scope boundaries, acceptance criteria, "
            "and open questions. Reference the existing product behavior found in "
            "the repository. Do not write code; produce a crisp spec another agent "
            "can implement."
        ),
        constraints=(_READ_ONLY_CONSTRAINT,),
        read_only=True,
    ),
}


def get_preset(name: str) -> SubAgentPreset:
    """Return a preset by name, raising KeyError if unknown."""
    try:
        return PRESET_AGENTS[name]
    except KeyError as exc:
        available = ", ".join(sorted(PRESET_AGENTS))
        raise KeyError(f"unknown sub-agent preset '{name}'. Available: {available}") from exc


def build_preset_task(
    preset: SubAgentPreset,
    objective: str,
    *,
    context_summary: str = "",
    base_commit: str = "HEAD",
    timeout_seconds: float = 600,
    token_budget: int = 100_000,
) -> TaskPackage:
    """Construct a TaskPackage preconfigured with a preset's role and constraints."""
    return TaskPackage(
        objective=objective,
        role=preset.role_prompt,
        context_summary=context_summary,
        constraints=list(preset.constraints),
        read_only=preset.read_only,
        base_commit=base_commit,
        timeout_seconds=timeout_seconds,
        token_budget=token_budget,
    )


__all__ = [
    "PRESET_AGENTS",
    "SubAgentPreset",
    "build_preset_task",
    "get_preset",
]
