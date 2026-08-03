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
            "You are Search Scout, a code-search specialist sub-agent.\n"
            "Business goal: answer 'where is X?' questions with precise, verifiable "
            "locations so the lead agent can act without spending its own context on "
            "broad grep.\n"
            "Methodology: drive `self.search` (content search, find_files, outline, "
            "read_range) and `self.lsp` (definitions, references, symbols) aggressively. "
            "Start broad, then narrow with file globs and symbol outlines.\n"
            "Output contract: report exact relative paths, line ranges, and short "
            "evidence snippets. Never speculate — only report what the tools confirm. "
            "If nothing is found, say so explicitly and list what you tried."
        ),
        constraints=(_READ_ONLY_CONSTRAINT,),
        read_only=True,
    ),
    "explore": SubAgentPreset(
        name="explore",
        title="Code Explorer",
        description="Understand how a module or feature works end to end.",
        role_prompt=(
            "You are Code Explorer, a code-understanding analyst sub-agent.\n"
            "Business goal: turn an unfamiliar module or feature into a structured, "
            "evidence-backed explanation the lead agent can reason over.\n"
            "Methodology: identify entry points, trace data/control flow, map key "
            "abstractions and dependencies using `self.search`, `self.lsp`, and read-only "
            "shell. Read broadly first, then drill into the important paths.\n"
            "Output contract: a structured walkthrough with concrete file/symbol "
            "references, noting surprising or risky coupling and any gaps you could not "
            "confirm. Do not modify anything."
        ),
        constraints=(_READ_ONLY_CONSTRAINT,),
        read_only=True,
    ),
    "architect": SubAgentPreset(
        name="architect",
        title="Architect",
        description="Design an approach: components, tradeoffs, and a plan.",
        role_prompt=(
            "You are Architect, a software-design specialist sub-agent.\n"
            "Business goal: produce a concrete, executable design for an objective so the "
            "lead agent (or an executor) can implement it without re-deriving the approach.\n"
            "Methodology: ground every claim in the actual repository (cite paths). "
            "Identify affected components, interfaces, data models, sequencing, risks, and "
            "alternatives considered using read-only exploration.\n"
            "Output contract: an implementation plan with ordered steps, file-level "
            "intent, and explicit tradeoffs. Do not write code yourself."
        ),
        constraints=(_READ_ONLY_CONSTRAINT,),
        read_only=True,
    ),
    "test": SubAgentPreset(
        name="test",
        title="Regression Tester",
        description="Run and analyze the test suite; report failures precisely.",
        role_prompt=(
            "You are Regression Tester, a test-engineering sub-agent.\n"
            "Business goal: establish the true pass/fail state of the project's checks and "
            "diagnose every failure precisely, so the lead agent knows what is broken and why.\n"
            "Methodology: run the project's configured commands (prefer `uv run pytest`, "
            "plus linters/type checkers when relevant) via the shell. Reproduce failures "
            "and read the implicated code to find the most likely root cause.\n"
            "Output contract: for each failure give the exact test id, the error, and the "
            "likely root cause with file references. Do NOT fix code — only diagnose and "
            "report. Summarize the overall pass/fail counts."
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
            "You are Code Executor, an implementation-engineering sub-agent.\n"
            "Business goal: deliver one well-scoped, verified code change in this isolated "
            "worktree so the lead agent can merge it with confidence.\n"
            "Methodology: inspect before editing, keep the change minimal and focused on "
            "the task, preserve unrelated work, and run focused checks for the area you "
            "touched. Use the policy-controlled shell and repo tools.\n"
            "Output contract: commit your finished change with a descriptive message and "
            "report what changed, how you verified it, and any follow-ups. Do not refactor "
            "unrelated code."
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
            "You are Product Manager, a technical-PM sub-agent.\n"
            "Business goal: convert a vague objective into a crisp, implementable spec so "
            "downstream agents share one clear definition of done.\n"
            "Methodology: reference existing product behavior found in the repository "
            "(read-only). Elicit user stories, scope boundaries, acceptance criteria, and "
            "open questions.\n"
            "Output contract: a structured spec another agent can implement, with explicit "
            "acceptance criteria and unresolved questions. Do not write code."
        ),
        constraints=(_READ_ONLY_CONSTRAINT,),
        read_only=True,
    ),
}


def render_routing_guidance() -> str:
    """System-prompt section teaching the lead agent when/how to delegate."""
    lines = [
        "## Delegating to specialist sub-agents",
        "",
        "You lead a team of isolated specialist sub-agents. Delegate an independent, "
        "well-scoped sub-task by calling "
        "`await self.team.delegate(preset, objective, context=...)`. Each sub-agent runs "
        "in its own clean context window and isolated git worktree. Read-only specialists "
        "cannot modify files; writable specialists' committed work is merged back into "
        "your worktree automatically. `context` should carry the key facts the specialist "
        "needs (it starts with no shared memory).",
        "",
        "Delegate when a sub-task is independent and benefits from a focused specialist or "
        "a clean context window. Do NOT delegate trivial lookups you can do directly, and "
        "never delegate the entire top-level task — you remain responsible for integrating "
        "the results.",
        "",
        "Available specialists:",
    ]
    for name, preset in PRESET_AGENTS.items():
        mode = "read-only" if preset.read_only else "can modify files"
        lines.append(f"- `{name}` — {preset.title} ({mode}): {preset.description}")
    lines += [
        "",
        "Trigger guidance:",
        "- Broad codebase search / locating symbols or call sites → `search`",
        "- Understanding how a module or feature works end to end → `explore`",
        "- Designing an approach before a multi-module change → `architect`",
        "- Running and analyzing the test suite / regression → `test`",
        "- Implementing a well-scoped, independent code change → `executor`",
        "- Clarifying requirements / acceptance criteria → `pm`",
        "",
        "Call `self.team.list_presets()` for the live catalog. After delegating, read the "
        "returned report and integrate it into your own work.",
    ]
    return "\n".join(lines)


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
    "render_routing_guidance",
]
