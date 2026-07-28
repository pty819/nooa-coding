"""Load repository instructions and NOOA skills into an agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nooa import Context
from nooa.skill_registry import SkillRegistry

from .config import ResourceSettings


def load_agents_context(repo: Path, settings: ResourceSettings) -> str:
    """Load configured AGENTS files, including nested files, under one size cap."""
    requested: list[Path] = []
    for value in settings.agents_files:
        path = Path(value).expanduser()
        requested.append(path if path.is_absolute() else repo / path)

    if "AGENTS.md" in settings.agents_files:
        requested.extend(sorted(repo.glob("**/AGENTS.md")))

    seen: set[Path] = set()
    chunks: list[str] = []
    used = 0
    for candidate in requested:
        resolved = candidate.resolve()
        if resolved in seen or not resolved.is_file():
            continue
        try:
            resolved.relative_to(repo)
        except ValueError:
            # Explicit absolute paths are allowed; discovered paths must stay local.
            if not Path(candidate).is_absolute():
                continue
        text = resolved.read_text(encoding="utf-8")
        relative = resolved.relative_to(repo) if resolved.is_relative_to(repo) else resolved
        rendered = f"## {relative}\n\n{text.strip()}\n"
        remaining = settings.max_context_chars - used
        if remaining <= 0:
            break
        chunks.append(rendered[:remaining])
        used += min(len(rendered), remaining)
        seen.add(resolved)
    return "\n".join(chunks)


def install_resources(agent: Any, repo: Path, settings: ResourceSettings) -> SkillRegistry:
    """Install AGENTS context plus project and user SKILL.md directories."""
    instructions = load_agents_context(repo, settings)
    if instructions:
        agent.context_manager["project_instructions"] = Context(instructions, prefix=True)

    registry = SkillRegistry(agent)
    directories: list[Path] = [Path("~/.config/nooa-coding/skills").expanduser()]
    for value in settings.skills_dirs:
        path = Path(value).expanduser()
        directories.append(path if path.is_absolute() else repo / path)
    registry.discover_skills_dirs(directories)
    if settings.activate_skills:
        registry.activate(list(settings.activate_skills))
    return registry


__all__ = ["install_resources", "load_agents_context"]
