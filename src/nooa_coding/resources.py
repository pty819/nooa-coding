"""Load repository instructions and NOOA skills into an agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from nooa import Context
from nooa.skill_registry import SkillRegistry

from .config import ResourceSettings

_BUILTIN_SKILLS_DIR = Path(__file__).parent / "skills"


class SkillDescriptor:
    """Structured metadata for one configurable Agent Skill."""

    __slots__ = ("name", "description", "trigger", "route", "output_contract", "path")

    def __init__(
        self,
        *,
        name: str,
        description: str,
        trigger: str,
        route: str,
        output_contract: str,
        path: Path,
    ) -> None:
        self.name = name
        self.description = description
        self.trigger = trigger
        self.route = route
        self.output_contract = output_contract
        self.path = path

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "description": self.description,
            "trigger": self.trigger,
            "route": self.route,
            "output_contract": self.output_contract,
            "path": str(self.path),
        }

    def __repr__(self) -> str:
        return f"SkillDescriptor(name={self.name!r}, route={self.route!r})"


class SkillManifest:
    """Registry of configurable Agent Skills with trigger/route/output-contract metadata.

    Scans SKILL.md directories, parses enhanced frontmatter, and provides
    a queryable manifest of all configured skills.
    """

    def __init__(self) -> None:
        self._skills: dict[str, SkillDescriptor] = {}

    def discover(self, directories: list[Path]) -> None:
        """Scan directories for SKILL.md folders and parse enhanced frontmatter."""
        for skills_dir in directories:
            if not skills_dir.is_dir():
                continue
            for entry in sorted(skills_dir.iterdir()):
                if not entry.is_dir():
                    continue
                skill_md = entry / "SKILL.md"
                if not skill_md.exists():
                    skill_md = entry / "skill.md"
                if not skill_md.exists():
                    continue
                descriptor = self._parse_skill_md(skill_md, entry)
                if descriptor is not None:
                    self._skills[descriptor.name] = descriptor

    @staticmethod
    def _parse_skill_md(skill_md: Path, skill_dir: Path) -> SkillDescriptor | None:
        """Parse a SKILL.md file into a SkillDescriptor."""
        try:
            content = skill_md.read_text(encoding="utf-8")
        except OSError:
            return None
        if not content.startswith("---"):
            return None
        parts = content.split("---", 2)
        if len(parts) < 3:
            return None
        try:
            meta = yaml.safe_load(parts[1]) or {}
        except yaml.YAMLError:
            return None
        if not isinstance(meta, dict):
            return None
        name = str(meta.get("name", skill_dir.name)).strip()
        return SkillDescriptor(
            name=name,
            description=str(meta.get("description", "")).strip(),
            trigger=str(meta.get("trigger", "")).strip(),
            route=str(meta.get("route", name)).strip(),
            output_contract=str(meta.get("output-contract", "")).strip(),
            path=skill_dir.resolve(),
        )

    def configured_skills(self) -> list[SkillDescriptor]:
        """Return all configured skills sorted by name."""
        return sorted(self._skills.values(), key=lambda s: s.name)

    def get(self, name: str) -> SkillDescriptor | None:
        """Look up a skill by name."""
        return self._skills.get(name)

    def routes(self) -> dict[str, str]:
        """Return a mapping of route name -> skill name."""
        return {s.route: s.name for s in self._skills.values()}

    def match_trigger(self, text: str) -> list[SkillDescriptor]:
        """Return skills whose trigger keywords appear in the given text."""
        normalized = text.lower()
        matched: list[SkillDescriptor] = []
        for skill in self._skills.values():
            # Extract quoted keywords from trigger description.
            keywords = [
                kw.strip().lower()
                for kw in skill.trigger.replace('"', "'").split("'")
                if kw.strip() and kw.strip().lower() in normalized
            ]
            if keywords:
                matched.append(skill)
        return matched

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        return name in self._skills


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
    """Install AGENTS context plus built-in, project, and user SKILL.md directories."""
    instructions = load_agents_context(repo, settings)
    if instructions:
        agent.context_manager["project_instructions"] = Context(instructions, prefix=True)

    registry = SkillRegistry(agent)
    directories: list[Path] = [
        _BUILTIN_SKILLS_DIR,
        Path("~/.config/nooa-coding/skills").expanduser(),
    ]
    for value in settings.skills_dirs:
        path = Path(value).expanduser()
        directories.append(path if path.is_absolute() else repo / path)
    registry.discover_skills_dirs(directories)
    if settings.activate_skills:
        registry.activate(list(settings.activate_skills))

    # Build the skill manifest and register builtin skills under skill.* prefix
    # so they appear in the configured skills inventory (cmd.* is excluded from status).
    manifest = build_skill_manifest(directories)
    for descriptor in manifest.configured_skills():
        reg_name = f"skill.{descriptor.name}"
        if reg_name not in registry.loaded():
            from nooa.skill import TextSkill

            try:
                text_skill = TextSkill(path=descriptor.path)
                registry.register(reg_name, text_skill)
            except Exception:
                pass
    registry.activate(["skill.*"])

    # Attach manifest to agent for runtime queries.
    agent._skill_manifest = manifest  # noqa: SLF001
    return registry


def build_skill_manifest(directories: list[Path]) -> SkillManifest:
    """Build a SkillManifest from the given skill directories."""
    manifest = SkillManifest()
    manifest.discover(directories)
    return manifest


__all__ = [
    "SkillDescriptor",
    "SkillManifest",
    "build_skill_manifest",
    "install_resources",
    "load_agents_context",
]
