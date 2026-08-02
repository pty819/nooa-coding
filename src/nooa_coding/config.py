"""Typed, project-aware configuration owned by the coding application."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from nooa_memory import MemoryConfig
from pydantic import BaseModel, ConfigDict, Field, model_validator

PermissionMode = Literal["allow", "ask", "deny"]


class ModelEndpoint(BaseModel):
    """One model in failover order."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    api_base: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None
    client_type: Literal["completion", "responses"] | None = None


class CompactionSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    context_fraction: float = Field(default=0.8, gt=0, le=1)
    max_tokens: int | None = Field(default=None, gt=0)
    preserve_recent: int = Field(default=10, ge=1)
    target_chars: int = Field(default=4_000, gt=0)


class MemorySettings(MemoryConfig):  # pyright: ignore[reportGeneralTypeIssues]
    """The complete nooa-memory schema with application-friendly defaults."""

    enabled: bool = True
    path: str = ".nooa-coding/memory.sqlite"
    owner: str = "CodingAgent"


class PermissionSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    file_read: PermissionMode = "allow"
    file_write: PermissionMode = "ask"
    shell: PermissionMode = "ask"
    allow_shell: tuple[str, ...] = (
        "git status*",
        "git diff*",
        "git log*",
        "git show*",
        "rg *",
        "find *",
        "ls*",
        "pwd",
        "uv run pytest*",
        "uv run ruff*",
        "uv run pyright*",
    )
    deny_shell: tuple[str, ...] = (
        "rm -rf*",
        "git reset --hard*",
        "git clean -f*",
        "git push --force*",
    )


class LimitSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    command_timeout: float = Field(default=120, gt=0)
    verification_timeout: float = Field(default=300, gt=0)
    max_output_chars: int = Field(default=100_000, gt=0)
    max_stdin_chars: int = Field(default=1_000_000, gt=0)


class ResourceSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    agents_files: tuple[str, ...] = ("AGENTS.md",)
    skills_dirs: tuple[str, ...] = (
        ".codex/skills",
        ".nooa/skills",
        ".nooa-coding/skills",
    )
    activate_skills: tuple[str, ...] = ("cmd.*",)
    max_context_chars: int = Field(default=100_000, gt=0)


class MCPPermissionSettings(BaseModel):
    """Policy for external MCP calls, matched as ``server.tool`` globs."""

    model_config = ConfigDict(frozen=True)

    default: PermissionMode = "ask"
    allow: tuple[str, ...] = ()
    deny: tuple[str, ...] = ()
    read_only: tuple[str, ...] = ()


class MCPSettings(BaseModel):
    """External MCP servers consumed by the coding agent."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    config_files: tuple[str, ...] = (
        "~/.config/nooa-coding/mcp.json",
        ".mcp.json",
    )
    enabled_servers: tuple[str, ...] = ("*",)
    disabled_servers: tuple[str, ...] = ()
    servers: dict[str, dict[str, Any]] = Field(default_factory=dict)
    fail_on_error: bool = False
    call_timeout: float = Field(default=60, gt=0)
    max_output_chars: int = Field(default=100_000, gt=0)
    permissions: MCPPermissionSettings = Field(default_factory=MCPPermissionSettings)


class SubAgentSettings(BaseModel):
    """Configuration for sub-agent spawning and execution."""

    model_config = ConfigDict(frozen=True)

    max_concurrent: int = Field(default=3, ge=1, le=10)
    timeout_seconds: float = Field(default=600, gt=0)
    token_budget: int = Field(default=100_000, gt=0)
    allow_shell: tuple[str, ...] = (
        "uv run pytest*",
        "uv run ruff*",
        "uv run pyright*",
        "rg *",
        "find *",
        "ls*",
        "pwd",
        "git status*",
        "git diff*",
        "git log*",
        "git show*",
    )
    deny_shell: tuple[str, ...] = (
        "rm -rf*",
        "git push*",
        "git reset --hard*",
        "git clean*",
        "git merge*",
        "git rebase*",
    )


class CodingSettings(BaseModel):
    """Complete local application settings."""

    model_config = ConfigDict(frozen=True)

    models: tuple[ModelEndpoint, ...] = ()
    sessions_dir: str = "~/.local/share/nooa-coding/sessions"
    worktrees_dir: str = "~/.local/share/nooa-coding/worktrees"
    verification_commands: tuple[str, ...] = ()
    compaction: CompactionSettings = Field(default_factory=CompactionSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    permissions: PermissionSettings = Field(default_factory=PermissionSettings)
    limits: LimitSettings = Field(default_factory=LimitSettings)
    resources: ResourceSettings = Field(default_factory=ResourceSettings)
    mcp: MCPSettings = Field(default_factory=MCPSettings)
    subagent: SubAgentSettings = Field(default_factory=SubAgentSettings)

    @model_validator(mode="before")
    @classmethod
    def accept_single_model(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        raw = dict(value)
        if "models" not in raw and isinstance(raw.get("model"), str):
            raw["models"] = [{"name": raw.pop("model")}]
        return raw

    def sessions_path(self) -> Path:
        return Path(self.sessions_dir).expanduser().resolve()

    def worktrees_path(self) -> Path:
        return Path(self.worktrees_dir).expanduser().resolve()


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read settings {path}: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"settings root must be a mapping: {path}")
    return loaded


def settings_paths(repo: str | Path, explicit: str | Path | None = None) -> list[Path]:
    """Return existing setting layers from low to high priority."""
    root = Path(repo).expanduser().resolve()
    candidates = [
        Path("~/.config/nooa-coding/settings.yaml").expanduser(),
        root / ".nooa-coding" / "settings.yaml",
    ]
    env_path = os.environ.get("NOOA_CODING_SETTINGS")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    if explicit is not None:
        candidates.append(Path(explicit).expanduser())

    result: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file() and resolved not in seen:
            result.append(resolved)
            seen.add(resolved)
    return result


def load_settings(repo: str | Path, explicit: str | Path | None = None) -> CodingSettings:
    """Load application-owned settings without relying on NOOA internals."""
    root = Path(repo).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"repo must be an existing directory: {root}")
    merged: dict[str, Any] = {}
    for path in settings_paths(root, explicit):
        merged = _deep_merge(merged, _read_yaml(path))
    raw = merged.get("coding_agent", merged)
    if not isinstance(raw, dict):
        raise ValueError("coding_agent settings must be a mapping")
    return CodingSettings.model_validate(raw)


__all__ = [
    "CodingSettings",
    "CompactionSettings",
    "LimitSettings",
    "MemorySettings",
    "MCPPermissionSettings",
    "MCPSettings",
    "ModelEndpoint",
    "PermissionSettings",
    "ResourceSettings",
    "SubAgentSettings",
    "load_settings",
    "settings_paths",
]
