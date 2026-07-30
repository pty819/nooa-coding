from __future__ import annotations

from pathlib import Path

from nooa_coding.config import load_settings, settings_paths
from nooa_coding.resources import load_agents_context


def test_layered_settings_target_explicit_repo(tmp_path: Path, monkeypatch) -> None:
    user = tmp_path / "user"
    repo = tmp_path / "repo"
    user.mkdir()
    (repo / ".nooa-coding").mkdir(parents=True)
    (user / "settings.yaml").write_text(
        "coding_agent:\n  permissions:\n    shell: deny\n  limits:\n    command_timeout: 30\n",
        encoding="utf-8",
    )
    (repo / ".nooa-coding" / "settings.yaml").write_text(
        "coding_agent:\n  permissions:\n    shell: ask\n  memory:\n    embedding:\n      backend: litellm\n      model: openai/text-embedding-3-small\n  mcp:\n    enabled: true\n    enabled_servers: [docs]\n    permissions:\n      default: ask\n      read_only: [docs.search]\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".config" / "nooa-coding").mkdir(parents=True)
    (tmp_path / ".config" / "nooa-coding" / "settings.yaml").write_text(
        (user / "settings.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )

    loaded = load_settings(repo)

    assert loaded.permissions.shell == "ask"
    assert loaded.limits.command_timeout == 30
    assert loaded.memory.embedding.backend == "litellm"
    assert loaded.memory.embedding.model == "openai/text-embedding-3-small"
    assert loaded.mcp.enabled is True
    assert loaded.mcp.enabled_servers == ("docs",)
    assert loaded.mcp.permissions.read_only == ("docs.search",)
    assert settings_paths(repo) == [
        (tmp_path / ".config" / "nooa-coding" / "settings.yaml").resolve(),
        (repo / ".nooa-coding" / "settings.yaml").resolve(),
    ]


def test_agents_context_loads_root_and_nested_files(git_repo: Path, settings) -> None:
    (git_repo / "AGENTS.md").write_text("root instructions", encoding="utf-8")
    nested = git_repo / "src"
    nested.mkdir()
    (nested / "AGENTS.md").write_text("nested instructions", encoding="utf-8")

    content = load_agents_context(git_repo, settings.resources)

    assert "root instructions" in content
    assert "nested instructions" in content
    assert content.count("root instructions") == 1
