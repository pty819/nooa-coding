"""Analyze a repository and generate a starter AGENTS.md."""

from __future__ import annotations

from pathlib import Path

# ─── Language / framework detection ──────────────────────────────────────────

_MARKERS: dict[str, list[str]] = {
    "python": ["pyproject.toml", "setup.py", "setup.cfg", "requirements.txt"],
    "javascript": ["package.json", "next.config.js", "next.config.mjs"],
    "typescript": ["tsconfig.json", "next.config.ts"],
    "rust": ["Cargo.toml"],
    "go": ["go.mod"],
    "java": ["pom.xml", "build.gradle", "build.gradle.kts"],
    "ruby": ["Gemfile"],
    "csharp": ["*.sln", "*.csproj"],
}

_TEST_COMMANDS: dict[str, str] = {
    "python": "uv run pytest tests/ -x -q",
    "javascript": "npm test",
    "typescript": "npm test",
    "rust": "cargo test",
    "go": "go test ./...",
    "java": "mvn test",
    "ruby": "bundle exec rspec",
    "csharp": "dotnet test",
}

_LINT_COMMANDS: dict[str, str] = {
    "python": "uv run ruff check src/",
    "javascript": "npx eslint .",
    "typescript": "npx tsc --noEmit",
    "rust": "cargo clippy",
    "go": "golangci-lint run",
}


def _detect_languages(repo: Path) -> list[str]:
    """Detect project languages by marker files."""
    found: list[str] = []
    for language, markers in _MARKERS.items():
        for marker in markers:
            if "*" in marker:
                if list(repo.glob(marker)):
                    found.append(language)
                    break
            elif (repo / marker).exists():
                found.append(language)
                break
    return found


def _detect_package_manager(repo: Path) -> str | None:
    if (repo / "uv.lock").exists():
        return "uv"
    if (repo / "poetry.lock").exists():
        return "poetry"
    if (repo / "pnpm-lock.yaml").exists():
        return "pnpm"
    if (repo / "yarn.lock").exists():
        return "yarn"
    if (repo / "package-lock.json").exists():
        return "npm"
    if (repo / "Cargo.lock").exists():
        return "cargo"
    if (repo / "go.sum").exists():
        return "go"
    return None


def _detect_src_layout(repo: Path) -> str | None:
    """Detect the primary source directory."""
    candidates = ["src", "lib", "app", "pkg", "cmd", "internal"]
    for name in candidates:
        path = repo / name
        if path.is_dir() and any(path.iterdir()):
            return name
    return None


def _detect_ci(repo: Path) -> list[str]:
    """Detect CI configuration files."""
    ci_files: list[str] = []
    github_workflows = repo / ".github" / "workflows"
    if github_workflows.is_dir():
        ci_files.extend(f".github/workflows/{f.name}" for f in github_workflows.glob("*.yml"))
    if (repo / ".gitlab-ci.yml").exists():
        ci_files.append(".gitlab-ci.yml")
    if (repo / "Jenkinsfile").exists():
        ci_files.append("Jenkinsfile")
    return ci_files


def _count_files(repo: Path, suffix: str, *, max_walk: int = 500) -> int:
    count = 0
    for _ in repo.rglob(f"*{suffix}"):
        count += 1
        if count >= max_walk:
            break
    return count


# ─── AGENTS.md generation ────────────────────────────────────────────────────


def generate_agents_md(repo: Path) -> str:
    """Produce a starter AGENTS.md based on repository analysis."""
    languages = _detect_languages(repo)
    primary = languages[0] if languages else "unknown"
    pkg_manager = _detect_package_manager(repo)
    src_dir = _detect_src_layout(repo)
    ci_files = _detect_ci(repo)
    test_cmd = _TEST_COMMANDS.get(primary, "# TODO: add test command")
    lint_cmd = _LINT_COMMANDS.get(primary, "")

    lines: list[str] = [
        f"# {repo.name}",
        "",
    ]

    # Overview
    lines.append("## Overview")
    lines.append("")
    if languages:
        lines.append(f"Primary language: **{primary}**")
        if len(languages) > 1:
            lines.append(f"Also detected: {', '.join(languages[1:])}")
    if pkg_manager:
        lines.append(f"Package manager: `{pkg_manager}`")
    if src_dir:
        lines.append(f"Source layout: `{src_dir}/`")
    lines.append("")

    # Commands
    lines.append("## Commands")
    lines.append("")
    lines.append(f"- Test: `{test_cmd}`")
    if lint_cmd:
        lines.append(f"- Lint: `{lint_cmd}`")
    if pkg_manager == "uv":
        lines.append("- Install: `uv sync`")
        lines.append("- Run: `uv run <entrypoint>`")
    elif pkg_manager in ("npm", "pnpm", "yarn"):
        lines.append(f"- Install: `{pkg_manager} install`")
        lines.append(f"- Build: `{pkg_manager} run build`")
    lines.append("")

    # Architecture hints
    if src_dir:
        src_path = repo / src_dir
        subdirs = sorted(d.name for d in src_path.iterdir() if d.is_dir() and not d.name.startswith("."))
        if subdirs:
            lines.append("## Architecture")
            lines.append("")
            lines.append(f"Key modules under `{src_dir}/`:")
            for name in subdirs[:15]:
                lines.append(f"- `{src_dir}/{name}/`")
            lines.append("")

    # CI
    if ci_files:
        lines.append("## CI")
        lines.append("")
        for ci in ci_files:
            lines.append(f"- `{ci}`")
        lines.append("")

    # Conventions (placeholder for user to fill)
    lines.append("## Conventions")
    lines.append("")
    lines.append("- Keep changes minimal and focused.")
    lines.append("- Add tests for new behaviour.")
    lines.append("- Never commit secrets or credentials.")
    lines.append("")

    return "\n".join(lines)


def init_project(repo: Path, *, force: bool = False) -> str:
    """Generate AGENTS.md for a repository. Returns the generated content.

    Raises FileExistsError if AGENTS.md already exists and force is False.
    """
    target = repo / "AGENTS.md"
    if target.exists() and not force:
        raise FileExistsError(f"AGENTS.md already exists at {target}")
    content = generate_agents_md(repo)
    target.write_text(content, encoding="utf-8")
    return content


__all__ = ["generate_agents_md", "init_project"]
