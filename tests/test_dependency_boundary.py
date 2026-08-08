from __future__ import annotations

import re
import tomllib
from pathlib import Path

MIN_NOOA_VERSION = "0.0.8"


def test_all_nooa_packages_have_minimum_version_from_pypi() -> None:
    """All NOOA packages must specify a minimum version >= MIN_NOOA_VERSION."""
    root = Path(__file__).parents[1]
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = config["project"]["dependencies"]

    nooa_deps = [d for d in dependencies if d.startswith(("nooa", "nooa-cli", "nooa-memory"))]
    assert len(nooa_deps) == 3, f"Expected 3 nooa deps, got: {nooa_deps}"
    for dep in nooa_deps:
        assert f">={MIN_NOOA_VERSION}" in dep, f"{dep} missing >={MIN_NOOA_VERSION}"

    # Verify no git sources are configured (PyPI only).
    sources = config.get("tool", {}).get("uv", {}).get("sources", {})
    assert not sources, f"Expected no [tool.uv.sources], found: {list(sources)}"


def test_application_has_no_nooa_bench_dependency() -> None:
    root = Path(__file__).parents[1]
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in (root / "src" / "nooa_coding").glob("*.py")
    )
    assert "nooa_bench" not in source


def test_mcp_client_extra_is_enabled_with_compatible_protocol_version() -> None:
    root = Path(__file__).parents[1]
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = config["project"]["dependencies"]

    assert any(d.startswith("nooa[mcp]") for d in dependencies)
    assert "mcp>=1,<2" in dependencies


def test_semantic_release_automates_versioning_from_commits() -> None:
    """Version bumps stay automated: PSR config targets pyproject and stays local."""
    root = Path(__file__).parents[1]
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))

    assert re.fullmatch(r"\d+\.\d+\.\d+", config["project"]["version"])

    dev = config["dependency-groups"]["dev"]
    assert any(dep.startswith("python-semantic-release") for dep in dev)

    sr = config["tool"]["semantic_release"]
    assert sr["version_toml"] == ["pyproject.toml:project.version"]
    assert sr["allow_zero_version"] is True
    assert sr["major_on_zero"] is False
    assert sr["tag_format"] == "v{version}"
    # Publishing remains manual; releases only produce a version bump + tag.
    assert sr["publish"]["upload_to_vcs_release"] is False
