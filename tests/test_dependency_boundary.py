from __future__ import annotations

import tomllib
from pathlib import Path

PINNED_NOOA_REV = "f22805b52ea8a073dabc018cefe3db1ccf609a29"


def test_all_nooa_packages_are_pinned_to_one_reviewed_commit() -> None:
    root = Path(__file__).parents[1]
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    sources = config["tool"]["uv"]["sources"]

    assert {sources[name]["rev"] for name in ("nooa", "nooa-cli", "nooa-memory")} == {
        PINNED_NOOA_REV
    }
    lock = (root / "uv.lock").read_text(encoding="utf-8")
    assert lock.count(f"#{PINNED_NOOA_REV}") >= 3


def test_application_has_no_nooa_bench_dependency() -> None:
    root = Path(__file__).parents[1]
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in (root / "src" / "nooa_coding").glob("*.py")
    )
    assert "nooa_bench" not in source
