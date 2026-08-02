"""Tests for SkillManifest: registration, triggers, routes, and output contracts."""

from __future__ import annotations

from pathlib import Path

from nooa_coding.resources import build_skill_manifest

_BUILTIN_SKILLS_DIR = Path(__file__).parent.parent / "src" / "nooa_coding" / "skills"

EXPECTED_SKILLS = {"create-pr", "explain-error", "refactor", "run-tests", "use-tools"}


class TestSkillManifestDiscovery:
    """Verify that all 5 builtin skills are discovered and configured."""

    def test_discovers_all_five_builtin_skills(self) -> None:
        manifest = build_skill_manifest([_BUILTIN_SKILLS_DIR])
        names = {s.name for s in manifest.configured_skills()}
        assert names == EXPECTED_SKILLS

    def test_configured_skills_count(self) -> None:
        manifest = build_skill_manifest([_BUILTIN_SKILLS_DIR])
        assert len(manifest) == 5

    def test_contains_check(self) -> None:
        manifest = build_skill_manifest([_BUILTIN_SKILLS_DIR])
        for name in EXPECTED_SKILLS:
            assert name in manifest

    def test_missing_directory_yields_empty(self, tmp_path: Path) -> None:
        manifest = build_skill_manifest([tmp_path / "nonexistent"])
        assert len(manifest) == 0


class TestSkillDescriptorFields:
    """Each skill must have non-empty trigger, route, and output_contract."""

    def test_all_skills_have_trigger(self) -> None:
        manifest = build_skill_manifest([_BUILTIN_SKILLS_DIR])
        for skill in manifest.configured_skills():
            assert skill.trigger, f"{skill.name} missing trigger"

    def test_all_skills_have_route(self) -> None:
        manifest = build_skill_manifest([_BUILTIN_SKILLS_DIR])
        for skill in manifest.configured_skills():
            assert skill.route, f"{skill.name} missing route"

    def test_all_skills_have_output_contract(self) -> None:
        manifest = build_skill_manifest([_BUILTIN_SKILLS_DIR])
        for skill in manifest.configured_skills():
            assert skill.output_contract, f"{skill.name} missing output-contract"

    def test_all_skills_have_description(self) -> None:
        manifest = build_skill_manifest([_BUILTIN_SKILLS_DIR])
        for skill in manifest.configured_skills():
            assert skill.description, f"{skill.name} missing description"

    def test_routes_are_unique(self) -> None:
        manifest = build_skill_manifest([_BUILTIN_SKILLS_DIR])
        routes = [s.route for s in manifest.configured_skills()]
        assert len(routes) == len(set(routes))

    def test_route_matches_skill_name(self) -> None:
        manifest = build_skill_manifest([_BUILTIN_SKILLS_DIR])
        for skill in manifest.configured_skills():
            assert skill.route == skill.name

    def test_to_dict_roundtrip(self) -> None:
        manifest = build_skill_manifest([_BUILTIN_SKILLS_DIR])
        skill = manifest.get("create-pr")
        assert skill is not None
        d = skill.to_dict()
        assert d["name"] == "create-pr"
        assert d["route"] == "create-pr"
        assert "trigger" in d
        assert "output_contract" in d


class TestSkillRoutes:
    """Verify route mapping is correct."""

    def test_routes_mapping(self) -> None:
        manifest = build_skill_manifest([_BUILTIN_SKILLS_DIR])
        routes = manifest.routes()
        assert routes == {
            "create-pr": "create-pr",
            "explain-error": "explain-error",
            "refactor": "refactor",
            "run-tests": "run-tests",
            "use-tools": "use-tools",
        }

    def test_get_by_name(self) -> None:
        manifest = build_skill_manifest([_BUILTIN_SKILLS_DIR])
        skill = manifest.get("refactor")
        assert skill is not None
        assert skill.name == "refactor"
        assert "behaviour" in skill.description.lower() or "behavior" in skill.description.lower()

    def test_get_unknown_returns_none(self) -> None:
        manifest = build_skill_manifest([_BUILTIN_SKILLS_DIR])
        assert manifest.get("nonexistent") is None


class TestTriggerMatching:
    """Verify trigger keyword matching against user input."""

    def test_create_pr_trigger(self) -> None:
        manifest = build_skill_manifest([_BUILTIN_SKILLS_DIR])
        matched = manifest.match_trigger("please create PR for this branch")
        names = {s.name for s in matched}
        assert "create-pr" in names

    def test_explain_error_trigger(self) -> None:
        manifest = build_skill_manifest([_BUILTIN_SKILLS_DIR])
        matched = manifest.match_trigger("explain error in the login module")
        names = {s.name for s in matched}
        assert "explain-error" in names

    def test_run_tests_trigger(self) -> None:
        manifest = build_skill_manifest([_BUILTIN_SKILLS_DIR])
        matched = manifest.match_trigger("run tests and show results")
        names = {s.name for s in matched}
        assert "run-tests" in names

    def test_refactor_trigger(self) -> None:
        manifest = build_skill_manifest([_BUILTIN_SKILLS_DIR])
        matched = manifest.match_trigger("refactor the auth module")
        names = {s.name for s in matched}
        assert "refactor" in names

    def test_no_match_for_unrelated_text(self) -> None:
        manifest = build_skill_manifest([_BUILTIN_SKILLS_DIR])
        matched = manifest.match_trigger("hello world")
        assert len(matched) == 0


class TestTriggerWorkflowAlignment:
    """Confirm trigger descriptions align with project workflows."""

    def test_create_pr_workflow(self) -> None:
        manifest = build_skill_manifest([_BUILTIN_SKILLS_DIR])
        skill = manifest.get("create-pr")
        assert skill is not None
        # Must reference PR creation workflow
        assert "pr" in skill.trigger.lower() or "pull request" in skill.trigger.lower()

    def test_explain_error_workflow(self) -> None:
        manifest = build_skill_manifest([_BUILTIN_SKILLS_DIR])
        skill = manifest.get("explain-error")
        assert skill is not None
        # Must reference error diagnosis workflow
        assert "error" in skill.trigger.lower() or "traceback" in skill.trigger.lower()

    def test_run_tests_workflow(self) -> None:
        manifest = build_skill_manifest([_BUILTIN_SKILLS_DIR])
        skill = manifest.get("run-tests")
        assert skill is not None
        # Must reference test execution workflow
        assert "test" in skill.trigger.lower()

    def test_refactor_workflow(self) -> None:
        manifest = build_skill_manifest([_BUILTIN_SKILLS_DIR])
        skill = manifest.get("refactor")
        assert skill is not None
        # Must reference code reorganization workflow
        assert "refactor" in skill.trigger.lower()

    def test_use_tools_workflow(self) -> None:
        manifest = build_skill_manifest([_BUILTIN_SKILLS_DIR])
        skill = manifest.get("use-tools")
        assert skill is not None
        # Must reference tool discovery workflow
        assert "tool" in skill.trigger.lower() or "self." in skill.trigger


class TestCustomSkillDirectory:
    """Verify discovery works with custom skill directories."""

    def test_custom_skill_discovered(self, tmp_path: Path) -> None:
        custom = tmp_path / "my-skill"
        custom.mkdir()
        (custom / "SKILL.md").write_text(
            "---\n"
            "name: my-skill\n"
            "description: A custom test skill\n"
            'trigger: User says "my-skill" or "custom action"\n'
            "route: my-skill\n"
            "output-contract: Returns custom output\n"
            "---\n\n"
            "# My Skill\n\nDo things.\n",
            encoding="utf-8",
        )
        manifest = build_skill_manifest([tmp_path])
        assert "my-skill" in manifest
        skill = manifest.get("my-skill")
        assert skill is not None
        assert skill.route == "my-skill"
        assert skill.output_contract == "Returns custom output"

    def test_skill_without_enhanced_fields(self, tmp_path: Path) -> None:
        """Skills with only name/description still get discovered with empty fields."""
        basic = tmp_path / "basic-skill"
        basic.mkdir()
        (basic / "SKILL.md").write_text(
            "---\nname: basic-skill\ndescription: Basic\n---\n\n# Basic\n",
            encoding="utf-8",
        )
        manifest = build_skill_manifest([tmp_path])
        assert "basic-skill" in manifest
        skill = manifest.get("basic-skill")
        assert skill is not None
        assert skill.trigger == ""
        assert skill.route == "basic-skill"  # defaults to name
