"""Tests for the plugin system: lifecycle, hot-reload, Skill auto-discovery."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from nooa_coding.plugin import PluginRegistry

# ─── Helpers ─────────────────────────────────────────────────────────────────


def write_plugin(directory: Path, name: str, content: str) -> Path:
    """Write a plugin .py file and return its path."""
    path = directory / f"{name}.py"
    path.write_text(content, encoding="utf-8")
    return path


BASIC_PLUGIN = '''\
from nooa_coding.plugin import Plugin

class HelloPlugin(Plugin):
    name = "hello"
    version = "1.0.0"
    installed = False
    turns = []

    def on_install(self, agent):
        HelloPlugin.installed = True
        agent.hello_marker = True

    def on_turn_start(self, task):
        HelloPlugin.turns.append(task)

    def on_session_close(self):
        HelloPlugin.installed = False
'''

RELOADABLE_PLUGIN_V1 = '''\
from nooa_coding.plugin import Plugin

class VersionPlugin(Plugin):
    name = "versioned"
    version = "1.0.0"

    def on_install(self, agent):
        agent.plugin_version = "1.0.0"
'''

RELOADABLE_PLUGIN_V2 = '''\
from nooa_coding.plugin import Plugin

class VersionPlugin(Plugin):
    name = "versioned"
    version = "2.0.0"

    def on_install(self, agent):
        agent.plugin_version = "2.0.0"
'''

SKILL_PLUGIN = '''\
from nooa import Skill
from nooa_coding.plugin import Plugin

class MyTool(Skill):
    """A test skill."""
    def greet(self) -> str:
        return "hi from skill"

class SkillPlugin(Plugin):
    name = "skill_plugin"

    def on_install(self, agent):
        agent.skill_plugin_active = True
'''


# ─── Basic Lifecycle ─────────────────────────────────────────────────────────


class TestPluginLifecycle:
    def test_discover_directory(self, tmp_path: Path):
        write_plugin(tmp_path, "myplug", BASIC_PLUGIN)
        registry = PluginRegistry()
        count = registry.discover_directory(tmp_path)
        assert count == 1
        assert "myplug" in registry.discovered()

    def test_install_all(self, tmp_path: Path):
        write_plugin(tmp_path, "hello", BASIC_PLUGIN)
        registry = PluginRegistry()
        registry.discover_directory(tmp_path)
        agent = SimpleNamespace()
        installed = registry.install_all(agent)
        assert "hello" in installed
        assert agent.hello_marker is True

    def test_turn_notifications(self, tmp_path: Path):
        write_plugin(tmp_path, "hello", BASIC_PLUGIN)
        registry = PluginRegistry()
        registry.discover_directory(tmp_path)
        agent = SimpleNamespace()
        registry.install_all(agent)
        registry.notify_turn_start("fix bug")
        # Verify via status that plugin is installed.
        statuses = registry.status()
        assert any(s["name"] == "hello" and s["status"] == "installed" for s in statuses)

    def test_skips_underscore_files(self, tmp_path: Path):
        write_plugin(tmp_path, "_private", BASIC_PLUGIN)
        registry = PluginRegistry()
        count = registry.discover_directory(tmp_path)
        assert count == 0

    def test_error_isolation(self, tmp_path: Path):
        """A failing plugin should not prevent others from installing."""
        write_plugin(tmp_path, "bad", "raise RuntimeError('boom')\n")
        write_plugin(tmp_path, "good", BASIC_PLUGIN)
        registry = PluginRegistry()
        registry.discover_directory(tmp_path)
        agent = SimpleNamespace()
        installed = registry.install_all(agent)
        assert "good" in installed
        assert "bad" not in installed


# ─── Hot-Reload ──────────────────────────────────────────────────────────────


class TestHotReload:
    def test_reload_picks_up_new_code(self, tmp_path: Path):
        plugin_path = write_plugin(tmp_path, "versioned", RELOADABLE_PLUGIN_V1)
        registry = PluginRegistry()
        registry.discover_directory(tmp_path)
        agent = SimpleNamespace()
        registry.install_all(agent)
        assert agent.plugin_version == "1.0.0"

        # Modify the plugin on disk.
        plugin_path.write_text(RELOADABLE_PLUGIN_V2, encoding="utf-8")

        # Reload.
        reloaded = registry.reload("versioned")
        assert "versioned" in reloaded
        assert agent.plugin_version == "2.0.0"

    def test_reload_all(self, tmp_path: Path):
        write_plugin(tmp_path, "versioned", RELOADABLE_PLUGIN_V1)
        registry = PluginRegistry()
        registry.discover_directory(tmp_path)
        agent = SimpleNamespace()
        registry.install_all(agent)

        reloaded = registry.reload()
        assert "versioned" in reloaded

    def test_reload_before_install_raises(self, tmp_path: Path):
        write_plugin(tmp_path, "x", BASIC_PLUGIN)
        registry = PluginRegistry()
        registry.discover_directory(tmp_path)
        with pytest.raises(RuntimeError, match="Cannot reload"):
            registry.reload()

    def test_reload_nonexistent_skips(self, tmp_path: Path):
        write_plugin(tmp_path, "hello", BASIC_PLUGIN)
        registry = PluginRegistry()
        registry.discover_directory(tmp_path)
        agent = SimpleNamespace()
        registry.install_all(agent)
        reloaded = registry.reload("nonexistent")
        assert reloaded == []


# ─── Skill Auto-Discovery ────────────────────────────────────────────────────


class TestSkillAutoDiscovery:
    def test_skill_attached_to_agent(self, tmp_path: Path):
        write_plugin(tmp_path, "skillplug", SKILL_PLUGIN)
        registry = PluginRegistry()
        registry.discover_directory(tmp_path)
        agent = SimpleNamespace()
        registry.install_all(agent)

        # Plugin lifecycle should work.
        assert agent.skill_plugin_active is True
        # Skill should be auto-attached.
        assert hasattr(agent, "mytool")
        assert agent.mytool.greet() == "hi from skill"

    def test_no_skill_no_attach(self, tmp_path: Path):
        write_plugin(tmp_path, "hello", BASIC_PLUGIN)
        registry = PluginRegistry()
        registry.discover_directory(tmp_path)
        agent = SimpleNamespace()
        registry.install_all(agent)
        # No Skill in BASIC_PLUGIN, so nothing extra attached.
        assert agent.hello_marker is True


# ─── CommandRegistry Integration ─────────────────────────────────────────────


class TestCommandRegistryIntegration:
    def test_refresh_called_on_install(self, tmp_path: Path):
        write_plugin(tmp_path, "hello", BASIC_PLUGIN)
        registry = PluginRegistry()
        registry.discover_directory(tmp_path)

        refreshed = []

        class FakeRegistry:
            def refresh_skill_commands(self):
                refreshed.append(True)

        agent = SimpleNamespace(_command_registry=FakeRegistry())
        registry.install_all(agent)
        assert len(refreshed) == 1

    def test_refresh_called_on_reload(self, tmp_path: Path):
        write_plugin(tmp_path, "versioned", RELOADABLE_PLUGIN_V1)
        registry = PluginRegistry()
        registry.discover_directory(tmp_path)

        refreshed = []

        class FakeRegistry:
            def refresh_skill_commands(self):
                refreshed.append(True)

        agent = SimpleNamespace(_command_registry=FakeRegistry())
        registry.install_all(agent)
        assert len(refreshed) == 1

        registry.reload("versioned")
        assert len(refreshed) == 2

    def test_no_registry_no_error(self, tmp_path: Path):
        write_plugin(tmp_path, "hello", BASIC_PLUGIN)
        registry = PluginRegistry()
        registry.discover_directory(tmp_path)
        agent = SimpleNamespace()  # No _command_registry.
        installed = registry.install_all(agent)
        assert "hello" in installed
