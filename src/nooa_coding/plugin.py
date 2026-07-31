"""Third-party plugin registration API with lifecycle hooks."""

from __future__ import annotations

import logging
from collections.abc import Callable
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "nooa_coding.plugins"


class Plugin:
    """Base class for nooa-coding plugins.

    Subclass this and implement hooks to extend the coding agent.
    Plugins are discovered via the ``nooa_coding.plugins`` entry point group
    or by placing a Python file in a configured plugin directory.

    Lifecycle:
        on_install(agent)     — called once when the plugin is attached to the agent.
        on_turn_start(task)   — called before each coding turn.
        on_turn_end(result)   — called after each coding turn with the result.
        on_session_close()    — called when the session is closing.

    To register tools, add Skill instances in ``on_install``::

        class MyPlugin(Plugin):
            def on_install(self, agent):
                agent.my_tool = MyTool()
                spec(agent, "my_tool", hidden=False)
    """

    name: str = "unnamed"
    version: str = "0.0.0"

    def on_install(self, agent: Any) -> None:
        """Called once when the plugin is attached to the agent."""

    def on_turn_start(self, task: str) -> None:
        """Called before each coding turn begins."""

    def on_turn_end(self, result: Any) -> None:
        """Called after each coding turn completes."""

    def on_session_close(self) -> None:
        """Called when the session is shutting down."""


class _PluginEntry:
    __slots__ = ("name", "factory", "instance", "source")

    def __init__(self, name: str, factory: Callable[[], Plugin], source: str) -> None:
        self.name = name
        self.factory = factory
        self.instance: Plugin | None = None
        self.source = source


class PluginRegistry:
    """Discover, load, and manage plugin lifecycle."""

    def __init__(self) -> None:
        self._entries: dict[str, _PluginEntry] = {}
        self._installed: list[Plugin] = []
        self._discover_entry_points()

    def _discover_entry_points(self) -> None:
        try:
            eps = entry_points(group=ENTRY_POINT_GROUP)
        except Exception:
            eps = []
        for ep in eps:
            self._entries[ep.name] = _PluginEntry(
                name=ep.name,
                factory=ep.load,
                source=f"entry_point:{ep.name}",
            )

    def discover_directory(self, directory: str | Path) -> int:
        """Scan a directory for .py plugin files. Returns count discovered."""
        path = Path(directory).expanduser()
        if not path.is_dir():
            return 0
        count = 0
        for file in sorted(path.glob("*.py")):
            if file.name.startswith("_"):
                continue
            name = file.stem
            if name in self._entries:
                continue
            self._entries[name] = _PluginEntry(
                name=name,
                factory=self._make_file_loader(file),
                source=str(file),
            )
            count += 1
        return count

    @staticmethod
    def _make_file_loader(path: Path) -> Callable[[], Plugin]:
        def loader() -> Plugin:
            import importlib.util

            module_spec = importlib.util.spec_from_file_location(path.stem, path)
            if module_spec is None or module_spec.loader is None:
                raise ImportError(f"cannot load plugin from {path}")
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            # Find the first Plugin subclass in the module.
            for attr_name in dir(module):
                obj = getattr(module, attr_name)
                if (
                    isinstance(obj, type)
                    and issubclass(obj, Plugin)
                    and obj is not Plugin
                ):
                    return obj()
            raise TypeError(f"no Plugin subclass found in {path}")

        return loader

    def discovered(self) -> list[str]:
        """All discovered plugin names."""
        return sorted(self._entries)

    def install_all(self, agent: Any) -> list[str]:
        """Instantiate and install all discovered plugins. Returns installed names."""
        installed: list[str] = []
        for name, entry in sorted(self._entries.items()):
            try:
                plugin = entry.factory()
                if not isinstance(plugin, Plugin):
                    logger.warning("Plugin %s did not return a Plugin instance", name)
                    continue
                plugin.on_install(agent)
                entry.instance = plugin
                self._installed.append(plugin)
                installed.append(name)
            except Exception as exc:
                logger.warning("Plugin %s failed to install: %s", name, exc)
        return installed

    def notify_turn_start(self, task: str) -> None:
        for plugin in self._installed:
            try:
                plugin.on_turn_start(task)
            except Exception as exc:
                logger.warning("Plugin %s on_turn_start failed: %s", plugin.name, exc)

    def notify_turn_end(self, result: Any) -> None:
        for plugin in self._installed:
            try:
                plugin.on_turn_end(result)
            except Exception as exc:
                logger.warning("Plugin %s on_turn_end failed: %s", plugin.name, exc)

    def notify_close(self) -> None:
        for plugin in self._installed:
            try:
                plugin.on_session_close()
            except Exception as exc:
                logger.warning("Plugin %s on_session_close failed: %s", plugin.name, exc)

    def status(self) -> list[dict[str, str]]:
        """Return status info for all discovered plugins."""
        result: list[dict[str, str]] = []
        for name, entry in sorted(self._entries.items()):
            result.append({
                "name": name,
                "source": entry.source,
                "status": "installed" if entry.instance else "discovered",
            })
        return result


__all__ = ["Plugin", "PluginRegistry"]
