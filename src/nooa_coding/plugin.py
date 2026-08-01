"""Third-party plugin registration API with lifecycle hooks.

Capabilities absorbed from NOOA LibraryManager:
- Hot-reload with sys.modules cache-busting
- Skill auto-discovery (if a plugin module exports a Skill, it is attached)
- CommandRegistry integration (slash commands refresh after install/reload)
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from importlib.metadata import entry_points
from pathlib import Path
from types import ModuleType
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
    __slots__ = ("name", "factory", "instance", "source", "module", "path")

    def __init__(
        self,
        name: str,
        factory: Callable[[], Plugin],
        source: str,
        *,
        path: Path | None = None,
    ) -> None:
        self.name = name
        self.factory = factory
        self.instance: Plugin | None = None
        self.source = source
        self.module: ModuleType | None = None
        self.path = path


class PluginRegistry:
    """Discover, load, and manage plugin lifecycle.

    Supports hot-reload, Skill auto-discovery, and CommandRegistry refresh.
    """

    def __init__(self) -> None:
        self._entries: dict[str, _PluginEntry] = {}
        self._installed: list[Plugin] = []
        self._agent: Any = None
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
                path=file,
            )
            count += 1
        return count

    @staticmethod
    def _make_file_loader(path: Path) -> Callable[[], Plugin]:
        def loader() -> Plugin:
            module = PluginRegistry._import_with_cache_bust(path)
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

    @staticmethod
    def _import_with_cache_bust(path: Path) -> ModuleType:
        """Import a module from file path, busting all caches for fresh code.

        Uses direct source exec to bypass .pyc timestamp granularity issues.
        """
        module_name = f"_nooa_plugin_{path.stem}"
        # Remove cached module and sub-modules.
        to_remove = [k for k in sys.modules if k == module_name or k.startswith(module_name + ".")]
        for key in to_remove:
            del sys.modules[key]
        # Read source directly — no bytecode cache involved.
        source = path.read_text(encoding="utf-8")
        module = ModuleType(module_name)
        module.__file__ = str(path)
        module.__loader__ = None
        sys.modules[module_name] = module
        code = compile(source, str(path), "exec")
        exec(code, module.__dict__)  # noqa: S102
        return module

    def discovered(self) -> list[str]:
        """All discovered plugin names."""
        return sorted(self._entries)

    def install_all(self, agent: Any) -> list[str]:
        """Instantiate and install all discovered plugins. Returns installed names."""
        self._agent = agent
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
                # Skill auto-discovery: attach exported Skills to agent.
                self._try_attach_skills(entry, agent)
            except Exception as exc:
                logger.warning("Plugin %s failed to install: %s", name, exc)
        # Refresh slash commands if a CommandRegistry is available.
        self._refresh_commands(agent)
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

    # ─── Hot-Reload ──────────────────────────────────────────────────────

    def reload(self, name: str | None = None) -> list[str]:
        """Hot-reload plugins from disk. Reloads all if name is None.

        Cache-busts sys.modules, re-imports, calls on_session_close on the
        old instance, then on_install on the fresh one.

        Returns list of successfully reloaded plugin names.
        """
        if self._agent is None:
            raise RuntimeError("Cannot reload before install_all() is called.")

        targets = [name] if name else sorted(self._entries)
        reloaded: list[str] = []

        for plugin_name in targets:
            entry = self._entries.get(plugin_name)
            if entry is None or entry.path is None:
                continue  # entry_point plugins or unknown — skip.
            try:
                # Close old instance.
                if entry.instance is not None:
                    try:
                        entry.instance.on_session_close()
                    except Exception:
                        pass
                    self._installed = [
                        p for p in self._installed if p is not entry.instance
                    ]

                # Re-import with cache bust.
                module = self._import_with_cache_bust(entry.path)
                entry.module = module

                # Find Plugin subclass in fresh module.
                plugin_cls: type[Plugin] | None = None
                for attr_name in dir(module):
                    obj = getattr(module, attr_name)
                    if (
                        isinstance(obj, type)
                        and issubclass(obj, Plugin)
                        and obj is not Plugin
                    ):
                        plugin_cls = obj
                        break
                if plugin_cls is None:
                    logger.warning("Reload %s: no Plugin subclass found", plugin_name)
                    continue

                plugin = plugin_cls()
                plugin.on_install(self._agent)
                entry.instance = plugin
                entry.factory = self._make_file_loader(entry.path)
                self._installed.append(plugin)
                reloaded.append(plugin_name)

                # Skill auto-discovery on fresh module.
                self._try_attach_skills(entry, self._agent)
            except Exception as exc:
                logger.warning("Reload %s failed: %s", plugin_name, exc)

        if reloaded:
            self._refresh_commands(self._agent)
        return reloaded

    # ─── Skill Auto-Discovery ────────────────────────────────────────────

    @staticmethod
    def _try_attach_skills(entry: _PluginEntry, agent: Any) -> None:
        """If the plugin module exports Skill subclasses, attach them to agent."""
        if entry.path is None:
            return
        try:
            module = entry.module
            if module is None:
                module = PluginRegistry._import_with_cache_bust(entry.path)
                entry.module = module

            from nooa import Skill

            for attr_name in dir(module):
                obj = getattr(module, attr_name)
                if (
                    isinstance(obj, type)
                    and issubclass(obj, Skill)
                    and obj is not Skill
                ):
                    skill_name = attr_name.lower()
                    if not hasattr(agent, skill_name):
                        instance = obj()
                        setattr(agent, skill_name, instance)
                        if hasattr(instance, "attach"):
                            instance.attach(agent)
                        logger.info(
                            "Plugin %s: auto-attached Skill %s as agent.%s",
                            entry.name, attr_name, skill_name,
                        )
        except Exception as exc:
            logger.debug("Skill auto-discovery for %s: %s", entry.name, exc)

    # ─── CommandRegistry Integration ─────────────────────────────────────

    @staticmethod
    def _refresh_commands(agent: Any) -> None:
        """Refresh slash commands if the agent has a CommandRegistry."""
        registry = getattr(agent, "_command_registry", None)
        if registry is not None and hasattr(registry, "refresh_skill_commands"):
            try:
                registry.refresh_skill_commands()
            except Exception:
                pass


__all__ = ["Plugin", "PluginRegistry"]
