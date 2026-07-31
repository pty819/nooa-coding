"""Tests for the Claude Code compatible hooks system."""

from __future__ import annotations

import json

import pytest

from nooa_coding.hooks import (
    HookAction,
    HookEvent,
    HookRunner,
    HooksConfig,
    load_hooks_config,
)

# ─── Configuration Parsing ───────────────────────────────────────────────────


class TestHooksConfig:
    def test_empty_config(self):
        config = HooksConfig()
        assert config.rules_for(HookEvent.PRE_TOOL_USE) == []
        assert config.rules_for(HookEvent.STOP) == []

    def test_parse_config(self):
        config = HooksConfig(hooks={
            "PreToolUse": [
                {"matcher": "Bash", "hooks": [{"type": "command", "command": "echo hi"}]}
            ],
            "PostToolUse": [
                {"matcher": "Edit|Write", "hooks": [{"type": "command", "command": "lint"}]}
            ],
        })
        pre_rules = config.rules_for(HookEvent.PRE_TOOL_USE)
        assert len(pre_rules) == 1
        assert pre_rules[0].matcher == "Bash"
        assert pre_rules[0].hooks[0].command == "echo hi"

        post_rules = config.rules_for(HookEvent.POST_TOOL_USE)
        assert len(post_rules) == 1
        assert post_rules[0].matcher == "Edit|Write"

    def test_hook_action_alias(self):
        action = HookAction(type="command", command="test", statusMessage="Running...")
        assert action.status_message == "Running..."


# ─── Matcher Logic ───────────────────────────────────────────────────────────


class TestMatcher:
    def test_wildcard(self):
        assert HookRunner._matches("*", "Bash") is True
        assert HookRunner._matches("", "anything") is True

    def test_exact(self):
        assert HookRunner._matches("Bash", "Bash") is True
        assert HookRunner._matches("Bash", "Edit") is False

    def test_pipe_separated(self):
        assert HookRunner._matches("Bash|Edit|Write", "Edit") is True
        assert HookRunner._matches("Bash|Edit|Write", "Read") is False

    def test_glob(self):
        assert HookRunner._matches("B*", "Bash") is True
        assert HookRunner._matches("B*", "Edit") is False


# ─── PreToolUse Blocking ─────────────────────────────────────────────────────


class TestPreToolUse:
    @pytest.fixture()
    def blocking_config(self):
        return HooksConfig(hooks={
            "PreToolUse": [
                {
                    "matcher": "Bash",
                    "hooks": [
                        {"type": "command", "command": "exit 1", "timeout": 5}
                    ],
                }
            ]
        })

    @pytest.fixture()
    def allowing_config(self):
        return HooksConfig(hooks={
            "PreToolUse": [
                {
                    "matcher": "Bash",
                    "hooks": [
                        {"type": "command", "command": "exit 0", "timeout": 5}
                    ],
                }
            ]
        })

    async def test_blocks_on_nonzero_exit(self, blocking_config):
        runner = HookRunner(blocking_config, workspace="/tmp")
        result = await runner.trigger_pre_tool_use("Bash", tool_input="rm -rf /")
        assert result.allowed is False
        assert result.block_reason != ""

    async def test_allows_on_zero_exit(self, allowing_config):
        runner = HookRunner(allowing_config, workspace="/tmp")
        result = await runner.trigger_pre_tool_use("Bash", tool_input="ls")
        assert result.allowed is True

    async def test_no_matching_rules_allows(self):
        config = HooksConfig(hooks={
            "PreToolUse": [
                {"matcher": "Edit", "hooks": [{"type": "command", "command": "exit 1"}]}
            ]
        })
        runner = HookRunner(config, workspace="/tmp")
        result = await runner.trigger_pre_tool_use("Bash", tool_input="ls")
        assert result.allowed is True

    async def test_stderr_as_block_reason(self):
        config = HooksConfig(hooks={
            "PreToolUse": [
                {
                    "matcher": "Bash",
                    "hooks": [
                        {"type": "command", "command": "echo 'dangerous!' >&2; exit 1"}
                    ],
                }
            ]
        })
        runner = HookRunner(config, workspace="/tmp")
        result = await runner.trigger_pre_tool_use("Bash", tool_input="rm -rf /")
        assert result.allowed is False
        assert "dangerous!" in result.block_reason


# ─── PostToolUse ─────────────────────────────────────────────────────────────


class TestPostToolUse:
    async def test_runs_matching_hooks(self, tmp_path):
        marker = tmp_path / "ran.txt"
        config = HooksConfig(hooks={
            "PostToolUse": [
                {
                    "matcher": "Bash",
                    "hooks": [
                        {"type": "command", "command": f"touch {marker}"}
                    ],
                }
            ]
        })
        runner = HookRunner(config, workspace=str(tmp_path))
        results = await runner.trigger_post_tool_use("Bash", tool_input="ls")
        assert len(results) == 1
        assert results[0].returncode == 0
        assert marker.exists()

    async def test_non_matching_skips(self, tmp_path):
        config = HooksConfig(hooks={
            "PostToolUse": [
                {"matcher": "Edit", "hooks": [{"type": "command", "command": "exit 1"}]}
            ]
        })
        runner = HookRunner(config, workspace=str(tmp_path))
        results = await runner.trigger_post_tool_use("Bash", tool_input="ls")
        assert results == []


# ─── Environment Variables ───────────────────────────────────────────────────


class TestEnvironment:
    async def test_env_injection(self, tmp_path):
        output = tmp_path / "env.txt"
        config = HooksConfig(hooks={
            "PreToolUse": [
                {
                    "matcher": "Bash",
                    "hooks": [
                        {
                            "type": "command",
                            "command": f'echo "$TOOL_NAME:$TOOL_INPUT" > {output}',
                        }
                    ],
                }
            ]
        })
        runner = HookRunner(config, workspace=str(tmp_path), session_id="test-123")
        await runner.trigger_pre_tool_use("Bash", tool_input="hello world")
        content = output.read_text()
        assert "Bash:hello world" in content


# ─── Timeout Handling ────────────────────────────────────────────────────────


class TestTimeout:
    async def test_timeout_returns_124(self):
        config = HooksConfig(hooks={
            "PreToolUse": [
                {
                    "matcher": "*",
                    "hooks": [
                        {"type": "command", "command": "sleep 10", "timeout": 1}
                    ],
                }
            ]
        })
        runner = HookRunner(config, workspace="/tmp")
        result = await runner.trigger_pre_tool_use("anything")
        assert result.allowed is False  # Timeout = non-zero = block
        assert result.hook_results[0].timed_out is True


# ─── Stop / Notification / SessionStart ─────────────────────────────────────


class TestLifecycleHooks:
    async def test_stop_hook(self, tmp_path):
        marker = tmp_path / "stopped.txt"
        config = HooksConfig(hooks={
            "Stop": [
                {"matcher": "*", "hooks": [{"type": "command", "command": f"touch {marker}"}]}
            ]
        })
        runner = HookRunner(config, workspace=str(tmp_path))
        results = await runner.trigger_stop(summary="done")
        assert len(results) == 1
        assert marker.exists()

    async def test_notification_hook(self, tmp_path):
        output = tmp_path / "notify.txt"
        config = HooksConfig(hooks={
            "Notification": [
                {
                    "matcher": "*",
                    "hooks": [{"type": "command", "command": f'echo "$TOOL_INPUT" > {output}'}],
                }
            ]
        })
        runner = HookRunner(config, workspace=str(tmp_path))
        await runner.trigger_notification("task complete")
        assert "task complete" in output.read_text()

    async def test_session_start_hook(self, tmp_path):
        marker = tmp_path / "started.txt"
        config = HooksConfig(hooks={
            "SessionStart": [
                {"matcher": "*", "hooks": [{"type": "command", "command": f"touch {marker}"}]}
            ]
        })
        runner = HookRunner(config, workspace=str(tmp_path))
        await runner.trigger_session_start()
        assert marker.exists()


# ─── Config Loading ──────────────────────────────────────────────────────────


class TestConfigLoading:
    def test_load_from_project_file(self, tmp_path):
        hooks_dir = tmp_path / ".nooa-coding"
        hooks_dir.mkdir()
        hooks_file = hooks_dir / "hooks.json"
        hooks_file.write_text(json.dumps({
            "hooks": {
                "PreToolUse": [
                    {"matcher": "Bash", "hooks": [{"type": "command", "command": "check.sh"}]}
                ]
            }
        }))
        config = load_hooks_config(tmp_path)
        rules = config.rules_for(HookEvent.PRE_TOOL_USE)
        assert len(rules) == 1
        assert rules[0].hooks[0].command == "check.sh"

    def test_settings_override(self, tmp_path):
        settings_hooks = {
            "PostToolUse": [
                {"matcher": "*", "hooks": [{"type": "command", "command": "lint"}]}
            ]
        }
        config = load_hooks_config(tmp_path, settings_hooks=settings_hooks)
        rules = config.rules_for(HookEvent.POST_TOOL_USE)
        assert len(rules) == 1

    def test_empty_when_no_files(self, tmp_path):
        config = load_hooks_config(tmp_path)
        assert config.rules_for(HookEvent.PRE_TOOL_USE) == []


# ─── Event Sink ──────────────────────────────────────────────────────────────


class TestEventSink:
    async def test_events_emitted(self):
        events: list[tuple[str, dict]] = []

        def sink(name: str, data: dict) -> None:
            events.append((name, data))

        config = HooksConfig(hooks={
            "PreToolUse": [
                {"matcher": "*", "hooks": [{"type": "command", "command": "exit 0"}]}
            ]
        })
        runner = HookRunner(config, workspace="/tmp", event_sink=sink)
        await runner.trigger_pre_tool_use("Bash", tool_input="ls")
        event_names = [e[0] for e in events]
        assert "hook_started" in event_names
        assert "hook_finished" in event_names

    async def test_block_event_emitted(self):
        events: list[tuple[str, dict]] = []

        def sink(name: str, data: dict) -> None:
            events.append((name, data))

        config = HooksConfig(hooks={
            "PreToolUse": [
                {"matcher": "*", "hooks": [{"type": "command", "command": "exit 1"}]}
            ]
        })
        runner = HookRunner(config, workspace="/tmp", event_sink=sink)
        await runner.trigger_pre_tool_use("Bash", tool_input="rm -rf /")
        event_names = [e[0] for e in events]
        assert "hook_blocked" in event_names


# ─── PermissionRequest ──────────────────────────────────────────────────────


class TestPermissionRequest:
    async def test_triggers_on_matching_kind(self, tmp_path):
        output = tmp_path / "perm.txt"
        config = HooksConfig(hooks={
            "PermissionRequest": [
                {
                    "matcher": "shell",
                    "hooks": [{"type": "command", "command": f'echo "$TOOL_NAME:$TOOL_INPUT" > {output}'}],
                }
            ]
        })
        runner = HookRunner(config, workspace=str(tmp_path))
        results = await runner.trigger_permission_request("shell", "rm -rf /", "dangerous")
        assert len(results) == 1
        assert results[0].returncode == 0
        content = output.read_text()
        assert "shell:" in content

    async def test_non_matching_kind_skips(self, tmp_path):
        config = HooksConfig(hooks={
            "PermissionRequest": [
                {"matcher": "file_write", "hooks": [{"type": "command", "command": "exit 1"}]}
            ]
        })
        runner = HookRunner(config, workspace=str(tmp_path))
        results = await runner.trigger_permission_request("shell", "ls", "")
        assert results == []


# ─── PreCompact / PostCompact ───────────────────────────────────────────────


class TestCompactHooks:
    async def test_pre_compact(self, tmp_path):
        output = tmp_path / "pre_compact.txt"
        config = HooksConfig(hooks={
            "PreCompact": [
                {
                    "matcher": "*",
                    "hooks": [{"type": "command", "command": f'echo "$TOOL_INPUT" > {output}'}],
                }
            ]
        })
        runner = HookRunner(config, workspace=str(tmp_path))
        results = await runner.trigger_pre_compact(context_summary="saving context")
        assert len(results) == 1
        assert results[0].returncode == 0
        assert "saving context" in output.read_text()

    async def test_post_compact(self, tmp_path):
        output = tmp_path / "post_compact.txt"
        config = HooksConfig(hooks={
            "PostCompact": [
                {
                    "matcher": "*",
                    "hooks": [{"type": "command", "command": f'echo "$TOOL_OUTPUT" > {output}'}],
                }
            ]
        })
        runner = HookRunner(config, workspace=str(tmp_path))
        results = await runner.trigger_post_compact(summary="compacted summary")
        assert len(results) == 1
        assert "compacted summary" in output.read_text()


# ─── SubagentStart / SubagentStop ───────────────────────────────────────────


class TestSubagentHooks:
    async def test_subagent_start(self, tmp_path):
        output = tmp_path / "start.txt"
        config = HooksConfig(hooks={
            "SubagentStart": [
                {
                    "matcher": "*",
                    "hooks": [{"type": "command", "command": f'echo "$TOOL_NAME:$TOOL_INPUT" > {output}'}],
                }
            ]
        })
        runner = HookRunner(config, workspace=str(tmp_path))
        results = await runner.trigger_subagent_start("worker-0", task="refactor auth")
        assert len(results) == 1
        assert results[0].returncode == 0
        content = output.read_text()
        assert "worker-0:refactor auth" in content

    async def test_subagent_stop(self, tmp_path):
        output = tmp_path / "stop.txt"
        config = HooksConfig(hooks={
            "SubagentStop": [
                {
                    "matcher": "*",
                    "hooks": [{"type": "command", "command": f'echo "$TOOL_OUTPUT" > {output}'}],
                }
            ]
        })
        runner = HookRunner(config, workspace=str(tmp_path))
        results = await runner.trigger_subagent_stop(
            "worker-1", result="done", status="completed"
        )
        assert len(results) == 1
        assert "completed: done" in output.read_text()

    async def test_subagent_matcher_filters(self, tmp_path):
        config = HooksConfig(hooks={
            "SubagentStart": [
                {"matcher": "worker-0", "hooks": [{"type": "command", "command": "exit 0"}]}
            ]
        })
        runner = HookRunner(config, workspace=str(tmp_path))
        # Matching worker.
        results = await runner.trigger_subagent_start("worker-0", task="test")
        assert len(results) == 1
        # Non-matching worker.
        results = await runner.trigger_subagent_start("worker-5", task="test")
        assert results == []
