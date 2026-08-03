"""Tests for CodeSearch and LSPTools (tools.py)."""

from __future__ import annotations

import pytest

from nooa_coding.multi_agent import TaskStatus, WorkerReport
from nooa_coding.tools import CodeSearch, LSPTools, TeamTools


@pytest.fixture
def code_root(tmp_path):
    """Create a small code tree for search tests."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text(
        "def hello():\n    return 'world'\n\n\ndef goodbye():\n    return 'bye'\n",
        encoding="utf-8",
    )
    (src / "utils.py").write_text(
        "class Helper:\n    def assist(self):\n        pass\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("# Project\nHello world\n", encoding="utf-8")
    return tmp_path


class TestCodeSearchContext:
    @pytest.mark.asyncio
    async def test_search_no_context(self, code_root):
        cs = CodeSearch(code_root)
        result = await cs.search("hello")
        assert "main.py" in result
        # Without context, should not include surrounding lines.
        assert "goodbye" not in result

    @pytest.mark.asyncio
    async def test_search_with_context_lines(self, code_root):
        cs = CodeSearch(code_root)
        result = await cs.search("hello", context_lines=2)
        # With 2 lines of context, should include surrounding code.
        assert "hello" in result
        # The line after "def hello():" is "return 'world'" — should appear.
        assert "return" in result

    @pytest.mark.asyncio
    async def test_search_with_glob(self, code_root):
        cs = CodeSearch(code_root)
        result = await cs.search("Project", file_glob="*.md")
        assert "README.md" in result
        assert "main.py" not in result

    @pytest.mark.asyncio
    async def test_search_no_match(self, code_root):
        cs = CodeSearch(code_root)
        result = await cs.search("nonexistent_pattern_xyz")
        assert "No matches" in result


class TestReadRange:
    @pytest.mark.asyncio
    async def test_read_range_basic(self, code_root):
        cs = CodeSearch(code_root)
        result = await cs.read_range("src/main.py", 1, 3)
        assert "1: def hello():" in result
        assert "2:" in result
        assert "3:" in result

    @pytest.mark.asyncio
    async def test_read_range_clamps(self, code_root):
        cs = CodeSearch(code_root)
        # Start < 1 should clamp to 1.
        result = await cs.read_range("src/main.py", -5, 2)
        assert "1:" in result

    @pytest.mark.asyncio
    async def test_read_range_end_beyond_file(self, code_root):
        cs = CodeSearch(code_root)
        result = await cs.read_range("src/main.py", 5, 999)
        # Should not error, just return up to end of file.
        assert "5:" in result

    @pytest.mark.asyncio
    async def test_read_range_file_not_found(self, code_root):
        cs = CodeSearch(code_root)
        result = await cs.read_range("nonexistent.py", 1, 10)
        assert "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_read_range_invalid(self, code_root):
        cs = CodeSearch(code_root)
        result = await cs.read_range("src/main.py", 5, 2)
        assert "Invalid range" in result


class TestFindFiles:
    @pytest.mark.asyncio
    async def test_find_python_files(self, code_root):
        cs = CodeSearch(code_root)
        result = await cs.find_files("*.py")
        assert "main.py" in result
        assert "utils.py" in result

    @pytest.mark.asyncio
    async def test_find_no_match(self, code_root):
        cs = CodeSearch(code_root)
        result = await cs.find_files("*.rs")
        assert "No files" in result


class TestOutline:
    @pytest.mark.asyncio
    async def test_python_outline(self, code_root):
        cs = CodeSearch(code_root)
        result = await cs.outline("src/main.py")
        assert "class" not in result
        assert "def hello" in result
        assert "def goodbye" in result

    @pytest.mark.asyncio
    async def test_outline_with_class(self, code_root):
        cs = CodeSearch(code_root)
        result = await cs.outline("src/utils.py")
        assert "class Helper" in result
        assert "def assist" in result


class TestLSPTools:
    @pytest.mark.asyncio
    async def test_references(self, code_root):
        lsp = LSPTools(code_root)
        result = await lsp.references("hello")
        assert "main.py" in result

    @pytest.mark.asyncio
    async def test_symbols(self, code_root):
        lsp = LSPTools(code_root)
        result = await lsp.symbols("Helper")
        assert "utils.py" in result

    @pytest.mark.asyncio
    async def test_definitions(self, code_root):
        lsp = LSPTools(code_root)
        # "hello" starts at col 5 in "def hello():"
        result = await lsp.definitions("src/main.py", 1, 5)
        assert "hello" in result


class TestWorktreeContainment:
    @pytest.mark.asyncio
    async def test_read_range_blocks_path_traversal(self, code_root):
        cs = CodeSearch(code_root)
        result = await cs.read_range("../../etc/passwd", 1, 5)
        assert "escapes worktree" in result

    @pytest.mark.asyncio
    async def test_outline_blocks_path_traversal(self, code_root):
        cs = CodeSearch(code_root)
        result = await cs.outline("../../../etc/passwd")
        assert "escapes worktree" in result

    @pytest.mark.asyncio
    async def test_definitions_blocks_path_traversal(self, code_root):
        lsp = LSPTools(code_root)
        result = await lsp.definitions("../../etc/passwd", 1, 1)
        assert "escapes worktree" in result

    @pytest.mark.asyncio
    async def test_read_range_allows_valid_path(self, code_root):
        cs = CodeSearch(code_root)
        result = await cs.read_range("src/main.py", 1, 2)
        assert "hello" in result


class _FakeDelegator:
    """Records delegate_preset calls and returns a canned report."""

    def __init__(self, report: WorkerReport) -> None:
        self._report = report
        self.calls: list[tuple[str, str, str | None]] = []

    async def delegate_preset(
        self, preset_name, objective, *, context_summary=None, on_progress=None
    ):
        self.calls.append((preset_name, objective, context_summary))
        return self._report


class TestTeamTools:
    def test_list_presets_lists_all_specialists(self):
        team = TeamTools(object())
        listing = team.list_presets()
        for name in ("search", "explore", "architect", "test", "executor", "pm"):
            assert name in listing

    @pytest.mark.asyncio
    async def test_delegate_invokes_delegator_and_formats_report(self):
        report = WorkerReport(
            task_id="t1",
            status=TaskStatus.COMPLETED,
            summary="found the auth handler",
            files_changed=["src/auth.py"],
            commits=["abcdef1234567890"],
        )
        delegator = _FakeDelegator(report)
        team = TeamTools(delegator)

        out = await team.delegate("search", "locate auth", context="uses JWT")

        assert delegator.calls == [("search", "locate auth", "uses JWT")]
        assert "Search Scout" in out
        assert "found the auth handler" in out
        assert "src/auth.py" in out
        assert "abcdef12" in out

    @pytest.mark.asyncio
    async def test_delegate_defaults_context_to_none(self):
        delegator = _FakeDelegator(
            WorkerReport(task_id="t2", status=TaskStatus.COMPLETED, summary="ok")
        )
        team = TeamTools(delegator)
        await team.delegate("explore", "map the module")
        assert delegator.calls == [("explore", "map the module", None)]

    @pytest.mark.asyncio
    async def test_delegate_unknown_preset_returns_error_string(self):
        team = TeamTools(object())
        out = await team.delegate("does-not-exist", "x")
        assert "unknown sub-agent preset" in out
