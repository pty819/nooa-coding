"""Tests for CodeSearch and LSPTools (tools.py)."""

from __future__ import annotations

import pytest

from nooa_coding.tools import CodeSearch, LSPTools


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
