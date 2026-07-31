"""Codebase search and LSP tools installed on the agent."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

from nooa import Skill
from nooa.agentdoc import spec


class CodeSearch(Skill):
    """Fast codebase search using ripgrep and file structure indexing."""

    __nosnapshot__ = True

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).resolve()
        super().__init__()

    async def search(
        self,
        query: Annotated[str, spec(description="Regex or literal pattern to search for")],
        *,
        file_glob: Annotated[str | None, spec(description="File glob filter, e.g. '*.py'")] = None,
        max_results: Annotated[int, spec(description="Maximum results to return")] = 30,
    ) -> str:
        """Search file contents using ripgrep. Returns matching lines with paths."""
        cmd = ["rg", "--line-number", "--no-heading", "--color=never", "-m", str(max_results)]
        if file_glob:
            cmd.extend(["--glob", file_glob])
        cmd.extend(["--", query, str(self._root)])
        result = await self._run(cmd)
        if not result.strip():
            return f"No matches found for: {query}"
        lines = result.splitlines()[:max_results]
        # Strip the root prefix for readability.
        prefix = str(self._root) + "/"
        cleaned = [line.replace(prefix, "") for line in lines]
        return "\n".join(cleaned)

    async def find_files(
        self,
        pattern: Annotated[str, spec(description="Glob pattern, e.g. '**/test_*.py'")],
        *,
        max_results: Annotated[int, spec(description="Maximum files to list")] = 50,
    ) -> str:
        """Find files matching a glob pattern."""
        matches: list[str] = []
        for path in sorted(self._root.rglob(pattern.removeprefix("**/"))):
            if any(part.startswith(".") for part in path.relative_to(self._root).parts):
                continue
            if "__pycache__" in str(path) or "node_modules" in str(path):
                continue
            matches.append(str(path.relative_to(self._root)))
            if len(matches) >= max_results:
                break
        if not matches:
            return f"No files matching: {pattern}"
        return "\n".join(matches)

    async def outline(
        self,
        path: Annotated[str, spec(description="File path relative to the worktree")],
    ) -> str:
        """Show the symbol outline of a file (classes, functions, methods)."""
        target = self._root / path
        if not target.is_file():
            return f"File not found: {path}"
        if target.suffix == ".py":
            return self._python_outline(target)
        # Fallback: show lines that look like definitions.
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        defs = [
            f"{i + 1}: {line.rstrip()}"
            for i, line in enumerate(lines)
            if any(
                line.lstrip().startswith(keyword)
                for keyword in ("def ", "class ", "function ", "export ", "pub ", "fn ")
            )
        ][:60]
        return "\n".join(defs) if defs else "No symbols detected."

    @staticmethod
    def _python_outline(path: Path) -> str:
        """Extract class/function definitions from a Python file."""
        import ast

        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            return f"Syntax error: {exc}"
        entries: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                entries.append(f"{node.lineno}: class {node.name}")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
                entries.append(f"{node.lineno}: {prefix}def {node.name}")
        entries.sort(key=lambda e: int(e.split(":")[0]))
        return "\n".join(entries[:80]) if entries else "No symbols found."

    @staticmethod
    async def _run(cmd: list[str]) -> str:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
        return stdout.decode(errors="replace")


class LSPTools(Skill):
    """Lightweight LSP integration for code intelligence."""

    __nosnapshot__ = True

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).resolve()
        super().__init__()

    async def definitions(
        self,
        path: Annotated[str, spec(description="File path relative to worktree")],
        line: Annotated[int, spec(description="1-based line number")],
        character: Annotated[int, spec(description="1-based column number")],
    ) -> str:
        """Find definitions of the symbol at a position using ctags/ripgrep fallback."""
        target = self._root / path
        if not target.is_file():
            return f"File not found: {path}"
        # Read the symbol at the given position.
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        if line < 1 or line > len(lines):
            return f"Line {line} out of range (file has {len(lines)} lines)"
        text_line = lines[line - 1]
        symbol = self._extract_symbol(text_line, character - 1)
        if not symbol:
            return f"No symbol found at {path}:{line}:{character}"
        # Search for definition patterns.
        return await self._find_definition(symbol)

    async def references(
        self,
        symbol: Annotated[str, spec(description="Symbol name to find references for")],
        *,
        file_glob: Annotated[str | None, spec(description="Limit search to glob")] = None,
    ) -> str:
        """Find all references to a symbol across the codebase."""
        cmd = ["rg", "--line-number", "--no-heading", "--color=never", "-m", "40"]
        if file_glob:
            cmd.extend(["--glob", file_glob])
        cmd.extend(["--", rf"\b{symbol}\b", str(self._root)])
        result = await CodeSearch._run(cmd)
        if not result.strip():
            return f"No references found for: {symbol}"
        prefix = str(self._root) + "/"
        lines = [line.replace(prefix, "") for line in result.splitlines()[:40]]
        return "\n".join(lines)

    async def symbols(
        self,
        query: Annotated[str, spec(description="Symbol name pattern to search")],
    ) -> str:
        """Search for symbol definitions matching a query."""
        patterns = [
            rf"def {query}",
            rf"class {query}",
            rf"function {query}",
            rf"const {query}",
        ]
        combined = "|".join(patterns)
        cmd = [
            "rg", "--line-number", "--no-heading", "--color=never",
            "-m", "20", "-e", combined, str(self._root),
        ]
        result = await CodeSearch._run(cmd)
        if not result.strip():
            return f"No symbol definitions matching: {query}"
        prefix = str(self._root) + "/"
        lines = [line.replace(prefix, "") for line in result.splitlines()[:20]]
        return "\n".join(lines)

    async def _find_definition(self, symbol: str) -> str:
        """Locate the definition of a symbol."""
        patterns = [
            rf"(def|class|function|fn|pub fn)\s+{symbol}\b",
            rf"^{symbol}\s*=",
        ]
        combined = "|".join(patterns)
        cmd = [
            "rg", "--line-number", "--no-heading", "--color=never",
            "-m", "10", "-e", combined, str(self._root),
        ]
        result = await CodeSearch._run(cmd)
        if not result.strip():
            return f"Definition not found for: {symbol}"
        prefix = str(self._root) + "/"
        lines = [line.replace(prefix, "") for line in result.splitlines()[:10]]
        return "\n".join(lines)

    @staticmethod
    def _extract_symbol(line: str, col: int) -> str:
        """Extract the identifier at a column position."""
        if col >= len(line):
            return ""
        start = col
        while start > 0 and (line[start - 1].isalnum() or line[start - 1] == "_"):
            start -= 1
        end = col
        while end < len(line) and (line[end].isalnum() or line[end] == "_"):
            end += 1
        return line[start:end]


__all__ = ["CodeSearch", "LSPTools"]
