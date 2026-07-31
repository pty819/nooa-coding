---
name: run-tests
description: Run project test suites, interpret failures, and iterate until green.
---

# Running Tests

## Discover the test command

1. Check `AGENTS.md` or project config for a configured test command.
2. Common patterns by ecosystem:
   - Python: `uv run pytest tests/ -x -q` or `python -m pytest`
   - Node: `npm test` or `npx jest`
   - Rust: `cargo test`
   - Go: `go test ./...`

## Execution workflow

```python
# 1. Run the full suite first to establish baseline
result = await self.shell.run("uv run pytest tests/ -x -q", timeout=120)
print(result.stdout)
print(result.stderr)
print(f"exit: {result.returncode}")
```

## On failure

1. Read the error output carefully — identify the failing test and assertion.
2. Use `await self.shell.read(path)` to inspect the failing test file.
3. Inspect the implementation file that the test exercises.
4. Fix the root cause (not the test, unless the test is wrong).
5. Re-run only the failing test for fast iteration:

```python
result = await self.shell.run("uv run pytest tests/test_foo.py::test_bar -x", timeout=60)
```

6. Once the targeted test passes, run the full suite to catch regressions.

## Rules

- Never delete or skip tests to make the suite pass.
- Never modify test assertions to match broken behaviour.
- If a test is genuinely wrong, explain why before changing it.
- Always run lint (`uv run ruff check src/`) alongside tests.
- Report the final test count and status in your evidence.
