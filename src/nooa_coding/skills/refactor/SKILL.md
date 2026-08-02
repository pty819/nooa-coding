---
name: refactor
description: Systematic approach to refactoring code while preserving behaviour.
trigger: User says "refactor", "rename", "extract", "restructure", "clean up", or requests code reorganization without behaviour change
route: refactor
output-contract: Refactored code passing full test suite and lint; list of changed files; blocked status if tests are insufficient
---

# Refactoring Code

## Pre-flight

1. Ensure tests exist and pass before touching anything:

```python
result = await self.shell.run("uv run pytest tests/ -x -q", timeout=120)
assert result.returncode == 0, "Fix failing tests before refactoring"
```

2. Identify the refactoring scope — which files, which symbols.
3. Check for callers/consumers that will be affected:

```python
await self.shell.run("rg 'old_function_name' --type py -l")
```

## Execution strategy

- **Small, verifiable steps** — one logical change per edit.
- **Preserve public API** unless the task explicitly asks to change it.
- **Rename → Update callers → Verify** for each symbol.
- Use `await self.shell.replace(...)` for targeted edits.
- After each batch of edits, run affected tests immediately.

## Common patterns

### Extract function
1. Identify the code block to extract.
2. Write the new function with a clear name and docstring.
3. Replace the inline code with a call to the new function.
4. Run tests.

### Rename symbol
1. `rg 'old_name' --type py` to find all occurrences.
2. Replace in definition first, then all call sites.
3. Check string references (config keys, CLI args).
4. Run tests + lint.

### Restructure module
1. Create new module with the moved code.
2. Add re-exports in the old location for backwards compatibility.
3. Update imports across the codebase.
4. Remove re-exports only if the task says breaking changes are OK.
5. Run full test suite.

## Verification

```python
# Always end with full suite + lint
await self.shell.run("uv run ruff check src/", timeout=30)
result = await self.shell.run("uv run pytest tests/ -q", timeout=120)
```

## Rules

- Never refactor and add features in the same turn.
- Never leave the codebase in a broken intermediate state.
- Preserve type annotations and docstrings.
- If the refactoring is too risky without more tests, say `blocked` and explain what tests are needed.
