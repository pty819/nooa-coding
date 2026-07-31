---
name: explain-error
description: Diagnose an error message or traceback, find the root cause, and explain it clearly.
---

# Explaining Errors

## Workflow

1. **Reproduce** — run the command that produces the error:

```python
result = await self.shell.run("uv run pytest tests/ -x -q", timeout=60)
print(result.stderr or result.stdout)
```

2. **Locate** — identify the file, line number, and exception type from the traceback.

3. **Read context** — inspect the failing code and its callers:

```python
match = await self.shell.read("src/module.py", lines=(40, 80))
print(match.text)
```

4. **Trace dependencies** — check imports, configuration, and recent changes:

```python
await self.shell.run("git log --oneline -5 -- src/module.py")
await self.shell.run("git diff HEAD -- src/module.py")
```

5. **Explain** — produce a structured diagnosis:
   - What happened (exception + message)
   - Where (file:line + call chain)
   - Why (root cause reasoning)
   - How to fix (concrete next steps)

## Output format

Structure your answer as:

```
## Error
<exception type and message>

## Location
<file:line, function name>

## Root Cause
<1-3 sentences explaining why this happens>

## Fix
<numbered steps or code suggestion>
```

## Rules

- Always reproduce before explaining — do not guess from the message alone.
- Cite exact file paths and line numbers in your explanation.
- If the error is environment-related (missing dep, wrong version), say so explicitly.
- Do not modify code in inspect mode — only diagnose and explain.
