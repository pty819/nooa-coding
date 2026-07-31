---
name: create-pr
description: Prepare the worktree for submission — commit, push, and generate a PR description.
---

# Creating a Pull Request

## Pre-flight checks

1. Ensure all tests pass:

```python
result = await self.shell.run("uv run pytest tests/ -q", timeout=180)
assert result.returncode == 0
```

2. Ensure lint is clean:

```python
result = await self.shell.run("uv run ruff check src/", timeout=30)
assert result.returncode == 0
```

3. Review the diff for unintended changes:

```python
result = await self.shell.run("git diff --stat HEAD~1")
print(result.stdout)
```

## Commit

Write a clear, conventional commit message:

```python
await self.shell.run('git add -A')
await self.shell.run('git commit -m "feat: add user authentication with JWT"')
```

Format: `<type>: <imperative summary>`
- Types: feat, fix, refactor, docs, test, chore
- Summary: lowercase, no period, ≤72 chars

## Generate PR description

Produce a structured description:

```markdown
## Summary
<1-2 sentences: what changed and why>

## Changes
- <bullet list of key modifications>

## Testing
- <how it was verified: test commands, manual steps>

## Breaking Changes
- <any API/behaviour changes that affect consumers, or "None">
```

## Push

```python
result = await self.shell.run("git push -u origin HEAD", timeout=30)
print(result.stdout)
```

If push fails due to network restrictions, inform the user and provide the
branch name so they can push manually.

## Rules

- Never force-push.
- Never push to main/master directly — always a feature branch.
- Include test evidence in the PR description.
- If verification failed, do NOT push — report blocked status.
