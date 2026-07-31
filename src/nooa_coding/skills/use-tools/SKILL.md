---
name: use-tools
description: Reference guide for the agent's built-in capabilities and how to call them correctly.
---

# Using Your Built-in Tools

You have policy-controlled capabilities installed on `self`. Always call them
instead of raw filesystem/subprocess access.

## Shell — `self.shell`

```python
result = await self.shell.run("uv run pytest tests/ -x -q", timeout=120)
print(result.stdout, result.returncode)
```

- Commands run inside the isolated worktree.
- Read-only inspection mode restricts mutations automatically.
- Long-running commands: pass `timeout=` (seconds).
- Stdin: `await self.shell.run("cat", stdin="hello")`

## File reading — `self.shell.read`

```python
match = await self.shell.read("src/main.py", lines=(1, 50))
print(match.text)
```

## File editing — `self.shell.replace` / `self.shell.write_file`

```python
# Targeted replacement (preferred for edits):
result = await self.shell.replace("src/main.py", "old_text", "new_text")

# Full file creation:
result = await self.shell.write_file("src/new_module.py", content)
```

- Both require host approval unless permissions are set to allow.
- `replace` returns a `FileWrite` with `.path` and diff info.

## Repository tools — `self.repo`

```python
status = await self.repo.status()
log = await self.repo.log(n=10)
diff = await self.repo.diff()
```

## Todo list — `self.todo`

```python
self.todo.add("Implement the parser")
self.todo.complete("Implement the parser")
print(self.todo.status())
```

Keep the todo list current during multi-step changes.

## Skills — `self.skills`

```python
print(self.skills.status())  # list active skills
```

## Memory — `self.recall` / `self.remember`

```python
# Retrieve relevant knowledge before starting work:
memories = await self.recall("authentication flow")

# Store verified, durable knowledge:
await self.remember("The auth module uses JWT with 15-min expiry")
```

## Events — `self.events`

```python
print(self.events.keys())       # active event tags
print(self.events["tag_name"])  # read one event
```

## Notifications — `await self.notify(...)`

```python
await self.notify("Found 3 matching files, starting refactor")
```

Send concise progress updates only for meaningful milestones.

## Rules

1. Never use `open()`, `os`, `subprocess`, `pathlib`, or `socket` directly.
2. Never access attributes starting with `_` — they are host internals.
3. Always `await` async methods.
4. Use `doc(self)` at runtime to inspect the exact current API.
