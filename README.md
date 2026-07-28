# NOOA Coding Agent

An independent, single-user coding-agent product built on the public NVIDIA
NOOA runtime. The application pins NOOA to a reviewed upstream commit, so an
upstream fetch or checkout cannot silently change agent behavior.

## What it provides

- a formal asynchronous `AgentSession` API;
- interactive terminal UI and one-shot CLI operation;
- streamed lifecycle, model, tool, and command-output events;
- cancel, resume, explicit compaction, session fork, and crash recovery;
- a dedicated Git worktree for every session;
- file and shell approval policies;
- diff, checkpoint, and rollback operations;
- layered project settings, `AGENTS.md`, and `SKILL.md` discovery;
- resource limits, durable replay traces, and ordered model failover.

The application is intentionally local. It does not contain a service layer,
multi-tenant database, worker queue, or web API.

## Quick start

```bash
uv sync
uv run nooa-code --repo /path/to/git/repository --model openai/gpt-5
```

Inside the terminal UI, enter a coding request or use `/help` to list session
commands. Project settings live in `.nooa-coding/settings.yaml`; the complete
schema is shown in `settings.example.yaml`.

## Python API

```python
from nooa_coding import AgentSessionManager, load_settings

manager = AgentSessionManager(repo, load_settings(repo))
session = manager.create()

turn = session.start("Fix the parser and run its focused tests")
async for event in session.stream():
    render(event)
    if event.name == "approval_requested":
        session.approve(event.data["request_id"])
    if event.name in {"turn_finished", "turn_failed", "turn_cancelled"}:
        break

result = await turn
checkpoint = session.checkpoint("parser-fixed")
child = await session.fork()
await session.close()
```

The same API supports `prompt()`, `cancel()`, `compact()`, `diff()`,
`rollback()`, `replay()`, `resume()`, and project-scoped session listing.

## Trust boundary

Every session edits a dedicated Git worktree. File and shell operations exposed
to the model pass through application-owned approval policy, timeout, and output
limits. A CodeAct middleware also rejects common attempts to bypass those tools
with direct Python file/process access.

These Python checks are guardrails, not an adversarial security sandbox. This
release targets a trusted single user running local repositories. Running
untrusted models, skills, repositories, or generated code requires an OS-level
container/VM boundary in addition to this application. Skills that are activated
by project configuration are trusted code.

See [architecture.md](docs/architecture.md) for lifecycle and persistence details.

## Dependency boundary

`pyproject.toml` and `uv.lock` pin `nooa`, `nooa-cli`, and `nooa-memory` to an
exact upstream commit. To upgrade:

1. update all three `rev` values together;
2. run `uv lock --upgrade-package nooa --upgrade-package nooa-cli --upgrade-package nooa-memory`;
3. run the complete test and type-check suite;
4. inspect the dependency diff before committing the new lockfile.

No code in this repository should require modifications to a NOOA checkout.
