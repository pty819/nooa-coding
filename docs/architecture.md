# Architecture

## Dependency boundary

`nooa-coding` is an application repository. It consumes public APIs from three
packages pinned to the same reviewed upstream commit:

- `nooa` for Agent, CodeAct, events, snapshots, compaction, and model clients;
- `nooa-cli` for repository navigation;
- `nooa-memory` for project-scoped long-term memory.

The application owns configuration layering, session lifecycle, workspaces,
approvals, traces, failover, and terminal interaction. It does not patch NOOA or
import runtime-private symbols.

## Session lifecycle

An `AgentSession` is the product-level unit. Creating it:

1. creates `nooa-coding/<session-id>` at a dedicated Git worktree;
2. creates a project-scoped session directory and SQLite event/snapshot store;
3. loads `AGENTS.md`, configured `SKILL.md` directories, and project settings;
4. installs memory and automatic token-budget compaction;
5. attaches the approval-controlled shell and repository tools;
6. creates an initial Git checkpoint.

Before and after every prompt, the agent snapshot is saved. Metadata is marked
`running` before model execution. If the process disappears, `resume()` detects
that state, restores the latest snapshot, preserves the worktree, and records a
crash-recovery event.

## State separation

- `session.db`: NOOA events and snapshots for one conversation.
- `events.jsonl`: stable host-level streaming/replay events.
- `metadata.json`: status, lineage, workspace, snapshots, and checkpoints.
- `memory.sqlite`: shared project memory across distinct sessions/worktrees.
- Git worktree branch: source changes and recoverable checkpoint commits.

Conversation fork copies the SQLite history, restores the latest snapshot, and
creates a new worktree branch from a source checkpoint. Code rollback resets only
the dedicated worktree and restores checkpointed agent state; it never resets the
user's primary checkout.

## Failure handling

Provider clients are ordered. Endpoint retries happen inside each NOOA client;
after a client still fails, `FailoverLLM` tries the next configured model and
emits `model_failover`. Shell commands have bounded duration, stdin, and retained
output. Active turns are cancellable, and cancellation also releases pending
approval futures before persisting a final snapshot.

This project deliberately has no service/database/multi-tenant layer.
