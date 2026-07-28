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

## Turn routing and completion authority

The application routes each request before giving it tools:

- conversation: no repository tools; host-known questions such as the active
  model are answered directly from live session state;
- inspection: read-only file access and a strict allowlist of non-mutating shell
  commands;
- change: policy-controlled editing followed by host verification.

Generation methods investigate or implement one task. A deterministic Python
orchestrator owns the workflow and final status. For change turns it compares
the worktree before and after model execution, derives changed paths from Git,
runs `git diff --check`, and then runs configured or policy-approved behavioral
checks. A model-authored claim or file list cannot by itself produce a
`completed` result.

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

The terminal renderer intentionally hides internal lifecycle chatter while the
durable event stream retains it for replay and future clients. Rich output is
written through prompt-toolkit in raw VT100 mode, and provider-supplied ANSI
control sequences are stripped before Markdown rendering.

This project deliberately has no service/database/multi-tenant layer.
