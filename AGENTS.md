# NOOA Coding Agent

This repository is the application layer. It may depend on public NOOA APIs,
but it must not patch or import NOOA internals.

- Use `uv` for dependencies and commands.
- Keep model-driven work in one-purpose generation methods.
- Keep lifecycle, policy, verification, and recovery in deterministic Python.
- Never bypass approval policy from a prompt.
- Never mutate the user's primary checkout; coding sessions operate in dedicated Git worktrees.
- Add evidence-backed tests for every session or safety behavior.
