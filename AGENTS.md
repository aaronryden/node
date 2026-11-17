# AGENTS.md — node-v26 worktree (V8 + Node)

> Full engineering standards live in the shared file at `/Users/aryden/dev/AGENTS.md`
> (Code Quality, Testing, Performance, Architecture, Review Checklist, Execution
> Contract). Read it for all of that. This file adds the one local rule that matters
> most: **where** work happens.

## Hard rule: work only here

All agent work for **both V8 and Node** happens exclusively in this worktree.

- **Path:** `/Users/aryden/dev/node-v26`
- **Repo / branch:** the Node.js repo (`~/dev/node`), branch `feature/green-threads`.
- **V8 home:** `deps/v8/` — this single worktree is the source of truth for V8 too.
  A change to V8 and the equivalent Node change live in the same tree.

**Never do feature work in:**
- `/Users/aryden/dev/v8` — upstream V8 (read-only reference only)
- `/Users/aryden/dev/v8-14.6` — a separate V8 worktree, not our branch
- the `~/dev/node` main worktree (`main` branch)

## Tooling / commands

- **Every** shell / git / build command MUST run with `workdir: /Users/aryden/dev/node-v26`.
- The OpenCode session working directory can default elsewhere (e.g. the parent
  `~/dev` tree) — always set it explicitly first: `cd /Users/aryden/dev/node-v26`.
- Verify before acting: `pwd` and `git rev-parse --show-toplevel` should both report
  `/Users/aryden/dev/node-v26` before you touch code, git, or the build.

## Why

We once spent effort chasing cross-worktree confusion — a change described as "in v8"
versus "in node-v26" when the two are separate checkouts of related code. Keeping V8
and Node work in this one worktree makes the git diff sane and keeps history linear.
