# AGENTS.md — node-v26 worktree (V8 + Node)

> Full engineering standards (code quality, testing, performance, review, execution
> contract) live in the shared file at `/Users/aryden/dev/AGENTS.md`. **Read that
> for all of that.** This file is the *actionable* layer for this checkout:
> where work happens (and where it never does), and the build/CPU discipline that
> keeps this machine from OOM-killing the session.

## Before you run anything — preflight gate

Check these **before** every shell / git / build / `node` command. If any fail,
fix it or stop and ask — do not proceed.

1. **Right tree, right branch.** `pwd` and `git rev-parse --show-toplevel` must
   both report `/Users/aryden/dev/node-v26`, and you must be on `feature/green-threads`:
   ```sh
   pwd
   git rev-parse --show-toplevel
   git branch --show-current   # expect: feature/green-threads
   ```
   If either path or the branch is wrong, you're in the wrong checkout or on `main`.
2. **No build already running.** Nothing `ninja` / `clang` / `cc1plus` compiling
   (rule 2 under Build discipline, below).
3. **No zombie `node`.** Nothing left hung from a previous repro/test
   (rule 4 under Build discipline, below).

## Where work happens (and where it never does)

Work for **both V8 and Node** happens **only** in this worktree:

- **Path:** `/Users/aryden/dev/node-v26`
- **Repo / branch:** the Node.js repo (`~/dev/node`), branch `feature/green-threads`.
- **V8 home:** `deps/v8/` — this one tree is the source of truth for V8 too. A
  change to V8 and its equivalent Node change live together here.
- **Shell default:** every shell / git / build command MUST carry
  `workdir: /Users/aryden/dev/node-v26`. The OpenCode session cwd may default
  elsewhere — always `cd` here first.

**Never do feature work in:**

- `/Users/aryden/dev/v8` — upstream V8 (read-only reference)
- `/Users/aryden/dev/v8-14.6` — a separate V8 worktree, not our branch
- the `~/dev/node` main worktree (`main` branch)

Reading any of those to cross-check a commit or API is fine; editing them is not.
(This list also appears in the shared file — it's kept *here* on purpose, because
this rule needs to sit right next to the build commands where the mistake happens.)

## Build discipline (hard rule — memory/CPU safety)

This machine has ~30 cores. An unconstrained `ninja`/`clang` build spins up 30
compilers at once, exhausts RAM, and OOM-kills the session — it has before.
These rules apply to **every** build and process run.

1. **Cap parallelism to 20, always.** Every `ninja` MUST pass `-j20`:
   ```sh
   ninja -C out/Release node -j20
   ```
   Never `-j0`, never `-j` with no number, never `-j$(nproc)`. (`ulimit -p` limits
   CPU *time*, not concurrency — the only real lever is ninja's `-j`.)
2. **One build at a time.** Before starting a build, confirm nothing is compiling:
   ```sh
   ps -eo pid,args | grep -iE 'ninja|/clang|cc1plus' | grep -v grep
   ```
   If anything shows up, wait for it or `kill` it (`kill -9` only if genuinely
   stale) — never launch a second build over a running one.
3. **Capture output to a log, never through `head`/`grep`.** Pipe to `tee`:
   ```sh
   ninja -C out/Release node -j20 2>&1 | tee /tmp/build.log
   ```
   Closing the pipe with `head`/`grep` raises SIGPIPE upstream and kills ninja
   mid-build — it links fine but never finishes, leaving a stale binary. Afterward
   confirm the linker actually ran:
   ```sh
   grep -E 'error:|LINK node|ninja: build stopped' /tmp/build.log
   ```
   A log that ends at `2 warnings generated.` with no `LINK node` means ninja died
   before linking — rebuild.
4. **No zombie `node`.** A repro, the test runner, or a `GT_SHARED_HEAP=1` harness
   can hang (stuck green thread / deadlock) and leave a `node` burning cores
   silently. Before **and** after every `node` run:
   ```sh
   ps -eo pid,args | grep -E '[n]ode' | grep -v grep
   ```
   The `[n]ode` bracket matches `node` without matching the `grep` line. A stale
   `node` = kill it and find out *why* it hung before the next run — an unbounded,
   never-exiting process is a bug to diagnose, not something to rerun past.

## Why

We once spent effort chasing cross-worktree confusion — a change described as
"in v8" vs "in node-v26" when the two are separate checkouts of related code.
Keeping V8 and Node in one worktree makes the git diff sane and history linear.
The build rules exist because the 30-core box OOM-killed this session before.
