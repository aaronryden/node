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

## Build Playbooks

Recurring, non-obvious gotchas specific to this V8 14.6 fork, and how to resolve them
without chasing core dumps. They live *here* (checkout-specific) so they sit next to the
build commands.

#### mksnapshot crashes after a header-driven size/layout change

**Symptom.** After adding an external reference (or any change that grows a layout
constant) and running the manual relink, `mksnapshot` aborts during bytecode-handler
compilation. Classic signals: `UNREACHABLE()` in `GraphBuilder::Run()`, or a null
`interpreter_` / `heap.safepoint()` while `Isolate::Init` runs.

**Root cause (the subtle one — not the layout itself).** A change to a header such as
`deps/v8/src/codegen/external-reference.h` grows `ExternalReferenceTable::kSizeIsolateIndependent` → `kSize` → `kSizeInBytes`, and that propagates through `external-reference-table.h` → `isolate-data.h` → `isolate-group.h`. Objects that were compiled *before* the edit (e.g. `isolate-group.o`, `heap.o`, `startup-serializer.o`) bake in the **old** count. The freshly rebuilt `external-reference-table.o` then writes `kSizeIsolateIndependent` (= old+1) entries into the stale `isolate-group.o`'s `external_ref_table_[old]` array — an 8-byte overflow that silently corrupts adjacent `Isolate`/`IsolateGroup` state. The result is null pointers deep inside `Isolate::Init`, which look like a V8 bug but are really a build-state mismatch.

**This is NOT a real layout bug.** The offsets are correct by construction:
- Every `k##CamelName##Offset` is generated by `DEFINE_ONE_FIELD_OFFSET` as a *running enum over declared field `Size`s* (via `ISOLATE_DATA_FIELDS`), so it self-adjusts when `ExternalReferenceTable::kSizeInBytes` grows.
- `static_assert(sizeof(ExternalReferenceTable) == kSizeInBytes)` holds after growth; `ref_name_` is a `static` (out-of-layout) member and `ref_addr_` grows by exactly `kEntrySize` bytes.
- All members accessed via hardcoded `Internals::kXXXOffset` precede `external_reference_table_`, so their offsets do not shift.

**Solution — real incremental `make -j20`, never manual relink alone.** Run `make -j20`
from `node-v26/` to rebuild every object whose changed header it includes (make tracks
header mtimes), then relink. The manual
`c++ <cwflags> <src> && ar r <archive> <obj> && /tmp/mksnap_link.sh` flow only *relinks*
`/tmp/mksnap_link.sh` — it never rebuilds `.o` files. That's fine for a single `.cc`
change that does **not** alter a header-driven size/layout constant, but it is the direct
cause of the corruption once such a constant changes. (Always pass `-j20`; see
"Build discipline".)

**Diagnosing which objects are stale (no core dump needed).** Compare header mtimes
against the candidate object mtimes:
```sh
stat -f " %Sm  %N" deps/v8/src/codegen/external-reference.h
stat -f " %Sm  %N" out/Release/obj.target/v8_base_without_compiler/deps/v8/src/init/isolate-group.o out/Release/obj.target/v8_base_without_compiler/deps/v8/src/heap/heap.o out/Release/obj.target/v8_base_without_compiler/deps/v8/src/snapshot/startup-serializer.o
```
Any `.o` whose timestamp **predates** the changed header is stale and must be rebuilt by `make -j20`.

**Related fixes in this area.**
- `graph-builder.cc`: Turboshaft can leave a branch successor without a terminal control (e.g. the dropped `IfFalse` edge of `SetNamedPropertyHandler`), so `case BasicBlock::kNone:` previously hit `UNREACHABLE()`. Emit a terminal `__ Unreachable()` instead.
- External references added to `EXTERNAL_REFERENCE_LIST` must also appear as a `FUNCTION_REFERENCE(...)` in `external-reference.cc` so the encoder can resolve their address.

#### Debugging green-thread CoW crashes with lldb (the child-isolation trap)

The concurrency test (`test/parallel/test-green-threads-concurrency.js`) runs the
stress workload **inside a `spawnSync(process.execPath, ['-e', CHILD_STRESS])` child**.
That has one hard consequence you cannot see from the parent:

> **A child killed by a signal is invisible to the parent.** `spawnSync` returns
> `res.status === null` with **empty** `res.stdout` / `res.stderr` (the crash
> output is lost when the process is signal-killed). So `grep`-ing the parent's
> log for `SEGV` / `EXC_BAD_ACCESS` finds *nothing*, and every real child crash
> looks like an innocuous `OTHER exit=1`. At 60×60 the earlier run reported
> "SEGV=0" — that was a measurement artifact; the SEGVs were the invisible
> "OTHER" bucket.

**Reproduce in-process, don't grep the parent.** Extract `CHILD_STRESS` to a
standalone file with `FIB_N`/`ROUNDS`/`CONCURRENCY` baked in (the env vars are
only interpolated at parent build time, so `STRESS_*` on the parent do *not*
propagate to the child) and run it directly — you then capture the child's own
report:
```sh
# /tmp/child_stress60.js  <-- copy of CHILD_STRESS, FIB_N=25 ROUNDS=60 CONCURRENCY=60
out/Release/node /tmp/child_stress60.js   # note the exit code / signal
```

**Two distinct 60×60 failure modes** (both kill the process, both must go):
- **Raw SIGSEGV mid-body.** Node prints `Stacktrace:` + `ptr0..ptr5` +
  `failure_message_object` + a `==== JS stack trace ====` (top frame
  `ApiCallbackExitFrame spawnAll` → `stressBody [line ~11]`), but **no**
  `----- Native stack trace -----`. Exit is a signal (133). This is the CoW
  copy-move race (mark-compact/scavenger relocating a copy while the owning GT
  holds it in an un-redirection-able CPU register).
- **FATAL** `Check failed: concurrent_marking_handle_ implies !concurrent_marking_handle_->IsValid()`
  at `ConcurrentMarker::~ConcurrentMarker()` ← `CppHeap::FinishMarkingAndProcessWeakness()`
  ← `MarkCompactCollector::CollectGarbage()` ← reason `kFinalizeMarkingViaStackGuard`.
  Full Node fatal report here; baseline ~1/25 even without the CoW race.

**macOS does NOT write cores** here (hardening). `ulimit -c unlimited` and
searching `/tmp/node.core.*`, `/tmp/core.*`, `./core.*` finds nothing. Do not
build your debugging around a core dump.

**lldb invocation (gotchas, all cost real debugging cycles):**
- `lldb -b -c <file>` is WRONG: `-c` is `--core <file>`, *not* a command file.
  Use `-o "command"` flags for batch input.
- **Batch needs `-k` past a stop.** In optimized builds `process launch` stops on the
  `EXC_BAD_ACCESS`/`EXC_BREAKPOINT` and lldb then *exits* batch mode, so any `-o` `bt`,
  `thread list`, or `register read` after it silently never runs — you only get the stop
  line. Suffix every post-stop command with `-k` (keep-going), e.g.
  `lldb -b out/Release/node -o 'env GT_SHARED_HEAP=1' -o 'process launch' -k 'thread list' -k 'thread select 1' -k 'bt 45' -k 'register read' -- out/Release/node`.
  (The `env` command sets the var; `settings set target.env-vars …` fails at `run`.)
- Detecting a "real" stop by grepping for your own `echo ===== STOP =====`
  marker is a **false positive** — that echo runs on every invocation (normal
  completion too). Detect a real stop via `grep -E "EXC_\|stop reason =\|Process .* stopped"`.
- **Hold the signal**, otherwise lldb passes it to Node's handler, which prints
  the JS-only stack and dies before you can inspect. Pre-run:
  ```
  process handle SIGSEGV SIGABRT SIGILL --notify true --stop true --pass false
  ```
  The abort surfaces as `EXC_BREAKPOINT` (from `IMMEDIATE_CRASH()`), usually on a
  `node-V8Worker` thread, at `frame #0: v8::base::OS::Abort()`.
- `bt 90` on the *worker* thread shows **only** `frame #0: OS::Abort` — the
  frames that led to the abort do not unwind (the signal interrupts mid-exec).
  The abort was *initiated* by the embedding thread reaching a global
  `Check`/`UNREACHABLE`. **You must dump the main thread's backtrace**, not just
  the aborting worker. lldb stops only the first thread; the main thread is
  usually #1 — `thread select 1` then `bt 90` (thread numbers vary per run, so
  grab the `thread list` first).

**The decisive architectural finding.** The abort (`OS::Abort`, from a global
`Check`/`UNREACHABLE` — *not* a torn-pointer SEGV at that line) surfaces on
**whichever thread hits the invariant first**: observed on both a
`node-V8Worker` thread and the `node-MainThread`. It does **not** reliably tell
you which thread is the culprit.

IMPORTANT: the worker concurrency is already gated by a real barrier — this is
**not** naive missing hygiene. Register a GC worker's entry via
`RegisterCOWGCWorker()` (bails if `ActiveCount() != 0`), and the CoW view's
`0 -> 1` boundary drains via `JoinCOWGCWorkers()`; wired into scavenger,
concurrent-marking, baseline-compiler, and maglev-dispatcher worker paths. So a
naive "workers aren't suppressed → they race" reading is wrong and will send the
next debug in circles. The remaining races are subtler:
- a worker registered *before* the view and its in-flight job stepping past the
  suspension poll boundary, or
- the abort is a *consequence* of a torn pointer created elsewhere (the main
  thread's `spawnAll → ScheduleLoop` JS stack is where the embedding thread was
  executing, and the rediracted pointer it handed a worker was already bad).

Root-cause next: `GreenThreadCow::ActiveCount` is bumped in
`GreenThreadCow::Initialize` (per green thread, 0→1, then drain) and dropped in
`Teardown` (→0); confirm it covers the whole `spawnAll`/`ScheduleLoop`
body-execution window and that a mid-round GC window (count 0) can't launch a
worker that then runs while the next round's view is active.

**RESOLVED (commit `7e294e07ee3`).** The 60×60 regression was **my own
deferral change**, not a missing worker hygiene. Empirically decisive:
- committed baseline `e34bcf11106` (no deferral): 40×40 = 20/20, **60×60 =
  35/35** (10 parent + 25 direct-child), cow = **15/20** (~25% child null-crash).
- with the deferral added: 60×60 dropped to ~15/25 (child SEGVs, invisible to
  the parent via `spawnSync`). So the deferral *regressed* 60×60.

The real cow crash was a **young-allocated FixedDoubleArray backing store**.
Every CoW copy is old space (JS objects, FixedArray, PropertyArray), but
`CloneCowBackingStore` made the `FIXED_DOUBLE_ARRAY` copy with
`NewFixedDoubleArrayWithHoles(...)` → default `kYoung` → scavenger-movable.
`parent.values.push(...)` builds a double-backed array; its CoW copy, being
nursery-allocated, is relocated mid-body while the owning GT holds the pointer
in a live register the redirect can't rewrite → torn pointer → child SEGV.
Fix = allocate it old space (`NewFixedDoubleArray(len, AllocationType::kOld)`),
matching the other copies. One-line, well-commented fix; cow → 20/20, 40×40 and
60×60 stay green.

**Do NOT re-add the deferral (historical lesson).** The old deferral gated
`Heap::HandleGCRequest` across 5 files and *regressed* the very case it aimed to
fix. Don't re-add it.

## Why

We once spent effort chasing cross-worktree confusion — a change described as
"in v8" vs "in node-v26" when the two are separate checkouts of related code.
Keeping V8 and Node in one worktree makes the git diff sane and history linear.
The build rules exist because the 30-core box OOM-killed this session before.
