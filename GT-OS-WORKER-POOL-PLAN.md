# Plan: four concurrent GT mutators in one V8 isolate and heap

**Branch:** feature/green-threads  
**Worktree:** /Users/aryden/dev/node-v26  
**Parent fix:** 1190abcbc68 (handle-lifetime fixes)  
**Status:** architecture plan, not yet implemented

## Goal

Four OS workers execute four green-thread (GT) JavaScript bodies **simultaneously
in one green V8 isolate and its one ordinary V8 heap**. A global run-lock,
v8::Locker around each body, or four separate isolates does not meet this goal.
This requires a V8 multi-mutator fork: the public V8 contract currently permits
only one thread to use an isolate at a time (deps/v8/include/v8-locker.h).

The first acceptance test must prove overlapping JS execution on four distinct
OS threads in that same isolate. Four thread IDs in a log only prove that a
pool exists. The existing serial path remains a reference until the parallel
path passes GC, CoW, error, and teardown tests.

## Ground truth and immediate corrections

| Area | Current source behavior | Implication |
| --- | --- | --- |
| Driver | RunTasks enqueues GTs and ScheduleLoop drives them on the embedding OS thread (src/green_threads.cc). | A persistent worker pool needs a per-batch completion barrier. |
| Yield | GreenScheduler::Yield requeues its current GT while its JS/C++ stack is live, then enters another GT recursively (green-thread-scheduler.cc:190-230). | A second worker must never dequeue that GT. Disable this yield in the first parallel milestone. |
| Current GT | g_current_green_thread_ is process-global because earlier TLS symbols aliased across force-linked archives (green-thread.cc). | Make attribution worker-local and test the *actual barrier reader* against the scheduler writer in the linked Node binary. |
| Processor | GetOrCreateProcessor picks processors_[0] for every OS thread. The implemented queue is FIFO; its declared capacity is not enforced. | Bind worker w explicitly to processor P_w and synchronize queue operations. |
| Scheduler identity | RunTasks gets a scheduler keyed by the embedding isolate; the yield callback gets one keyed by its current green isolate. | Store the owner scheduler on the GT or its batch. |
| Heap and GC | LocalHeap and IsolateSafepoint support background heap work, but not arbitrary simultaneous JS entry. | Extend their contracts to JS mutators, including every live JS stack and register root. |
| CoW | The per-GT table, dirty set, and global active-count/GC gate were built around one active JS mutator. | Prove concurrent attribution, publication, and GC root visibility for four active views. |

Do not assume the TLS issue disappears through shared-library linkage:
node_shared defaults to false in common.gypi and node.gyp.

## Required invariants

### I1. Worker-local execution state

Each worker owns a current GT, processor, JS entry stack, handle scopes,
exception/try-catch state, stack limit, and GC root record. Audit every mutable
field accessed through Isolate::thread_local_top() and by Execution::Invoke,
interpreter/JIT entry, runtime calls, and callbacks. Each field must be
worker-local, immutable after publication, atomic, or protected by a narrow
lock. A process-global current GT is forbidden in parallel mode.

The fork must establish thread-specific V8 state without a whole-isolate
Locker. Isolate::Scope alone does not make concurrent entry safe. Existing
LocalHeap support is a starting point, not evidence that JS mutators are safe.

### I2. Exactly one owner per GT

A GT is exactly one of queued, running on worker w, safely parked, or complete.
Claim and state transitions are synchronized. A GT with live JS/C++ frames is
never enqueued or stolen. First run bodies to completion. Later suspension must
publish every live root and either keep the GT pinned to its original worker or
prove migration of its V8 thread state.

### I3. All-mutator GC handshake

Start with **stop-the-world GC**. One coordinator requests a safepoint; all
four mutators stop, publish their registers, stacks, handles, active GTs, and
CoW tables; the coordinator collects; then workers resume. Each allocation
slow path and native callback that can block must participate. A parked worker
may claim to be quiescent only after its roots are visible to GC. No object may
move while another worker holds an unreported pointer in a register.

Audit young/old allocation, compaction, remembered sets, write barriers, weak
handles, and the CoW roots provider. Existing LocalHeap/IsolateSafepoint
machinery can be reused only where its root and safepoint contracts cover JS
frames. Concurrent GC is an optional later optimization, not a prerequisite
for parallel JS.

### I4. Shared mutation semantics

Use per-worker allocation buffers or synchronize allocation slow paths. A GT's
CoW ref table and dirty set have one mutating owner; GC reads them only after
the handshake. Publish a clone before it can be observed. Audit shared Maps,
feedback vectors, inline caches, tiering state, Code publication, global
handles, microtask queues, and other isolate-global metadata. For ordinary JS
writes to the same object from different GTs, specify CoW isolation, explicit
shared atomic storage, or a synchronization primitive. C++ data races are not
acceptable JS semantics.

### I5. Batch lifetime

Workers are persistent. RunTasks prepares rooted contexts, functions,
arguments, GTs, and CoW views before publishing the batch. While workers run,
the embedding thread cannot access the green isolate or free batch state.
Each worker publishes one result or error, marks its GT complete, and
decrements the batch outstanding count with release semantics. RunTasks waits
with acquire semantics, then copies results, tears down CoW, and frees the
batch. At exit, stop and join workers before disposing the green isolate.
Replace the mutable global GT registry with immutable per-batch call data or
a direct GT-to-data association.

## Worker example (design sketch, not drop-in code)

The worker has a queue phase, a concurrent V8 entry phase, and a completion
phase. There is **no global execution lock** around RunBodyToCompletion.

~~~cpp
void Worker::Run() {
  BindProcessor(id_);                       // Explicit P_id, not P_0.
  RegisterGreenMutator(green_isolate_, id_); // Per-worker V8 and GC state.
  for (;;) {
    GreenThread* gt = queue_.WaitAndClaim(id_);
    if (gt == nullptr) break;               // Shutdown after batches drain.

    DCHECK(gt->owner_worker() == id_);
    SetCurrentGreenThread(gt);              // Worker-local barrier source.
    EnterConcurrentGreenJS(green_isolate_); // Implements I1-I4.
    RunBodyToCompletion(gt);                // No recursive Yield in M1.
    LeaveConcurrentGreenJS(green_isolate_); // Publish result/error and roots.
    SetCurrentGreenThread(nullptr);
    gt->batch()->Complete(gt);              // Release and notify RunTasks.
  }
  UnregisterGreenMutator(green_isolate_, id_);
}
~~~

EnterConcurrentGreenJS represents substantial V8 work, not a wrapper around
Isolate::Scope. Keep GreenJSEntry's existing scope handling on the legacy
serial path; factor common body logic after the parallel entry protocol exists.

Example first batch: four pure JS computations on disjoint inputs, such as
fib(30), fib(31), fib(32), and fib(33). Prepare and root all inputs before
publishing the batch. All four workers call into the *same* green isolate.
Begin with disjoint output objects; add shared-object CoW cases after entry,
allocation, and GC roots are validated.

## Scheduler specifics

- First use one mutex-protected global FIFO and a condition variable. Four
  concurrent mutators are the hard part; work stealing is an optimization.
- Claim removes a queued GT and assigns worker ownership in one synchronized
  transition. Complete occurs once. The batch outstanding count includes
  queued and running GTs; an empty queue means park, not batch completion.
- A worker asleep outside V8 must park its heap-access registration first.
  Wake predicates cover new work, safepoints, and shutdown to prevent lost
  notifications.
- The current recursive Yield is prohibited in parallel mode. After a real
  suspension boundary exists, resume on the original worker first. Add
  cross-worker migration and per-P stealing only with a separate root/thread
  state proof.
- GT_NUM_WORKERS=1 exercises the new worker path with one worker. Keep a
  separate legacy serial mode for comparison. Validate and cap the count
  before creating workers.

## Milestones and verification

**M0 — inventory and baseline.** Run existing green-thread tests and record
results. Map mutable isolate state, root and allocation paths, GC safepoints,
Code/feedback writes, and CoW accesses reached by a GT body. Fix the
scheduler-identity mismatch. Test TLS through the real linked write barrier,
not only two worker-local sentinel reads.

**M1 — architecture probe: four concurrent bodies.** Implement worker-local
entry state, immutable batch publication, synchronized claim/completion, and
disjoint preallocated inputs. Temporarily disable GC *only in this bounded
probe*. Four JS bodies reach a native rendezvous callback with live JS frames
in the same isolate; all four must arrive before any returns. After release,
sample JS progress on each worker and show overlapping execution intervals.
Set a timeout so serial execution fails rather than hangs. Do not ship M1 as
a general mode.

**M2 — general allocation, GC, and CoW.** Implement all-mutator safepoints,
root scanning, allocation and write-barrier protocols, and per-GT CoW
ownership. Force young and full collections while four workers are active.
Test disjoint writes, shared reads, private clones, errors, repeated batches,
and teardown. General parallel bodies are enabled only after this gate.

**M3 — yielding and scheduling.** Replace recursive Yield with a safe
suspension/resumption mechanism. Test block/wakeup and worker-pinned resume.
Add per-P queues and stealing only for GTs proven safe to move. Benchmark
one versus four workers and compare against the legacy serial path.

**M4 — optional performance work.** Profile GC pauses, lock contention,
allocation, CoW cloning, and actual worker utilization. Consider concurrent
GC only after the stop-the-world design is correct.

Follow AGENTS.md for every build: preflight, one build at a time, -j20,
complete log, and linker confirmation. Use focused repros first. Run the
existing Node green-thread suite, overlap and forced-GC tests, stress loops,
and TSan where supported. TSan is evidence, not a replacement for root and
semantic tests.

## Stop conditions

Do not enable general four-way workloads if a worker can enter JS while a
mutable isolate field has no concurrency rule; GC cannot enumerate every
worker's roots; a running GT can appear in a queue; or CoW can attribute a
write to another worker's GT. A passing CPU benchmark cannot waive these
conditions.
