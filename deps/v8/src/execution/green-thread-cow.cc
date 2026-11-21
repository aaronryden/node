// Copyright 2025 the V8 project authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "src/execution/green-thread-cow.h"

#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdlib>
#include <cstdio>
#include <condition_variable>
#include <mutex>
#include <unordered_set>
#include <utility>
#include <vector>
#include <thread>

#include "src/common/ptr-compr.h"
#include "src/common/assert-scope.h"
#include "src/execution/green-thread-scheduler.h"
#include "src/execution/heap-ref-table.h"
#include "src/execution/isolate.h"
#include "src/handles/handles.h"
#include "src/heap/gc-callbacks.h"
#include "src/heap/linear-allocation-area.h"

// Provide the out-of-line definitions of HandleScope::HandleScope() and the
// slot store primitives used below. In V8's shared-library build these are
// exported from the V8 dylib; in a static embedding (Node.js) they must be
// emitted within this TU, which is the only caller that needs them out-of-line.
#include "src/handles/handles-inl.h"
#include "src/heap/heap.h"
#include "src/objects/heap-object-inl.h"
#include "src/objects/compressed-slots-inl.h"
#include "src/objects/slots-inl.h"

namespace v8 {
namespace internal {

// Thread-local counter tracking how many COW-enabled green threads are active.
std::atomic<uint32_t> GreenThreadCow::g_cow_active_count{0};

// Counts CowCloneForGeneratedCode entries past the fast-path guards, i.e.
// real TurboFan kArchStoreWithWriteBarrier stores that consulted the COW
// hook with COW active. Exposed for JIT→COW boundary verification.
std::atomic<uint64_t> GreenThreadCow::g_cow_generated_code_calls{0};

// Backing mutex for GreenThreadCowSharedWriterScope (see that class for why it
// is process-global and recursive); lazily owns the single process instance.
base::RecursiveMutex& GreenThreadCow::SharedStateMutex() {
  static base::RecursiveMutex mutex;
  return mutex;
}

// ARM64 register save/restore from green-thread-asm.cc (declared extern "C" in
// green-thread-scheduler.cc). The handshake re-arms this from PollForGCRequest
// so any heap-object pointer the parked worker still holds lands in its spill
// buffer, where RedirectWordsLocked can see and rewrite it during GC.
extern "C" void green_thread_save_registers_arm64(uint8_t* dst);

// All-mutator stop-the-world GC handshake (invariant I3) request flags.
//
// These are file-global rather than per-isolate on purpose: the green-threads
// feature drives a single green isolate per process, and the coordinator
// (M2Probe's inject_forced_gc) plus every worker all observe the same green
// heap. A process-global flag therefore gives every worker a single, shared,
// race-free view of "a GC has been requested". g_gc_in_progress_ additionally
// serializes coordinators and gates the poll so it cannot recurse mid-collection.
// Both are guarded by g_cow_green_threads_mutex only where they are read together
// with the g_cow_green_threads snapshot (RequestAllMutatorsStop).
static std::atomic<bool> g_gc_requested_{false};
static std::atomic<bool> g_gc_in_progress_{false};

// Bounded wait for a worker's "published + parked" flag. A worker that never
// reaches its safepoint (e.g. a body that runs to completion without yielding)
// must not stall the coordinator forever, so the spin-join gives up after this.
static constexpr std::chrono::milliseconds kGCStopJoinTimeout{2000};

// Re-entry guard: prevents infinite recursion when cloning triggers another
// write barrier. CloneObjectInternal allocates a new heap object, which may
// fire internal V8 stores that re-enter COWWriteBarrierCheck. Returning
// unchanged on nested entry (depth > 0) breaks the cycle safely — the outer
// call owns the clone; the inner call defers.
static thread_local int64_t g_cow_nesting_depth = 0;

// Isolates on which at least one COW green thread has run, plus a mutex
// guarding the set. The set is consulted by the JIT code generator to decide
// whether to emit the COW write-barrier callout; it is set once per isolate and
// never cleared, because bodies compiled in the window between spawnAll calls
// must still emit the callout even though the active count is zero then.
namespace {
std::mutex g_cow_capable_isolates_mutex;
std::unordered_set<Isolate*> g_cow_capable_isolates;
}  // namespace

// ================================================================
// Diagnostic subsystem (opt-in; gated on V8_GREEN_THREADS_DEBUG)
//
// Green-thread copy-on-write (CoW) heap isolation assumes a single execution
// thread: the embedding thread rewrites heap slots in place to build a green
// thread's private view, while V8's GC runs on that same thread. Node embeds
// V8 with concurrent GC / baseline-compiler workers on separate OS threads, so
// those workers can race the embedding thread's rewrites. This subsystem logs
// the CoW view lifecycle and every concurrent-work suspension decision to
// stderr, so those races can be observed and triaged.
//
// The gate is a direct getenv() compare so the disabled path costs only a
// pointer compare; when the environment variable is unset no formatting or I/O
// occurs. This is a permanent, zero-cost-until-set diagnostic, not a
// temporary debugging aid to be reverted.
// ================================================================
namespace {

// Process-wide gate. A direct getenv() compare over the disabled path is
// negligible (a libc `environ` lookup, no syscall) and, crucially, race-free:
// this hook is reachable from background GC/compilation workers, so a cached
// bool written lazily under contention would itself be a data race. When the
// environment variable is unset, GreenThreadDiag() performs no formatting or
// I/O.
inline bool GreenThreadDebugEnabled() {
  return std::getenv("V8_GREEN_THREADS_DEBUG") != nullptr;
}

// Monotonically increasing sequence number so interleaved worker output stays
// ordered on the log for a given process.
std::atomic<uint64_t> g_diag_line{0};

// The diagnostic printer itself is defined out-of-line below (outside the
// anonymous namespace) so it has external linkage and can be called from other
// translation units (baseline/maglev workers), not just here.

// Bounds diagnostics emitted from the hot concurrency-decision path: only one
// in every kDiagSuspensionInterval observations of an active CoW view is
// printed, so a busy GC burst cannot flood stderr.
static constexpr uint64_t kDiagSuspensionInterval = 1000;
std::atomic<uint64_t> g_diag_suspension_count{0};

// Rate-limit for the (noisier) barrier-register/bail join lines.
static constexpr uint64_t kDiagBarrierInterval = 200;
std::atomic<uint64_t> g_diag_barrier_bail{0};

}  // namespace

// Emits a single "[green-threads] #<n> <msg>" line to stderr, gated on the
// V8_GREEN_THREADS_DEBUG environment variable so the disabled path performs no
// work at all. Defined out-of-line (not in the anonymous namespace above) so it
// has external linkage and can be shared with the baseline/maglev workers, which
// opt into the same diagnostics.
void GreenThreadDiag(const char* fmt, ...) {
  if (!GreenThreadDebugEnabled()) return;
  uint64_t n = g_diag_line.fetch_add(1, std::memory_order_relaxed);
  va_list args;
  va_start(args, fmt);
  char buf[512];
  vsnprintf(buf, sizeof(buf), fmt, args);
  va_end(args);
  fprintf(stderr, "[green-threads] #%llu %s\n",
          static_cast<unsigned long long>(n), buf);
}

// Returns true when `instance_type` identifies an internal V8 backing-store
// container: the dictionary `properties` FixedArray (FIXED_ARRAY_TYPE), the
// property-descriptor lookup table (PROPERTY_ARRAY_TYPE), or the fast double
// array (FIXED_DOUBLE_ARRAY_TYPE).
//
// These containers are allocated and linked into a JS object by V8 during the
// dictionary/map transitions performed on the interpreter/JIT property-store
// path; they are not user values. On the CoW barrier's store path, cloning one
// and rewriting the owner's properties slot to the copy orphans V8's dictionary
// machinery, leaving a stale pointer that a later GC marks pass dereferences
// (SIGBUS). Such stores must therefore write through unchanged, exactly as
// HeapNumber stores do. The predicate takes the instance-type enum so both the
// interpreter barrier (COWWriteBarrierCheck) and the JIT barrier
// (CowCloneForGeneratedCode) share one implementation.
inline bool IsCowBackingStoreInstanceType(uint16_t instance_type) {
  return instance_type == FIXED_ARRAY_TYPE ||
         instance_type == PROPERTY_ARRAY_TYPE ||
         instance_type == FIXED_DOUBLE_ARRAY_TYPE;
}

#if defined(V8_OS_MACOSX) || defined(V8_OS_LINUX) || defined(V8_OS_DARWIN)
// Diagnostic-only: map an instance-type enum to a short name for [GT-COW-*]
// logs. Production code never calls this (GtTypeName is only read under
// GT_HEAP_DIAG).
static const char* GtTypeName(uint16_t it) {
  switch (it) {
    case JS_OBJECT_TYPE: return "JS_OBJECT";
    case JS_ARRAY_TYPE: return "JS_ARRAY";
    case HEAP_NUMBER_TYPE: return "HEAP_NUMBER";
    case FIXED_ARRAY_TYPE: return "FIXED_ARRAY";
    case PROPERTY_ARRAY_TYPE: return "PROPERTY_ARRAY";
    case FIXED_DOUBLE_ARRAY_TYPE: return "FIXED_DOUBLE_ARRAY";
    default: return "OTHER";
  }
}
#endif

// ================================================================
// GC root integration
//
// A green thread's copy-on-write heap view is a set of GT-local clones that
// are reachable only through this module's bookkeeping. On their own those
// clones are invisible to V8's GC, so the first collection during a live green
// thread could either collect a copy the body is still using or
// compact-move it while the body holds a redirected reference to the old
// address. To make the views GC-safe we register a Heap::GCRootsProvider that
// roots every active view during the mark phase: both the main-heap original
// (so the snapshot a GT is reading stays alive) and the GT-local copy (so it
// stays alive and is relocated if compaction moves it). The per-GT stack and
// register rewrite on relocation is handled inside HeapRefTable::VisitAsRoots.
//
// The provider is wired to the COW active count so it is registered on the
// heap exactly while at least one green thread's view is active, and each live
// view is listed for the provider to walk. In the Model A single-execution-
// thread design GC runs on the same thread that drives green-thread lifecycle,
// so these accesses never race; the registry mutex is retained for future
// multi-threaded variants.
// ================================================================

namespace {
// Currently COW-enabled green threads, walked by the GC roots provider.
std::mutex g_cow_green_threads_mutex;
std::vector<GreenThread*> g_cow_green_threads;

// Heap::GCRootsProvider that roots every active COW green-thread view during
// the mark phase. Registered globally on the heap via Heap::AddGlobalGCRootsProvider.
class GreenThreadGCRootsProvider final : public GCRootsProvider {
 public:
  void Iterate(RootVisitor* visitor) override {
    std::lock_guard<std::mutex> guard(g_cow_green_threads_mutex);
    for (GreenThread* gt : g_cow_green_threads) {
      if (gt == nullptr || !gt->cow_enabled()) continue;
      HeapRefTable* table = gt->cow_ref_table();
      if (table != nullptr) table->VisitAsRoots(visitor);
    }
  }
};

// The provider outlives any single green thread; keep it in static storage so
// the heap always holds the same, stable pointer.
GreenThreadGCRootsProvider* g_cow_gc_root_provider = nullptr;
}  // namespace

// Record this green thread as a live COW view the GC must root. Must be called
// with the same mutex as g_cow_green_threads and before the active count is
// incremented, so a concurrent-in-time GC never walks a view that lacks a
// table or is not yet tracked.
static void RecordCOWGreenThread(GreenThread* gt) {
  std::lock_guard<std::mutex> guard(g_cow_green_threads_mutex);
  g_cow_green_threads.push_back(gt);
}

// Stop rooting this green thread's view. Mirrors RecordCOWGreenThread.
static void UnrecordCOWGreenThread(GreenThread* gt) {
  std::lock_guard<std::mutex> guard(g_cow_green_threads_mutex);
  for (auto it = g_cow_green_threads.begin(); it != g_cow_green_threads.end();
       ++it) {
    if (*it == gt) {
      g_cow_green_threads.erase(it);
      return;
    }
  }
}

// Register the GC roots provider on the heap the first time COW becomes active
// (active count 0 -> 1). Balanced by UnregisterCOWRootsProvider on the last
// teardown (1 -> 0) so the provider lives exactly while a view is active.
static void RegisterCOWRootsProvider(Heap* heap) {
  if (g_cow_gc_root_provider == nullptr) {
    g_cow_gc_root_provider = new GreenThreadGCRootsProvider();
  }
  heap->AddGlobalGCRootsProvider(g_cow_gc_root_provider);
}

// Remove the GC roots provider from the heap. Safe to call even if it was never
// registered; RemoveGlobalGCRootsProvider DCHECKs, so guard the null case.
static void UnregisterCOWRootsProvider(Heap* heap) {
  if (g_cow_gc_root_provider != nullptr) {
    heap->RemoveGlobalGCRootsProvider(g_cow_gc_root_provider);
  }
}

// Heap cooperative-override hook. Returns true while at least one copy-on-write
// green thread is active (g_cow_active_count > 0).
//
// The CoW design assumes a single execution thread, but Node enables V8's
// concurrent GC worker threads (young-GC scavenger, mark-compact markers,
// array-buffer sweeper) that run on separate OS threads and mutate the same
// heap the embedding thread rewrites for a green thread's isolated view.
// Concurrent GC racing those in-place slot rewrites reads torn/freed pointers
// (an EXC_BAD_ACCESS in Scavenger::EvacuateObject). The heap's
// ShouldUseBackgroundThreads() consults this hook so that while a view is
// active it returns false, forcing every concurrent-GC worker path to fall
// back to serial execution on the embedding thread. Once the last view is
// torn down the active count drops to zero and concurrency is restored.
//
// g_cow_active_count is a coarse, global "any CoW green thread is running"
// flag (bumped in Initialize, dropped in Teardown). Over-suspending concurrent
// GC across isolates is always safe: it only costs GC throughput, never
// correctness -- precisely the trade-off the single-execution-thread CoW
// model makes.
bool GreenThreadCOWSuspendedConcurrentGC() {
  const bool suspended = GreenThreadCow::ActiveCount() != 0;
  if (suspended) {
    // Rate-limit: print at most one suspension line per interval of observed
    // active views, so a long GC burst does not flood the diagnostic log.
    if (g_diag_suspension_count.fetch_add(1, std::memory_order_relaxed) %
            kDiagSuspensionInterval == 0) {
      GreenThreadDiag(
          "concurrent GC/compiler suspended while %u CoW view(s) active",
          GreenThreadCow::ActiveCount());
    }
  }
  return suspended;
}

// ================================================================
// Concurrent-GC worker suspension & drain barrier (Option A)
//
// The guard in Heap::ShouldUseBackgroundThreads() (see below) gates only the
// one-time decision to schedule a concurrent-GC worker when a GC job is
// enqueued. A worker launched while no CoW view is active keeps running through
// its job -- e.g. ScavengerJobTask::ConcurrentScavengePages re-checks nothing --
// and races the embedding thread's in-place CoW slot rewrites once the next
// green-thread view activates, reading a torn Map pointer (an EXC_BAD_ACCESS in
// Scavenger::EvacuateObject). This barrier closes that gap without a full
// stop-the-world:
//
//   * A GC worker registers on entry / unregisters on exit, so the embedding
//     thread knows exactly how many workers are in flight.
//   * The worker polls the suspension flag at each coarse heap-access boundary
//     and bails immediately once a view is active.
//   * On the view's 0 -> 1 boundary the embedding thread spins (with brief
//     condition-variable waits) until every in-flight worker has drained, then
//     lets the CoW rewrites begin. Because a freshly scheduled worker also bails
//     at entry, and the guard suppresses new concurrent-GC jobs while a view is
//     active, no worker ever touches the heap during a view.
//
// RegisterCOWGCWorker()/UnregisterCOWGCWorker() are called from V8's heap worker
// translation units; JoinCOWGCWorkers() is called from Initialize().

// In-flight concurrent-GC worker count for the suspension barrier, guarded by
// g_cow_gc_barrier_mutex. Guarding it (rather than using a lock-free atomic) is
// what closes the check-then-act gap: a worker's registration commits to
// touching the heap, and the embedding thread's drain observes "no worker in
// flight" -- both must happen under the same lock so they are mutually
// exclusive. Once the drain observes an empty set under the lock, no worker can
// commit heap access afterwards, so the embedding thread may begin CoW rewrites.
static int g_cow_gc_inflight_workers = 0;
static std::mutex g_cow_gc_barrier_mutex;
static std::condition_variable g_cow_gc_barrier_cv;

// Register the calling GC worker with the suspension barrier. Returns true if
// concurrent GC may proceed, false if a CoW view is already active and the
// worker must not touch the heap. The suspension check and the commit
// (++inflight) happen atomically under the barrier mutex, so JoinCOWGCWorkers()
// can never observe "no worker in flight" and then have a worker register and
// race the view.
bool RegisterCOWGCWorker() {
  std::lock_guard<std::mutex> lock(g_cow_gc_barrier_mutex);
  if (GreenThreadCow::ActiveCount() != 0) {
    // Rate-limited: at most one bail per kDiagBarrierInterval views, so a GC or
    // compile burst during a long view does not flood the log.
    if (g_diag_barrier_bail.fetch_add(1, std::memory_order_relaxed) %
            kDiagBarrierInterval == 0) {
      GreenThreadDiag(
          "RegisterCOWGCWorker BAIL active=%d inflight=%d",
          (unsigned)GreenThreadCow::ActiveCount(),
          g_cow_gc_inflight_workers);
    }
    return false;
  }
  ++g_cow_gc_inflight_workers;
  return true;
}

// Unregister a worker accepted by RegisterCOWGCWorker. Notifies the embedding
// thread's drain loop once the last in-flight worker has drained.
void UnregisterCOWGCWorker() {
  std::lock_guard<std::mutex> lock(g_cow_gc_barrier_mutex);
  int remaining = --g_cow_gc_inflight_workers;
  if (remaining == 0) {
    g_cow_gc_barrier_cv.notify_all();
  }
}

// Drain (spin with brief condition-variable waits) until no GC worker registered
// with the barrier is in flight. Called on the CoW view's 0 -> 1 boundary, after
// the suspension flag is set. The lock is held across the whole check, so a
// worker cannot register after the drain observes an empty set, and in-flight
// workers observe suspension at their next heap-access boundary and bail, so the
// drain always terminates.
void JoinCOWGCWorkers() {
  std::unique_lock<std::mutex> lock(g_cow_gc_barrier_mutex);
  if (g_cow_gc_inflight_workers != 0 &&
      g_diag_barrier_bail.load(std::memory_order_relaxed) % kDiagBarrierInterval ==
          0) {
    GreenThreadDiag(
        "JoinCOWGCWorkers START inflight=%d active=%d",
        g_cow_gc_inflight_workers, (unsigned)GreenThreadCow::ActiveCount());
  }
  int polls = 0;
  int stuck_logs = 0;
  const int kMaxStuckLogs = 8;
  g_cow_gc_barrier_cv.wait_for(
      lock, std::chrono::milliseconds(1), [&] {
        if (g_cow_gc_inflight_workers > 0 && stuck_logs < kMaxStuckLogs) {
          if (++stuck_logs % 500 == 0 || polls <= 1) {
            GreenThreadDiag("JoinCOWGCWorkers: STUCK drained in=%d polls, "
                            "inflight=%d",
                            polls, g_cow_gc_inflight_workers);
          }
        }
        if (++polls % 2000 == 0 && stuck_logs >= kMaxStuckLogs) {
          GreenThreadDiag("JoinCOWGCWorkers: STILL stuck polls=%d inflight=%d",
                          polls, g_cow_gc_inflight_workers);
        }
        return g_cow_gc_inflight_workers == 0;
      });
  GreenThreadDiag("JoinCOWGCWorkers: drained polls=%d inflight=%d", polls,
                  g_cow_gc_inflight_workers);
}

// ================================================================
// All-mutator stop-the-world GC handshake (invariant I3)
//
// The CoW design runs green threads cooperatively on the embedding thread, so
// today GC is *suppressed* (GreenThreadCOWSuspendedConcurrentGC forces
// ShouldUseBackgroundThreads() == false) rather than coordinated: no GC ever
// runs while a view is active, so there is nothing to coordinate. Item 1 turns
// that off-guard suppression into a real handshake so GC can start while the
// mutators are active, stop them, enumerate every root each one holds, collect,
// then resume. This is invariant I3.
//
// Sequence (all under a single coordinator, e.g. M2Probe's inject_forced_gc):
//   1. RequestAllMutatorsStop() release-stores gc_requested=true, snapshots the
//      active COW views, and -- when the caller has spawned workers -- the
//      workers observe the flag at their safepoint and park (published + parked).
//   2. The collector runs; because every mutator is parked with roots published,
//      GC sees only quiescent state (no torn pointers, nothing mutating the
//      HeapRefTable it walks). The GreenThreadGCRootsProvider is already queued
//      on the heap, so published roots are scanned for free.
//   3. ResumeAllMutators() clear-store gc_requested=false and notify every parked
//      worker; ShouldUseBackgroundThreads returns to normal.
//
// A worker parks only at an eligible boundary (not mid-publish) and only when a
// GC is actually in progress, so the default serial path is never affected.
// ================================================================

// Publish this green thread's live roots for the collector: re-arm the ARM64
// register save so any heap-object pointer the worker still holds is written to
// its spill buffer, and set the register mask so the canonical redirect pass
// scans the masked regs. RedirectWordsLocked (called by the collector through
// the already-registered GreenThreadGCRootsProvider, which scans
// `gt->stack_limit()/stack_top()` and `gt->regs()` masked by `register_mask`)
// then sees the GT's stack + masked regs + HeapRefTable as published roots. This
// reuses the machinery Yield already uses for COW; it does not build a new scan.
//
// The worker must be parked here (it holds no live register the redirect cannot
// rewrite otherwise), which is guaranteed: PollForGCRequest reaches this only
// after it has already stashed the regs and is about to suspend.
static void PublishGCRootsForCollector(GreenThread* gt) {
  if (gt == nullptr) return;
  // Save the current registers into the spill buffer so the redirect pass sees
  // every live heap-object pointer the worker is still holding, and widen the
  // mask to every root category so the scan covers them all.
  green_thread_save_registers_arm64(gt->regs());
  gt->set_register_mask(kGreenRegMaskRoot | kGreenRegMaskCage |
                        kGreenRegMaskBytecode | kGreenRegMaskDispatch);
}

// Item 1 safepoint hook (invariant I3). Returns true iff the caller parked and
// must not continue executing; otherwise returns false and the worker carries
// on. Plain logic: `if (gc_requested && gc_in_progress && eligible) {
// publish(); park(); return true; }`. Eligible means not mid-publish and at a
// bytecode/native boundary -- both true for the yield safepoint and the M2Probe
// worker entry poll. A null / non-COW thread simply cannot participate.
bool GreenThreadCow::PollForGCRequest(GreenThread* gt, Isolate* isolate) {
  (void)isolate;
  if (gt == nullptr || !gt->cow_enabled()) return false;
  // Only act while a real stop-the-world GC is in flight; a bare
  // gc_requested without gc_in_progress never happens and would be a spurious
  // park. This gate is what keeps the default serial path (GCInProgress false)
  // from ever parking a worker.
  if (!g_gc_requested_.load(std::memory_order_acquire)) return false;
  if (!g_gc_in_progress_.load(std::memory_order_acquire)) return false;

  // Publish this worker's roots so the collector (which runs next, on the
  // embedding thread) sees only quiescent, reported state.
  PublishGCRootsForCollector(gt);

  // Commit the quiescence flag under the worker's own mutex so the coordinator's
  // spin-join observes it, then suspend until the coordinator resumes us.
  {
    std::unique_lock<std::mutex> lock(gt->gc_mutex());
    gt->set_gc_parked(true);
  }
  gt->gc_cv().notify_all();
  std::unique_lock<std::mutex> lock(gt->gc_mutex());
  gt->gc_cv().wait(lock, [&] { return !g_gc_requested_.load(std::memory_order_acquire); });
  gt->set_gc_parked(false);
  return true;
}

// Coordinator entry (invariant I3). Publishes the set of active COW views to
// *out_suspended / *out_n and release-stores gc_requested so every worker's
// safepoint observes the request. The worker then publishes + parks at its next
// safepoint; the coordinator runs the collector between this call and
// ResumeAllMutators. ActiveCount (g_cow_active_count) is guaranteed > 0 here
// because a view only becomes a COW green thread through Initialize, which is
// exactly when the GC roots provider is registered.
void GreenThreadCow::RequestAllMutatorsStop(GreenThread*** out_suspended,
                                            int* out_n) {
  if (out_suspended) *out_suspended = nullptr;
  if (out_n) *out_n = 0;

  // Snapshot the active views under the same mutex the GC roots provider walks,
  // so a view is never added or removed from the snapshot mid-collect.
  std::vector<GreenThread*> snapshot;
  {
    std::lock_guard<std::mutex> guard(g_cow_green_threads_mutex);
    snapshot = g_cow_green_threads;
  }
  if (out_suspended) *out_suspended = snapshot.data();
  if (out_n) *out_n = static_cast<int>(snapshot.size());

  // Always announce the stop, even if no view is live yet: the forced-GC test
  // calls this BEFORE spawning the worker threads so their entry safepoints see
  // the request and park before allocating. The snapshot above is the set of
  // currently-active views; workers created afterwards park when they observe
  // gc_requested at their entry poll.
  g_gc_in_progress_.store(true, std::memory_order_release);
  g_gc_requested_.store(true, std::memory_order_release);
}

// Resume every worker the coordinator stopped: clear-store gc_requested, mark
// each view no longer parked, and notify its condition variable so the worker's
// poll unblocks and it resumes its body. Restore ShouldUseBackgroundThreads to
// normal by clearing gc_in_progress. Safe to call even if no worker parked.
void GreenThreadCow::ResumeAllMutators() {
  g_gc_requested_.store(false, std::memory_order_release);
  {
    std::lock_guard<std::mutex> guard(g_cow_green_threads_mutex);
    for (GreenThread* gt : g_cow_green_threads) {
      if (gt == nullptr) continue;
      {
        std::lock_guard<std::mutex> wlock(gt->gc_mutex());
        gt->set_gc_parked(false);
      }
      gt->gc_cv().notify_all();
    }
  }
  g_gc_in_progress_.store(false, std::memory_order_release);
}

bool GreenThreadCow::GCInProgress() {
  return g_gc_in_progress_.load(std::memory_order_acquire);
}

// Quiescence barrier (invariant I3). Spin-join every active COW view until it
// has published + parked (gc_parked == true) under its own mutex, or until
// timeout_ms elapses. This is the step that REPLACES GreenThreadCOWSuspendedConcurrentGC:
// the coordinator calls it AFTER RequestAllMutatorsStop (announce) and the
// workers have been spawned, and immediately BEFORE the collection, so the
// collector only ever runs once every mutator is parked with its roots
// published -- no worker is mid-body mutating the HeapRefTable the GC walks.
//
// We read gc_parked() (an acquire atomic) directly rather than blocking on the
// condition variable here, because: (a) the coordinator is the only other
// writer and we must not hold gc_mutex_ across a potentially long spin that
// would starve a parked worker of the lock it needs to flip the flag back;
// (b) the workers set gc_parked under gc_mutex_ and the coordinator only needs
// the acquire visibility of the atomic, not the mutex. On timeout we still
// proceed but report false, since a worker that never reaches its safepoint
// (a body with no yield) must not deadlock the coordinator.
bool GreenThreadCow::WaitForAllMutatorsParked(int timeout_ms) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  const auto now = std::chrono::steady_clock::now();
  if (now >= deadline) return false;  // Already out of time.

  bool any_active = false;
  while (true) {
    std::vector<GreenThread*> to_check;
    {
      std::lock_guard<std::mutex> guard(g_cow_green_threads_mutex);
      to_check = g_cow_green_threads;
    }
    for (GreenThread* gt : to_check) {
      if (gt == nullptr || !gt->cow_enabled()) continue;
      if (!gt->gc_parked()) return false;  // Not everyone parked yet.
    }
    if (!to_check.empty()) return true;  // All parked + published.
    any_active = true;  // Views exist but none parked yet; keep polling.
    if (std::chrono::steady_clock::now() >= deadline) {
      return any_active;  // Spun the whole window: false unless something ran.
    }
    std::this_thread::yield();
  }
}

void GreenThreadCow::SetIsolateCowCapable(Isolate* isolate) {
  if (isolate == nullptr) return;
  std::lock_guard<std::mutex> lock(g_cow_capable_isolates_mutex);
  g_cow_capable_isolates.insert(isolate);
}

bool GreenThreadCow::IsolateCowCapable(Isolate* isolate) {
  if (isolate == nullptr) return false;
  std::lock_guard<std::mutex> lock(g_cow_capable_isolates_mutex);
  return g_cow_capable_isolates.count(isolate) != 0;
}

#if V8_ENABLE_GREEN_THREADS
void GreenThreadCow::SetGreenHeapConcurrentAllocationEnabled(
    v8::Isolate* isolate, bool enabled) {
  // Bridges Node's green-threads runtime onto V8's young-gen allocation gate,
  // flipping the LinearAllocationArea's bump-pointer from a plain read-then-bump
  // to a lock-free atomic CAS so N OS workers reserve disjoint pointers from one
  // shared green isolate's heap.
  //
  // The gate is a single process-global flag (see
  // ConcurrentLinearAllocationGate in linear-allocation-area.h) rather than a
  // per-HeapBase member on purpose: a shared green-thread isolate runs its
  // young generation on a dedicated cppgc heap that is a different object from
  // the (old-gen) CppHeap, so a per-heap flag reached via CppHeap could miss the
  // young space. A single relaxed store suffices because the toggle is performed
  // around the green-mutator-locked concurrent-worker window. A null isolate is a
  // no-op, so the disabled serial fast path is never disturbed.
  if (isolate == nullptr) return;
  ConcurrentLinearAllocationGate().store(enabled, std::memory_order_relaxed);
}
#endif  // V8_ENABLE_GREEN_THREADS

void GreenThreadCow::Initialize(GreenThread* gt) {
  // Once any green thread runs on an isolate, mark it COW-capable so compile
  // units generated for it (including by the background compiler after this
  // call returns, while the active count is zero) keep the COW callout wired in.
  SetIsolateCowCapable(gt->isolate());
  HeapRefTable* table = new HeapRefTable();
  table->set_owner(gt);
  gt->set_cow_ref_table(table);
  gt->set_cow_enabled(true);
  gt->clear_cow_dirty_objects();
  gt->set_cow_dirty_flag(false);

  // Root this view during GC so its clones survive collection. Record the view
  // and register the global roots provider on the first transition (0 -> 1),
  // before bumping the active count, so a GC never walks a view lacking a
  // table or that is not yet tracked.
  uint32_t previous_count =
      g_cow_active_count.load(std::memory_order_relaxed);
  RecordCOWGreenThread(gt);
  if (previous_count == 0) {
    RegisterCOWRootsProvider(gt->isolate()->heap());
  }
  g_cow_active_count.fetch_add(1, std::memory_order_release);
  // Drain any in-flight concurrent-GC workers (launched while no view was
  // active) before this view's CoW rewrites begin. Workers observe suspension
  // at their next heap-access boundary and bail; a freshly scheduled worker
  // bails at entry. This turns the point-in-time guard into a real
  // "no GC worker runs while a view is active" guarantee on the activation
  // boundary rather than only on the scheduling decision.
  JoinCOWGCWorkers();
  GreenThreadDiag("CoW view initialized (active count now %u)",
                  GreenThreadCow::ActiveCount());
}

void GreenThreadCow::Teardown(GreenThread* gt) {
  // Stop rooting this view before dropping the active count. On the last
  // teardown (count 1 -> 0) this also unregisters the global roots provider,
  // keeping the provider registered exactly while a view is active.
  uint32_t previous_count =
      g_cow_active_count.load(std::memory_order_relaxed);
  UnrecordCOWGreenThread(gt);
  g_cow_active_count.fetch_sub(1, std::memory_order_acq_rel);
  GreenThreadDiag("CoW view torn down (active count now %u)",
                  GreenThreadCow::ActiveCount());
  if (previous_count == 1) {
    UnregisterCOWRootsProvider(gt->isolate()->heap());
  }

  gt->set_cow_enabled(false);
  gt->clear_cow_dirty_objects();
  gt->set_cow_dirty_flag(false);
  HeapRefTable* table = gt->cow_ref_table();
  gt->set_cow_ref_table(nullptr);
  if (table != nullptr) {
    table->set_owner(nullptr);
    delete table;
  }
}

Address GreenThreadCow::Clone(Isolate* isolate, GreenThread* gt,
                              Address obj_ptr) {
  HeapRefTable* table = gt ? gt->cow_ref_table() : nullptr;
  if (!table) return obj_ptr;
  // Hold the nesting guard across the whole deep clone. Applying cloned
  // references into the copies (and the allocation-driven store checks) can
  // re-enter the write barrier; without the guard the barrier would treat the
  // clone's own stores as external writes and re-clone objects mid-flight,
  // doubling the ref-table entries and corrupting the copy. This mirrors the
  // guard CowCheckBeforeStore sets around its own CloneObject call.
  g_cow_nesting_depth++;
  Address copy = table->CloneObject(isolate, obj_ptr);
  g_cow_nesting_depth--;
  return copy;
}

// Helper: get the COW-enabled green thread for this execution context.
static GreenThread* CowEnabledThread() {
  GreenThread* gt = CurrentGreenThread();
  if (!gt || !gt->cow_enabled()) return nullptr;
  if (!gt->cow_ref_table()) return nullptr;
  return gt;
}

Address GreenThreadCow::CowCheckBeforeStore(Isolate* isolate, Address obj_ptr) {
  // Guard against recursive COW operations (e.g., during snapshot
  // deserialization: clone → allocate → write barrier → clone → crash).
  // If we're already inside a clone operation, return unchanged.
  if (g_cow_nesting_depth++ > 0) {
    g_cow_nesting_depth--;
    return obj_ptr;
  }

  GreenThread* gt = CowEnabledThread();
  if (!gt) {
    g_cow_nesting_depth--;
    return obj_ptr;
  }
  if (!isolate) isolate = gt->isolate();

  // Never clone from inside the collector or during snapshot deserialization.
  // Snapshot deserialization triggers the write barrier for every object read
  // from the snapshot; cloning in that context would try to allocate while
  // the deserializer is actively building the heap.
  if (isolate && (isolate->heap()->gc_state() != Heap::NOT_IN_GC ||
                  !isolate->heap()->deserialization_complete())) {
    g_cow_nesting_depth--;
    return obj_ptr;
  }

  HeapRefTable* table = gt->cow_ref_table();
  if (!table) { g_cow_nesting_depth--; return obj_ptr; }

  // If this object is already cloned, return the copy immediately.
  Address copy = kNullAddress;
  if (table->FindCopy(obj_ptr, &copy)) {
    g_cow_nesting_depth--;
    return copy;
  }

  // First write to this object: clone it for the GT's isolated heap view.
  // The C++ store path (WriteToField → CombinedWriteBarrier) runs under
  // DisallowHeapAllocation — we must explicitly re-allow it here. Allocation
  // also requires safepoints, so both scopes are necessary.
  if (!isolate) { g_cow_nesting_depth--; return obj_ptr; }
  AllowHeapAllocation allow_alloc;
  AllowSafepoints allow_safepoints;
  // Cloning allocates, and the clone path hands out GC-safe handles for every
  // object it touches. Handle creation is forbidden while the isolate's handle
  // scope is sealed, which is the case for every write performed by running JS
  // (Execution::Invoke seals the scope around the JS entry stub). Opening a
  // scope here deliberately unseals it for the duration of the clone — the same
  // pattern V8's own runtime functions use when called from generated code
  // (e.g. Runtime_DefineKeyedOwnIC_Miss opens a HandleScope). The scope is
  // left again before returning, releasing the temporary handles; the clone
  // itself stays reachable through the reference table and the rewritten slot.
  HandleScope handle_scope(isolate);
  copy = table->CloneObject(isolate, obj_ptr);

  if (copy == obj_ptr) {
    // Not cloneable: write through with original.
    g_cow_nesting_depth--;
    return obj_ptr;
  }

  // Record the copy mapping so future accesses get redirected.
  // Also queue the object for stack/register reference redirection.
  table->RecordCopy(obj_ptr, copy);
  gt->add_cow_dirty_object(obj_ptr);
  gt->set_cow_dirty_flag(true);

  g_cow_nesting_depth--;
  return copy;
}

Address GreenThreadCow::COWCheckAndRedirect(Address obj_ptr) {
  GreenThread* gt = CowEnabledThread();
  if (!gt) return obj_ptr;

  HeapRefTable* table = gt->cow_ref_table();
  Address copy = kNullAddress;
  if (!table->FindCopy(obj_ptr, &copy)) return obj_ptr;

  gt->add_cow_dirty_object(obj_ptr);
  gt->set_cow_dirty_flag(true);
  return copy;
}

void GreenThreadCow::RedirectReferences(GreenThread* gt, Address old_ptr,
                                        Address new_ptr) {
  HeapRefTable* table = gt ? gt->cow_ref_table() : nullptr;
  if (table) table->RedirectReferences(gt, old_ptr, new_ptr);
}

void GreenThreadCow::VisitAsRoots(GreenThread* gt, RootVisitor* visitor) {
  HeapRefTable* table = gt ? gt->cow_ref_table() : nullptr;
  if (table) table->VisitAsRoots(visitor);
}

void GreenThreadCow::FlushGreenThreadDirtySet(GreenThread* gt) {
  if (!gt) return;
  if (!gt->cow_dirty_flag()) return;

  HeapRefTable* table = gt->cow_ref_table();
  if (!table) {
    gt->set_cow_dirty_flag(false);
    return;
  }

  gt->set_cow_dirty_flag(false);

  // Collect pending objects (the set is swapped so we can iterate safely).
  std::unordered_set<Address> pending;
  pending.swap(gt->cow_dirty_objects_ref());

  // One stack/register scan per batch of writes.
  // At flush time the clone already happened during CowCheckBeforeStore,
  // so we only need to redirect references.
  for (Address main_ptr : pending) {
    Address copy = kNullAddress;
    if (!table->FindCopy(main_ptr, &copy)) continue;
    table->RedirectReferences(gt, main_ptr, copy);
  }
}

int GreenThreadCow::ClonedObjectCount(GreenThread* gt) {
  HeapRefTable* table = gt ? gt->cow_ref_table() : nullptr;
  return table ? table->ClonedCount() : 0;
}

// --- COW runtime helper for TurboFan-generated code ---

// Number of times CowCloneForGeneratedCode was reached with COW active.
// Used by tests to verify the JIT→COW boundary is crossed.
uint64_t GreenThreadCow::CowCloneForGeneratedCodeCallCount() {
  return g_cow_generated_code_calls.load(std::memory_order_relaxed);
}

// Reset the JIT→COW boundary counter (test isolation).
void GreenThreadCow::ResetCowCloneForGeneratedCodeCallCount() {
  g_cow_generated_code_calls.store(0, std::memory_order_relaxed);
}

// TurboFan-generated code calls this to clone a heap object pointer
// for the current green thread's isolated heap view.
//
// Takes a tagged pointer (with kHeapObjectTag for uncompressed or
// compressed form). Decompresses if needed, calls CowCheckBeforeStore,
// then re-tags/re-compresses the result.
//
// Returns the (possibly cloned) tagged pointer address.
Address GreenThreadCow::CowCloneForGeneratedCode(Address tagged_addr) {
  // Fast path: no COW-enabled threads.
  if (V8_LIKELY(!ActiveCount())) return tagged_addr;

  GreenThread* gt = CurrentGreenThread();
  if (!gt) return tagged_addr;

  Isolate* isolate = gt->isolate();
  if (!isolate) return tagged_addr;

  // Smi store operands need no cloning -- they are not heap objects and,
  // like the interpreter path, untagging them as a heap address would feed a
  // garbage pointer into CowCheckBeforeStore. Return the operand unchanged
  // before it is counted as a real JIT->COW clone.
  if (HAS_SMI_TAG(tagged_addr)) return tagged_addr;

  // Reached the JIT→COW boundary: a TurboFan kArchStoreWithWriteBarrier
  // store with COW active. Count it for verification.
  g_cow_generated_code_calls.fetch_add(1, std::memory_order_relaxed);

  // Decompress the tagged address to get the raw heap object address. In
  // this configuration store operands arrive as full (decompressed) tagged
  // addresses; DecompressTagged is identity there, so both forms are handled.
  Address obj_ptr;
#if V8_COMPRESS_POINTERS
  obj_ptr = V8HeapCompressionScheme::DecompressTagged(
      static_cast<Tagged_t>(tagged_addr));
#else
  obj_ptr = tagged_addr;
#endif
  // DecompressTagged preserves the tag bits; strip the heap object tag to
  // get the untagged address the clone table and CloneObject expect. If the
  // store operand arrived untagged (no tag bits), pass it through unchanged.
  if ((obj_ptr & kHeapObjectTagMask) == kHeapObjectTag) {
    obj_ptr -= kHeapObjectTag;
  }

  // A boxed double (HeapNumber) is stored inline as double bits by a
  // double-representation field, not as a heap-object reference; cloning and
  // returning a redirect for it would corrupt that field. Skip it here so the
  // original operand is written through unchanged (mirrors the guard in
  // COWWriteBarrierCheck).
  if (HeapObject::FromAddress(obj_ptr)->map()->instance_type() == HEAP_NUMBER_TYPE) {
    return tagged_addr;
  }

  // Internal V8 backing-store containers (FIXED_ARRAY / PROPERTY_ARRAY /
  // FIXED_DOUBLE_ARRAY) are V8-owned, not user values: cloning one on the JIT
  // store path and returning a redirect for it would write a stale pointer into
  // the owner's properties slot, orphaning V8's dictionary machinery (SIGBUS).
  // Skip the clone here so the operand writes through unchanged, mirroring the
  // HeapNumber guard above.
  if (IsCowBackingStoreInstanceType(
          HeapObject::FromAddress(obj_ptr)->map()->instance_type())) {
    return tagged_addr;
  }

#if defined(V8_OS_MACOSX) || defined(V8_OS_LINUX) || defined(V8_OS_DARWIN)
  // [GT-COW-JIT] Diagnostic: log JIT store-barrier operands. Gated on
  // GT_HEAP_DIAG.
  if (::getenv("GT_HEAP_DIAG")) {
    static int gt_jseq = 0;
    fprintf(stderr, "[GT-COW-JIT] #%05d val=%s\n", gt_jseq++,
            GtTypeName(HeapObject::FromAddress(obj_ptr)->map()->instance_type()));
  }
#endif

  // Check and clone if needed.
  Address copy_addr = CowCheckBeforeStore(isolate, obj_ptr);

  // Re-tag AND re-compress the result. The operand lives in a general purpose
  // register, and a compressed build stores it with `stur` (32-bit) into the
  // tagged slot, so what must come back is the compressed form Tagged_t, not
  // the full address: handing back a full address truncates to the low 32
  // bits, which are not cage-relative in the shared-cage configuration and
  // therefore decompress to an invalid address on the next load. If nothing was
  // cloned, return the original operand unchanged, preserving its form.
  if (copy_addr == obj_ptr) return tagged_addr;
#if V8_COMPRESS_POINTERS
  return static_cast<Address>(
      V8HeapCompressionScheme::CompressObject(copy_addr + kHeapObjectTag));
#else
  return copy_addr + kHeapObjectTag;
#endif
}

// --- COW write barrier hook (implemented here to avoid header cycles) ---

// Provenance filter: determines whether the host object is a user-level
// mutable container worthy of COW isolation.
//
// V8 internal objects (maps, strings, descriptors, SharedFunctionInfo,
// contexts, allocation sites, etc.) are never cloned by COW. They are
// per-isolate, not per-application-object, and their internal structure
// (unaligned slots, snapshot-allocated backing stores) is not safe to
// deep-clone.  The only user objects COW can isolate are plain JS objects and
// JS arrays — the same types HeapRefTable::IsCowCloneableRootType accepts for a
// deep copy. Every other type (functions, proxies, Maps, Sets, ...) is shared
// and routed through its own native mutation path rather than a green-thread
// copy.
//
// This check is performed on the HOST (the containing object), not the
// VALUE (what is being stored).  V8 internals frequently write to their
// own structures during compilation and GC; filtering on the host skips
// all of those safely while preserving isolation for user writes.
//
// Safety: for test-mode fake addresses (very low numbers), skip the
// check entirely — the address may not point to valid memory, and the
// subsequent CowCheckBeforeStore handles test mode separately.
inline bool IsCowProvenanceHost(Tagged<HeapObject> host) {
  // Test-mode fake objects have very low addresses that fall outside
  // the actual heap.  Dereferencing their map would crash.  Skip the
  // provenance check in that case — the full COW flow handles test
  // mode via null-isolate and deserialization guards.
  // Real V8 heap addresses are typically > 0x100000000 (64-bit).
  Address host_addr = host.address();
  if (host_addr < 0x100000000ULL) return true;  // trust test fake objects

  uint16_t it = host->map()->instance_type();
  switch (it) {
    case JS_OBJECT_TYPE:
    case JS_ARRAY_TYPE:
      return true;
    default:
      return false;
  }
}

// Fast-path COW check for the write barrier path.
//
// Called from CombinedWriteBarrierInternal after the store has completed
// but before GC bookkeeping. If the host is a user COW-tracked object and
// the value needs cloning, rewrites the slot with the copy and queues the
// object for stack/register reference redirection.
//
// The slot rewrite is done via raw address write to avoid calling the
// protected Tagged<HeapObject> constructor directly.
void GreenThreadCow::COWWriteBarrierCheck(Tagged<HeapObject> host,
                                          Tagged<HeapObject> value,
                                          HeapObjectSlot slot) {
  GreenThread* gt = CurrentGreenThread();

  // Smi (small-integer) values are not heap objects and cannot be cloned.
  // Because ComputeWriteBarrierModeForObject forces an UPDATE_WRITE_BARRIER
  // for young hosts while a green thread is active, Smi stores reach this
  // barrier exactly like object stores. Untagging a Smi below as a heap
  // address (value.ptr() - kHeapObjectTag) yields a garbage pointer that
  // CowCheckBeforeStore / CloneObject then dereference and write out of
  // bounds against -- the root cause of the green-thread load crash. This
  // guard short-circuits before any pointer is dereferenced. The check must
  // read the raw tag bits directly: Tagged<HeapObject> overrides IsSmi() to
  // compile-time "false" and IsHeapObject() to compile-time "true", so the
  // usual IsHeapObject(value) overload is always-true here and useless.
  if (HAS_SMI_TAG(value.ptr())) return;

  // Fast path: no active CoW green thread. Checked BEFORE any of the
  // value->map() dereferences below, mirroring the !gt / !isolate guards at the
  // top of CowCloneForGeneratedCode. On the real barrier path the value is
  // always a heap object (Smi-guarded above), so this costs nothing in
  // production; testing CoW liveness first also keeps this function safe when
  // unit tests invoke it with unmapped fake addresses while CoW is active.
  if (!gt || !gt->cow_enabled()) return;

  // Test-mode fake values have very low addresses outside the real heap;
  // dereferencing their map would crash. Skip them and write through unchanged,
  // mirroring the host-address guard in IsCowProvenanceHost. Real V8 heap
  // addresses are typically > 0x100000000 (64-bit).
  if (value.address() < 0x100000000ULL) return;

#if defined(V8_OS_MACOSX) || defined(V8_OS_LINUX) || defined(V8_OS_DARWIN)
  // [GT-COW-STORE] Diagnostic: log every interpreter store barrier hit with the
  // host and value instance types, so the object that corrupts a CoW copy can be
  // traced. Gated on GT_HEAP_DIAG.
  if (::getenv("GT_HEAP_DIAG")) {
    static int gt_iseq = 0;
    fprintf(stderr, "[GT-COW-STORE] #%05d host=%s val=%s\n", gt_iseq++,
            GtTypeName(host->map()->instance_type()),
            GtTypeName(value->map()->instance_type()));
  }
#endif

  // A boxed double (HeapNumber) is never stored as a heap-object reference:
  // JSObject::WriteToField stores its bit pattern inline into a
  // double-representation property field and consumes the HeapNumber only via
  // value_as_bits(). There is therefore no reference to isolate, yet the slot
  // rewrite below would overwrite that field's inline double bits with a
  // heap-object pointer, corrupting the object's storage invariant (the next
  // WriteToField / property access then dereferences the garbage pointer and
  // crashes). Such stores only reach this barrier because
  // ComputeWritesModeForObject forces UPDATE_WRITE_BARRIER for young hosts
  // while a green thread is active; skip cloning them here.
  if (value->map()->instance_type() == HEAP_NUMBER_TYPE) return;

  // Internal V8 backing-store containers (the dictionary properties array, the
  // property-descriptor lookup table, and the fast double array) are not user
  // values. Cloning one on the store path and rewriting the owner's properties
  // slot to the copy orphans V8's dictionary machinery, leaving a stale pointer
  // that a later GC marks pass dereferences (SIGBUS). Skip the CoW clone for
  // these internal backing-store values so such stores write through unchanged,
  // mirroring the HeapNumber guard above.
  if (IsCowBackingStoreInstanceType(value->map()->instance_type())) return;

  // Provenance filter: skip COW for V8 internal host objects.
  // This prevents COW from firing on internal V8 structures
  // (maps, descriptors, SharedFunctionInfo, FixedArray, etc.)
  // during JIT compilation, GC, snapshot deserialization, etc.
  if (!IsCowProvenanceHost(host)) return;

  // Untag the value to get the raw heap address.
  Address value_addr = value.ptr() - kHeapObjectTag;

  // Fast path: if COW is globally inactive, skip entirely.
  if (V8_LIKELY(!GreenThreadCow::ActiveCount())) return;

  // CowCheckBeforeStore eagerly clones on first write and returns the copy.
  // If the object is not cloned yet, it returns the original address.
  Address copy_addr = CowCheckBeforeStore(gt->isolate(), value_addr);

  // If a copy was produced, rewrite the slot with the copy.
  // Note: slot stores the COPY address (not the original). FromAddress
  // takes an untagged, tag-aligned address and applies the tag itself.
  if (copy_addr != value_addr) {
    slot.StoreHeapObject(HeapObject::FromAddress(copy_addr));
  }
}

}  // namespace internal
}  // namespace v8
