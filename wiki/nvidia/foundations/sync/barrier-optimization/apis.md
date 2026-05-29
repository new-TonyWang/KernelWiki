---
title: Barrier Optimization APIs
status: draft
source:
- path: spec
  anchor: Reference
apis:
- func_name: __syncthreads
  namespace: cuda-runtime
  kind: block-barrier
  signature: void __syncthreads();
  notes: 'Block-level barrier; backed by PTX bar.sync. BP §12.1.3 throughput: 16 ops/clk
    on sm_7.x / sm_8.x. Per-call cost dominated by idle-until-slowest-thread stall.'
- func_name: __syncthreads_count
  namespace: cuda-runtime
  kind: block-barrier-reduction
  signature: int __syncthreads_count(int predicate);
  notes: Block barrier + popcount of predicate across block. Useful for collective
    decisions without an extra reduction pass.
- func_name: __syncthreads_and
  namespace: cuda-runtime
  kind: block-barrier-reduction
  signature: int __syncthreads_and(int predicate);
  notes: Block barrier + AND-reduce of predicate.
- func_name: __syncthreads_or
  namespace: cuda-runtime
  kind: block-barrier-reduction
  signature: int __syncthreads_or(int predicate);
  notes: Block barrier + OR-reduce of predicate.
- func_name: __syncwarp
  namespace: cuda-runtime
  kind: warp-barrier
  signature: void __syncwarp(unsigned mask = 0xFFFFFFFFu);
  notes: Warp-level barrier (or subset via mask). Cheapest barrier; required before
    warp-synchronous code on sm_70+ (ITS).
- func_name: cuda::barrier
  namespace: cuda
  kind: async-barrier
  signature: template<thread_scope S> class cuda::barrier;
  notes: 'libcu++ async barrier with arrive/wait split. Scopes: block, cluster, device,
    system. Hardware-accelerated on sm_80+ for block/cluster scope.'
- func_name: cuda::barrier::arrive
  namespace: cuda
  kind: async-barrier-method
  signature: arrival_token arrive(ptrdiff_t update = 1);
  notes: Signal arrival, return token for the current phase. Does not block.
- func_name: cuda::barrier::wait
  namespace: cuda
  kind: async-barrier-method
  signature: void wait(arrival_token&& token);
  notes: Block until the phase identified by token completes. Token must be from current
    or immediately previous phase (pitfall P2).
- func_name: cuda::barrier::init
  namespace: cuda
  kind: async-barrier-init
  signature: friend void init(barrier*, ptrdiff_t expected);
  notes: One-time initialization; must be called from exactly one thread and followed
    by block.sync() before any arrive.
- func_name: cuda::device::barrier_native_handle
  namespace: cuda::device
  kind: async-barrier-ptx-bridge
  signature: uint64_t* barrier_native_handle(cuda::barrier<S>& b);
  notes: Get the underlying PTX mbarrier handle so you can call cuda::ptx::mbarrier_*
    directly. Used for expect_tx and TMA integration.
- func_name: thread_block::sync
  namespace: cooperative_groups
  kind: cg-block-barrier
  signature: void cg::thread_block::sync();
  notes: Cooperative-groups block barrier; equivalent to __syncthreads at the SASS
    level.
- func_name: thread_block_tile::sync
  namespace: cooperative_groups
  kind: cg-tile-barrier
  signature: void cg::thread_block_tile<N>::sync();
  notes: Sub-block barrier for a tile of N threads (N ∈ {1,2,4,8,16,32}). N=32 ==
    __syncwarp.
- func_name: bar.sync
  namespace: ptx
  kind: ptx-block-barrier
  signature: bar.sync a{, b};
  notes: PTX block barrier, 16 named barriers (a = 0..15). 'b' optional thread count
    for subset sync.
- func_name: bar.arrive
  namespace: ptx
  kind: ptx-block-arrive
  signature: bar.arrive a, b;
  notes: Arrive on a named barrier without waiting. Pairs with later bar.sync.
- func_name: barrier.cta.sync.aligned
  namespace: ptx
  kind: ptx-block-barrier-aligned
  notes: Aligned form; compiler uses when it can prove all warps participate fully.
    Has higher throughput than the non-aligned bar.sync.
- func_name: mbarrier.init
  namespace: ptx
  kind: ptx-mbarrier
  signature: mbarrier.init.shared::cta.b64 [mbar], expected_count;
  notes: Initialize a 64-bit mbarrier object in shared memory.
- func_name: mbarrier.arrive
  namespace: ptx
  kind: ptx-mbarrier
  signature: mbarrier.arrive.shared::cta.b64 token, [mbar] {, increment};
  notes: Signal arrival; returns a token identifying the current phase.
- func_name: mbarrier.arrive.expect_tx
  namespace: ptx
  kind: ptx-mbarrier-expect-tx
  signature: mbarrier.arrive.expect_tx.shared::cta.b64 token, [mbar], tx_count;
  notes: Arrive AND declare that `tx_count` bytes of async transactions will complete
    on this barrier. Enables hw-tracked TMA/cp.async synchronization.
- func_name: mbarrier.try_wait
  namespace: ptx
  kind: ptx-mbarrier
  signature: mbarrier.try_wait.shared::cta.b64 pred, [mbar], token, suspendTimeHint;
  notes: Blocking wait on the phase identified by token; returns 1 when phase completes.
- func_name: mbarrier.test_wait
  namespace: ptx
  kind: ptx-mbarrier
  signature: mbarrier.test_wait.shared::cta.b64 pred, [mbar], token;
  notes: Non-blocking check; returns 1 if phase already complete, else 0. Useful for
    interleaving.
- func_name: cp.async.mbarrier.arrive
  namespace: ptx
  kind: ptx-async-copy-mbarrier-bridge
  notes: Issued from a cp.async instruction target; increments the mbarrier transaction
    count when the async copy retires.
- func_name: barrier.cluster.arrive
  namespace: ptx
  kind: ptx-cluster-barrier
  notes: Cluster-scope barrier arrival (sm_90+). For DSMEM / multi-block kernels.
    See wiki/nvidia/hardware/thread-block-cluster/ (pending).
- func_name: barrier.cluster.wait
  namespace: ptx
  kind: ptx-cluster-barrier
  notes: Cluster-scope barrier wait (sm_90+).
- func_name: __nanosleep
  namespace: cuda-runtime
  kind: spin-wait-backoff
  signature: void __nanosleep(unsigned int ns);
  notes: Suspend the thread for ~ns nanoseconds. Used in spin-wait loops to reduce
    SM resource pressure. Legacy pitfall P8 documents the no-contention case where
    this adds minor overhead without benefit.
id: api-barrier-optimization-ref
type: api-definition
vendor: nvidia
func_name: Barrier Optimization APIs
namespace: runtime
header: cuda_runtime.h
signature: See documentation
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L3647-L3704
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/parallel-thread-execution/cuda_parallel-thread-execution_index.html.md
  anchor: L19217-L19482
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/parallel-thread-execution/cuda_parallel-thread-execution_index.html.md
  anchor: L20580-L21555
---
## Core APIs

| API / Instruction | Layer | Purpose |
|-------------------|-------|---------|
| `__syncthreads()` | Runtime | Block-level barrier (bar.sync backing) |
| `__syncwarp(mask)` | Runtime | Warp-level barrier (cheapest) |
| `cuda::barrier<thread_scope>` | libcu++ | Async barrier with arrive/wait split |
| `cuda::barrier::arrive()` / `.wait(token)` | libcu++ | Arrival and wait separated |
| `cg::thread_block_tile<N>::sync()` | coop-groups | Sub-block barrier (N threads) |
| `__syncthreads_count/and/or(pred)` | Runtime | Block barrier + collective reduction |

## PTX-level

| Instruction | Purpose |
|-------------|---------|
| `bar.sync a{, b}` / `barrier.cta.sync.aligned` | CTA barrier, 16 named slots |
| `bar.arrive a, b` | Non-blocking arrive on named barrier |
| `mbarrier.init` | Initialize shared-memory mbarrier |
| `mbarrier.arrive` / `.arrive.expect_tx` | Signal arrival / with tx-count tracking |
| `mbarrier.try_wait` / `.test_wait` | Blocking / non-blocking phase-complete check |
| `cp.async.mbarrier.arrive` | Async-copy-retire bridge to mbarrier |
| `barrier.cluster.arrive` / `.wait` | Cluster-scope barrier (sm_90+) |

## Cross-references

- **`memory-ordering` skill**: the fence/visibility side of sync — what writes are guaranteed visible after the barrier.
- **`warp-primitives` skill**: warp shuffles are `_sync` variants with an embedded barrier; usually removes the need for explicit `__syncwarp`.
- **`async-copy` skill**: `cp.async` / TMA + `mbarrier.arrive.expect_tx` is the canonical producer-consumer shape on sm_80+.
- **`warp-divergence` skill** (pitfall P1): re-converge divergent warps with `__syncwarp` before `mbarrier.arrive` to avoid per-lane arrivals.
- **`wiki/nvidia/hardware/thread-block-cluster/`** (pending bucket F): cluster-scope barriers belong there.
