# Asynchronous Pipeline Primitives

Hopper (sm_90) and Ampere (sm_80) provide hardware-level primitives for asynchronous data movement and synchronization. These compose into multi-stage software pipelines that overlap memory transfers with computation.

## cp.async (Ampere, sm_80+)

Per-thread asynchronous copy from global memory to shared memory, bypassing registers.

```
cp.async.ca.shared.global [dst], [src], cp-size;   // cache at all levels
cp.async.cg.shared.global [dst], [src], 16;        // cache at L2 only
```

- **Copy sizes**: 4, 8, or 16 bytes per instruction
- **All threads participate**: unlike TMA, every thread issues its own cp.async for its portion of the data
- **No tensor descriptor needed**: operates on raw pointers
- **Optional src-size**: can copy fewer bytes than cp-size, zero-filling the remainder
- **Completion**: async-group mechanism (commit + wait) or mbarrier-based

### Async-Group Completion (cp.async)

```
cp.async.commit_group;         // Commits all pending cp.async ops into a group
cp.async.wait_group N;         // Waits until at most N groups are still in flight
cp.async.wait_all;             // Waits for all groups (equivalent to wait_group 0)
```

Groups always complete in commit order. There is no ordering between operations within a group.

## cp.async.bulk (Hopper, sm_90+)

Bulk asynchronous copy -- the foundation of TMA. A single thread issues a copy of an arbitrary-sized contiguous region or a tensor tile.

```
// 1D contiguous: global -> shared
cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [dst], [src], size, [mbar];

// Tensor: global -> shared (via tensor map)
cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
    [dst], [tensorMap, {c0, c1}], [mbar];

// shared -> global
cp.async.bulk.global.shared::cta.bulk_group [dst], [src], size;
```

Key differences from cp.async:
- **Single-thread issue** (not per-thread cooperative)
- **Arbitrary transfer size** (not limited to 4/8/16 bytes)
- **Tensor-aware** variants use tensor map descriptors for multi-dimensional copies
- **Multicast** support for broadcasting to cluster-wide shared memory
- **Completion** via mbarrier (for loads) or bulk async-group (for stores)

### Bulk Async-Group Completion (cp.async.bulk stores)

```
cp.async.bulk.commit_group;       // Commits pending bulk async store ops
cp.async.bulk.wait_group N;       // Waits until at most N bulk groups pending
```

Note: these are separate from the non-bulk `cp.async.commit_group` / `cp.async.wait_group`. The two async-group families are independent.

## Asynchronous Barrier (mbarrier)

Hardware-supported barrier object residing in shared memory. Tracks both thread arrivals and asynchronous transaction completions. Introduced with basic form in Ampere; extended with tx-count tracking in Hopper.

### Object Properties

- **Type**: 64-bit opaque object in shared memory, 8-byte aligned
- **Tracks per phase**:
  - Pending arrival count (decremented by `mbarrier.arrive`)
  - tx-count: pending async transaction bytes (decremented by `complete_tx` operations)
  - Current phase (flips when phase completes)
- **Phase completion**: current phase completes when BOTH pending arrivals = 0 AND tx-count = 0
- **Auto-reinitialize**: on phase completion, pending arrivals reset to expected count, phase advances
- **Max arrival count**: 2^20 - 1

### Lifecycle

```
mbarrier.init.b64 [addr], expected_count;    // Initialize with arrival count
// ... use across multiple phases ...
mbarrier.inval.b64 [addr];                   // Invalidate before reuse of memory
```

### Key Operations

| Operation | PTX | Effect |
|-----------|-----|--------|
| **Init** | `mbarrier.init.b64 [addr], count` | Sets expected arrival count, phase = 0, tx-count = 0 |
| **Expect TX** | `mbarrier.expect_tx.b64 [addr], txCount` | Increases tx-count by txCount (sets up tracking for async ops) |
| **Arrive** | `mbarrier.arrive.b64 state, [addr]` | Decrements pending arrivals by 1; returns token |
| **Arrive + Expect TX** | `mbarrier.arrive.expect_tx.b64 state, [addr], txCount` | Arrive AND set expected tx bytes in one operation |
| **Complete TX** | (implicit from `cp.async.bulk` completion) | Decrements tx-count by completed bytes |
| **Try Wait** | `mbarrier.try_wait.parity.b64 complete, [addr], phaseParity` | Blocking wait until phase parity flips |
| **Test Wait** | `mbarrier.test_wait.b64 complete, [addr], token` | Non-blocking check for phase completion |

### Shared Memory Scope

| Operation | `.shared::cta` | `.shared::cluster` (remote CTA) |
|-----------|---------------|--------------------------------|
| init, inval, expect_tx, test_wait, try_wait | Supported | NOT supported |
| arrive, arrive_drop | Supported | Supported (one-way cross-CTA sync) |

Threads can **arrive** on mbarriers in other CTAs within a cluster, but can only **wait** on mbarriers in their own CTA.

## Fence Instructions

Fences establish ordering between memory accesses across different execution proxies or scopes.

### fence.proxy.async (Hopper, sm_90+)

Synchronizes memory between the generic proxy (normal thread loads/stores) and the async proxy (TMA / cp.async.bulk).

```
fence.proxy.async;                  // Bidirectional, all state spaces
fence.proxy.async.shared::cta;     // Bidirectional, shared::cta only
fence.proxy.async.shared::cluster;  // Bidirectional, shared::cluster only
fence.proxy.async.global;           // Bidirectional, global only
```

**Critical usage**: before a TMA store, `fence.proxy.async.shared::cta` must be issued to ensure that prior SMEM writes by threads (generic proxy) are visible to the TMA engine (async proxy). Without this fence, the TMA store may read stale SMEM data.

**After TMA load**: not explicitly needed because the `cp.async.bulk` completion mechanism includes an implicit generic-async proxy fence. Once the mbarrier wait returns true, the data is visible in the generic proxy.

### fence.sc / fence.acq_rel (General memory ordering)

```
fence.acq_rel.scope;    // Lightweight acquire-release fence
fence.sc.scope;         // Sequential consistency fence (stronger, slower)
// scope = { .cta, .cluster, .gpu, .sys }
```

### membar (Legacy)

```
membar.cta;   // equivalent to fence.sc.cta on sm_70+
membar.gl;    // equivalent to fence.sc.gpu
membar.sys;   // equivalent to fence.sc.sys
```

### fence.proxy.tensormap::generic (Tensor map updates)

```
fence.proxy.tensormap::generic.release.gpu;                // Before updating tensormap
fence.proxy.tensormap::generic.acquire.gpu [tmap], 128;    // After update, before use
```

Used when dynamically updating tensor map objects (rare in typical kernels).

## Multi-Stage Pipeline Composition

These primitives compose into producer-consumer pipelines for high-performance kernels.

### Pipeline Pattern (Hopper with TMA)

```
Initialization:
  mbarrier full_barrier[N_STAGES]    // Tracks "buffer is full / data ready"
  mbarrier empty_barrier[N_STAGES]   // Tracks "buffer is empty / safe to write"

Producer (single TMA thread):
  for each stage:
    1. producer_acquire: wait on empty_barrier[i] (buffer free?)
    2. expect_tx on full_barrier[i] (set expected bytes)
    3. cp.async.bulk.tensor ... [full_barrier[i]]  (TMA load)
       // TMA hardware will complete_tx on full_barrier[i] when done

Consumer (all compute threads):
  for each stage:
    1. consumer_wait: try_wait on full_barrier[i] (data ready?)
    2. Compute (WGMMA, etc.)
    3. consumer_release: arrive on empty_barrier[i] (buffer freed)
```

### Pipeline Pattern (Ampere with cp.async)

```
Pre-fill first N-1 stages:
  for k_pipe = 0 to N-2:
    all threads: cp.async ... (load tile to stage k_pipe)
    cp.async.commit_group
  cp.async.wait_group<N-2>    // Wait for oldest group

Main loop:
  for each iteration:
    all threads: cp.async ... (load next tile)
    cp.async.commit_group
    // ... compute on current tile ...
    cp.async.wait_group<N-2>  // Wait for next tile to be ready
    __syncthreads()
```

### Key Design Decisions for Pipeline Configuration

| Parameter | Trade-off |
|-----------|-----------|
| **Number of stages** | More stages = more overlap opportunity, but more SMEM consumed for buffers |
| **Multistage vs warp-specialization** | Multistage: all warps do both produce + consume. Warp-spec: dedicated producer/consumer warps. Warp-spec preferred on Hopper when using TMA + WGMMA. |
| **Number of warpgroups** | More consumer warpgroups = more compute throughput, but higher register pressure. Typical: 1 producer + 2-3 consumer warpgroups. |
| **Register budget** | With warp-spec, use `setmaxnreg` to give producers ~24-40 regs and consumers ~160-240 regs. Total must fit in 64K registers/SM (512 regs * 128 threads/warpgroup). |

### CUTLASS Pipeline Classes

| Class | Use Case |
|-------|----------|
| `PipelineTmaAsync<Stages>` | TMA-based pipeline; `producer_commit` is no-op (TMA signals mbarrier directly) |
| `PipelineAsync<Stages>` | Generic async pipeline with explicit producer commit |
| `PipelineTransactionAsync<Stages>` | Transaction-count-based pipeline |

Pipeline state is tracked per-thread via `PipelineState<Stages>`:
- `index()`: current buffer stage (0 to Stages-1, wrapping)
- `phase()`: current phase bit (0 or 1, flips on wrap-around)
- `operator++`: advances index mod Stages, flips phase when wrapping

### Synchronization Correctness Checklist

1. **mbarrier init visibility**: after init, use `fence.mbarrier_init.release.cluster` + `cluster_sync()` before any arrive/wait operations
2. **Before TMA store**: issue `fence.proxy.async.shared::cta` after `__syncthreads()` to make SMEM writes visible to async proxy
3. **After TMA load**: mbarrier wait is sufficient (implicit proxy fence on completion)
4. **Phase tracking**: consumer and producer must independently track their phase bits; the phase flips each time the pipeline index wraps around
5. **Cluster sync for multicast**: when using multicast TMA, a `cluster_sync()` is needed to ensure no CTA exits while others still depend on shared multicast data
