# Pipeline Overlap -- Skills

## When to Apply
- GEMM mainloop where copy and compute latencies are significant
- Any kernel that iterates over a K-dimension loading tiles and computing MMA
- "Feeding the beast" -- tensor cores are fast, memory is slow; pipeline hides memory latency

## Skill 1: Double/Multi-Buffering Design
- Allocate N copies of operand tiles in SMEM (N = number of pipeline stages)
- While stage i is being consumed by MMA, stage (i+1) % N is being filled by TMA/cp.async
- More stages = more overlap but more SMEM consumed
- Minimum: 2 stages (double buffering); typical Hopper: 2-4 stages

### SMEM budget per stage:
```
bytes_per_stage = (bM * bK + bN * bK) * sizeof(element)
total_smem = bytes_per_stage * num_stages + barrier_storage
```

## Skill 2: Hopper Pipeline with TMA and mbarrier
- Use `PipelineTmaAsync<N>` (CUTLASS) or manual mbarrier management
- Each stage has a `full_barrier` (producer signals data ready) and `empty_barrier` (consumer signals buffer free)
- mbarrier uses transaction-count completion: TMA automatically decrements tx-count on arrival
- `producer_acquire`: wait for empty_barrier (buffer available to write)
- `producer_get_barrier`: get mbarrier pointer to pass to TMA copy atom
- `consumer_wait`: wait for full_barrier (data ready to read)
- `consumer_release`: signal empty_barrier (buffer can be reused)

### Transaction bytes setup:
```
transaction_bytes = sizeof(tile_A_per_stage) + sizeof(tile_B_per_stage)
// Set via PipelineTmaAsync params.transaction_bytes
// Or manually: mbarrier.arrive.expect_tx(transaction_bytes)
```

## Skill 3: Ampere Pipeline with cp.async
- Use `cp.async.ca.shared.global` or `cp.async.cg.shared.global` for GMEM-to-SMEM
- Group copies into commit groups with `cp_async_fence()`
- Wait for completion: `cp_async_wait<N>()` waits until at most N groups still in flight
- Pre-fetch: before main loop, issue async copies for first N-1 stages

### Ampere pipeline skeleton:
```cuda
// Pre-fetch first K_PIPE_MAX-1 stages
for (int p = 0; p < K_PIPE_MAX-1; p++) {
    cp_async_copy(A_tile[p], B_tile[p]);
    cp_async_fence();
}
cp_async_wait<K_PIPE_MAX-2>();
__syncthreads();

// Main loop
for (int k = 0; k < k_tiles; k++) {
    // Issue async copy for next stage
    cp_async_copy(A_tile[next_stage], B_tile[next_stage]);
    cp_async_fence();

    // Compute MMA on current stage
    mma_compute(current_stage);

    // Wait for next stage data
    cp_async_wait<K_PIPE_MAX-2>();
    __syncthreads();

    advance_stages();
}
```

## Skill 4: Inner Pipeline (SMEM-to-RMEM) on Ampere
- WGMMA (Hopper) sources operand B directly from SMEM, but Ampere MMA needs registers
- Software-pipeline the SMEM-to-RMEM loads: load fragment[k+1] while computing on fragment[k]
- This hides SMEM load latency within MMA compute
- Effectively two nested pipelines: outer (GMEM->SMEM) and inner (SMEM->RMEM)

## Skill 5: Intra-Warpgroup Overlap of MMA and Non-MMA
- WGMMA is async: after `warpgroup_commit_batch()`, the warp can do other work
- Insert non-MMA operations (softmax, elementwise) between commit and wait
- `warpgroup_wait<N>()`: wait until at most N committed groups still in flight
- FlashAttention-3 technique: interleave softmax computation between GEMM1 and GEMM2

### Pattern for async overlap:
```
warpgroup_arrive();
gemm(tiled_mma, A, B, C);          // Issues async WGMMA
warpgroup_commit_batch();
// Do non-MMA work here while WGMMA executes
softmax_step(...);
warpgroup_wait<0>();                 // Now wait for WGMMA to complete
```

## Skill 6: Pipeline State Management
- `PipelineState<N>` tracks: index (0 to N-1, mod N) and phase (0 or 1)
- Increment (`++state`): index advances mod N; phase flips when index wraps to 0
- Producer and consumer each maintain their own state, offset by the pipeline depth
- Separate read state and write state: `smem_pipe_write` leads `smem_pipe_read` by the pre-fetched stages

## Skill 7: Proxy Fence for TMA-WGMMA Correctness
- TMA writes through async proxy; WGMMA reads through generic proxy
- `fence.proxy.async` required to make TMA writes visible to WGMMA
- In CUTLASS, `PipelineTmaAsync::consumer_wait` handles this internally
- For manual implementations: insert fence after mbarrier wait, before WGMMA issue

## Cross-References
- pattern/gemm/tiling-strategy -- tile dimensions determine per-stage SMEM
- pattern/gemm/warp-specialization -- WS kernel splits producer/consumer into different warps
- optimization/memory/data-prefetch -- TMA and cp.async instruction details
- optimization/synchronization/barrier-optimization -- mbarrier mechanics
