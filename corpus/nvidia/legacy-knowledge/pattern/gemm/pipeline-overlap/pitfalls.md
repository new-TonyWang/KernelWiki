# Pipeline Overlap -- Pitfalls

## P1: Pipeline Deadlock from Incorrect Initial Phase
- **Symptom**: Kernel hangs on first iteration
- **Root cause**: Producer and consumer start with same phase, causing both to wait indefinitely
- **Detail**: Buffers start empty, so producer should see "empty" (phase 1) and not block; consumer should see "not full" (phase 0) and block
- **Fix**: Initialize producer state with `make_producer_start_state<Pipeline>()` which sets phase = 1
- **Fix**: Initialize consumer state with default constructor (phase = 0)

## P2: Transaction Bytes Mismatch Causes Premature/Late Completion
- **Symptom**: Consumer proceeds before data is ready (corruption) or waits forever (deadlock)
- **Root cause**: `mbarrier.arrive.expect_tx` set to wrong byte count
- **Must match**: Sum of all TMA copy bytes arriving at this stage's mbarrier
- **Common mistake**: Setting per-operand bytes instead of total (A + B) per stage
- **Fix**: `transaction_bytes = sizeof(sA_one_stage) + sizeof(sB_one_stage)`

## P3: Missing __syncthreads Before First Pipeline Iteration
- **Symptom**: Race condition on pipeline barrier initialization
- **Root cause**: Pipeline barriers stored in SMEM; all threads must see initialized values before use
- **Fix**: Insert `__syncthreads()` after pipeline construction, before mainloop

## P4: Too Many Stages Starves Occupancy
- **Symptom**: Kernel launches but with low occupancy, latency not hidden
- **Root cause**: N stages * per_stage_bytes exceeds SMEM budget for multi-CTA occupancy
- **Example**: 4 stages * 128KB/stage = 512KB -- impossible on any current GPU (H100 max 228KB)
- **Fix**: Profile with 2-4 stages; usually 2 stages is optimal on Hopper with WS
- **Note**: On Hopper WS, 1 CTA per SM is the norm; extra stages still help hide latency within that CTA

## P5: Ampere cp.async Pipeline Without Proper Fence
- **Symptom**: Partial or incorrect data in SMEM when MMA reads it
- **Root cause**: `cp.async` commits are grouped; must fence between groups and wait for correct group
- **Fix**: Issue `cp_async_fence()` after each group; `cp_async_wait<N>()` before consuming data
- **Note**: `cp_async_wait<N>` means wait until at most N groups remain in-flight (not wait for N-th group)

## P6: Forgetting wgmma.fence Before First WGMMA
- **Symptom**: Incorrect results from first WGMMA in a pipeline stage
- **Root cause**: WGMMA requires a fence to establish happens-before with the data write
- **Fix**: Issue `wgmma.fence.sync.aligned` before the first WGMMA call for each stage
- **CUTLASS**: `cute::warpgroup_arrive()` internally handles the fence

## P7: Using Wrong Stage Index for SMEM Access
- **Symptom**: MMA computes on wrong data; results are shifted by one iteration
- **Root cause**: Using write_stage index for read or vice versa
- **Fix**: Always use `smem_pipe_read.index()` for consumer SMEM access and `smem_pipe_write.index()` for producer SMEM access
- **Debug**: Print stage indices to verify they alternate correctly

## P8: Consumer Release Before WGMMA Completion
- **Symptom**: Producer overwrites SMEM buffer while WGMMA is still reading from it
- **Root cause**: `consumer_release` signals that the buffer is free, but WGMMA is async
- **Fix**: Call `warpgroup_wait<0>()` before `consumer_release` to ensure all WGMMA ops have completed

## P9: Prologue/Epilogue Missing from Pipeline
- **Symptom**: First or last tile produces incorrect results
- **Root cause**: Main loop assumes steady-state pipeline; first load and last compute need special handling
- **Fix**: Pre-issue the first TMA load before entering the main loop; handle the last MMA after the loop exits
- **Pattern**: The main loop runs k_tile_count - 1 iterations; one extra compute step follows

## P10: cp (discovered in verification)

**Symptom**: cp.async pipeline on an already memory-saturated kernel can regress performance: the extra instructions (fence/wait/stage bookkeeping) and barrier synchronization cost more than the latency hiding saves; also global load sector utilization dropped from 100% to 0%, suggesting the generated cp.async code may have broken coalesced access patterns.
**Source**: Level 3 sandbox verification (2026-04-05)

## P11: On Ampere with already-high register pressure (128 regs), adding inner-loop double-buffering can push past the spill threshold, turning a latency-hiding optimization into a catastrophic regression due to register spills hitting DRAM (discovered in verification)

**Symptom**: On Ampere with already-high register pressure (128 regs), adding inner-loop double-buffering can push past the spill threshold, turning a latency-hiding optimization into a catastrophic regression due to register spills hitting DRAM.
**Source**: Level 3 sandbox verification (2026-04-05)

## P12: Doubling shared memory for pipeline buffers can shift the bottleneck from global memory latency to shared memory bank conflicts, increasing short scoreboard stalls by ~80% and executing 26% more instructions while delivering only 2% speedup (discovered in verification)

**Symptom**: Doubling shared memory for pipeline buffers can shift the bottleneck from global memory latency to shared memory bank conflicts, increasing short scoreboard stalls by ~80% and executing 26% more instructions while delivering only 2% speedup
**Source**: Level 3 sandbox verification (2026-04-06)

## P13: Inserting fence (discovered in verification)

**Symptom**: Inserting fence.proxy.async in a kernel that doesn't actually use TMA+WGMMA adds measurable overhead (~35% regression); the fence is only meaningful when both async TMA loads and WGMMA consumers are present.
**Source**: Level 3 sandbox verification (2026-04-06)
