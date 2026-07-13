# Warp Specialization -- Pitfalls

## P1: Register Reallocation Sum Exceeds SM Limit
- **Symptom**: Runtime crash (illegal instruction or similar)
- **Root cause**: Sum of (producer_regs + consumer1_regs + consumer2_regs + ...) * 128 threads > 65536
- **Example**: 3 warpgroups with 32/240/240 = 512 * 128 = 65536 -- OK. But 32/248/248 = 528 * 128 = 67584 -- crash
- **Fix**: Verify total <= 512 per-thread registers across all warpgroups (on Hopper)

## P2: setmaxnreg Not Called by All Threads in Warpgroup
- **Symptom**: Hang or undefined behavior
- **Root cause**: `setmaxnreg` is a warpgroup-collective instruction; all 128 threads must execute it
- **Fix**: Place setmaxnreg call before the producer/consumer branch, or ensure all threads within each branch call it

## P3: Producer Warps Other Than Leader Still Consuming Resources
- **Symptom**: Lower-than-expected performance in WS kernel
- **Root cause**: In a 4-warp producer warpgroup, only 1 thread in 1 warp issues TMA; the other 127 threads idle but still consume resources
- **Mitigation**: The 24-40 register allocation for producers minimizes this waste
- **Mitigation**: On Blackwell, fully async UMMA is single-thread launched, making this even more efficient

## P4: Forgetting proxy Fence Between TMA and WGMMA
- **Symptom**: Consumer reads stale or partially written data from SMEM
- **Root cause**: TMA writes via async proxy; WGMMA reads via generic proxy; fence needed to make TMA writes visible
- **Fix**: Insert `fence.proxy.async` (PTX) or use CUTLASS pipeline which handles this internally
- **Also**: `wgmma.fence.sync.aligned` required before first WGMMA call in each stage

## P5: Not Initializing Pipeline Phase Correctly
- **Symptom**: Deadlock on first iteration -- producer waits for empty_barrier that never signals
- **Root cause**: Producer start state should have opposite phase from initial barrier phase
- **Fix**: Use `cutlass::make_producer_start_state<Pipeline>()` which sets phase = 1 (opposite of barrier init 0)

## P6: Epilogue Placed in Wrong Warpgroup
- **Symptom**: Output is zero or garbage values
- **Root cause**: Accumulator `tCrC` lives in consumer warp registers; writing it out from producer warps accesses unrelated register content
- **Fix**: Always place epilogue code inside the consumer warpgroup code path

## P7: Register Spilling Without setmaxnreg
- **Symptom**: Kernel compiles and runs but at drastically reduced performance (~21 TFLOPS vs ~480 TFLOPS)
- **Root cause**: Without register reallocation, compiler assigns equal registers to all warps; FP32 accumulation with large tiles causes spilling
- **Detection**: Compile with `-Xptxas=--verbose` and check for "spill stores" / "spill loads"
- **Fix**: Enable `setmaxnreg` and verify zero spilling in compilation output
- **Note**: FP16 accumulation may not spill even without setmaxnreg (fewer registers needed)

## P8: Multiple TMA Threads Issuing to Same Barrier
- **Symptom**: Transaction bytes count inflated, barrier phase completes prematurely or late
- **Root cause**: More than one thread issuing TMA copies that arrive at the same mbarrier
- **Fix**: Exactly one elected thread per TMA operation; use `elect_one_sync()` guard

## P9: Mixing Up num_consumers in Pipeline Initialization
- **Symptom**: Pipeline synchronization off by factor of 2
- **Root cause**: `num_consumers` should count only consumer warpgroup threads (e.g., 128), not all CTA threads
- **Fix**: Set `params.num_consumers = num_consumer_warpgroups * 128`

## P10: Applying warp specialization without actual WGMMA/tensor-core consumers wastes a full warpgroup on producer duties while increasing barrier stall pressure; the pattern requires both TMA loads and WGMMA consumers to see any overlap benefit (discovered in verification)

**Symptom**: Applying warp specialization without actual WGMMA/tensor-core consumers wastes a full warpgroup on producer duties while increasing barrier stall pressure; the pattern requires both TMA loads and WGMMA consumers to see any overlap benefit.
**Source**: Level 3 sandbox verification (2026-04-06)

## P11: Both baseline and optimized kernels are ~8x slower than cuBLAS (340us), and neither uses tensor cores — the skill's advice is correct directionally but the custom kernel implementation has fundamental issues (no tensor core usage) that dominate absolute performance (discovered in verification)

**Symptom**: Both baseline and optimized kernels are ~8x slower than cuBLAS (340us), and neither uses tensor cores — the skill's advice is correct directionally but the custom kernel implementation has fundamental issues (no tensor core usage) that dominate absolute performance.
**Source**: Level 3 sandbox verification (2026-04-06)
