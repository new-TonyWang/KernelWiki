# Warp Specialization -- Skills

## When to Apply
- Hopper+ GEMM kernels where TMA is used for data movement
- Fused kernels with multiple accumulators (e.g., FlashAttention) that are register-intensive
- Any pipelined kernel where producer and consumer have asymmetric resource needs

## Skill 1: Producer/Consumer Warpgroup Split
- Assign warpgroup 0 as producer (issues TMA loads), remaining warpgroups as consumers (run WGMMA)
- Producer identification: `warp_group_idx == 0`
- Within producer warpgroup, only 1 thread issues TMA: `warp_idx_in_warpgroup == 0 && elect_one_sync()`
- Use named barriers or `PipelineTmaAsync` for synchronization

### Code pattern (raw CUDA):
```cuda
int warp_group_idx = threadIdx.x / 128;
if (warp_group_idx == 0) {
    // Producer: issue TMA loads
    if ((threadIdx.x / 32) % 4 == 0) {
        int elected = __shfl_sync(0xffffffff, threadIdx.x, 0) == threadIdx.x;
        if (elected) {
            // Issue TMA via inline PTX or CUTLASS copy atom
        }
    }
} else {
    // Consumer: run WGMMA
}
```

## Skill 2: Dynamic Register Reallocation with setmaxnreg
- PTX: `setmaxnreg.dec.sync.aligned.u32 <count>` (producer deallocates)
- PTX: `setmaxnreg.inc.sync.aligned.u32 <count>` (consumer allocates)
- CUTLASS wrapper: `cutlass::arch::warpgroup_reg_dealloc<N>()` / `warpgroup_reg_alloc<N>()`
- Must be called by all threads in the warpgroup simultaneously
- Valid range: 24 to 256, multiples of 8

### Register Budget Recipes:
| Config | Producer | Consumer WG1 | Consumer WG2 | Consumer WG3 | Total (x128 threads) |
|--------|----------|-------------|-------------|-------------|---------------------|
| 1P+1C  | 24       | 232         | -           | -           | 256*128 = 32K       |
| 1P+2C  | 24       | 240         | 240         | -           | 504*128 = 64K       |
| 1P+3C  | 32       | 160         | 160         | 160         | 512*128 = 64K       |

- Total must not exceed 64K registers per SM (Hopper: 65536 = 512 * 128)
- Exceeding this causes a runtime crash

## Skill 3: Pipeline Synchronization in Warp-Specialized Design
- Use `PipelineTmaAsync` (CUTLASS) or manual mbarrier management
- Producer path: `producer_acquire` -> TMA copy -> `producer_commit` (implicit for TMA) -> advance state
- Consumer path: `consumer_wait` -> WGMMA -> `consumer_release` -> advance state
- The pipeline has N stages with `full_barrier[N]` (producer->consumer) and `empty_barrier[N]` (consumer->producer)

## Skill 4: Choosing Between Multistage and Warp-Specialization
- **Prefer multistage** when:
  - Register pressure is low (FP16 accumulation, small tiles)
  - Ampere architecture (no setmaxnreg, cp.async enables async overlap anyway)
  - Simpler code desired
- **Prefer warp-specialization** when:
  - Register pressure is high (FP32 accumulation, large tiles, multiple accumulators)
  - Hopper+ architecture with TMA (register-light producer)
  - Fused kernels needing extra registers for non-MMA work (e.g., softmax in FlashAttention)
  - Best-in-class performance on Hopper (CUTLASS's fastest kernels use WS)

## Skill 5: Pingpong Scheduling for Multi-Consumer Overlap
- With 2 consumer warpgroups, schedule them in alternation
- Warpgroup 1 does GEMM while warpgroup 2 does non-GEMM work (softmax, normalization), then swap
- Use `bar.sync` between warpgroups to enforce ordering
- Enables overlapping compute (tensor cores) with non-MMA work (SFU, CUDA cores)
- Critical for FlashAttention-3+ performance: 570 -> 620 TFLOPS with pingpong scheduling

## Skill 6: Elect-One Pattern for TMA Issue
- Only one thread should issue each TMA operation
- PTX: `elect.sync` returns 1 for exactly one active thread
- CUTLASS: `cute::elect_one_sync()` wraps this
- Remaining threads in the producer warpgroup simply wait at barriers

## Cross-References
- optimization/compute/warp-specialization -- general warp-specialization technique
- pattern/gemm/pipeline-overlap -- pipeline mechanics details
- optimization/memory/register-pressure -- register budget management
- pattern/attention -- pingpong scheduling for FlashAttention
