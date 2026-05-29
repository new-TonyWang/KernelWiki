# Reduction Pattern -- Pitfalls

## P1: Warp Divergence in Interleaved Addressing
- **Symptom**: Performance only ~50% of theoretical for naive tree reduction
- **Root cause**: Interleaved addressing (stride 1, 2, 4, ...) causes half the warp to be inactive at each step, wasting SIMT lanes
- **Fix**: Use sequential addressing or (better) warp shuffle, which has no divergence
- **Note**: This is a classic GPU reduction anti-pattern from early CUDA tutorials

## P2: Missing __syncthreads Between SMEM Write and Read
- **Symptom**: Incorrect reduction results that vary between runs
- **Root cause**: Warp leaders write partial results to SMEM; without barrier, other threads may read stale values
- **Fix**: Always insert `__syncthreads()` between the SMEM write (warp leaders) and the SMEM read (first warp)

## P3: Reading Beyond Active Warp Count in Final Reduction
- **Symptom**: Incorrect results when block size is not a power of 2 or when num_warps < 32
- **Root cause**: Final warp reads from `warp_results[threadIdx.x]` for indices beyond actual number of warps
- **Fix**: Initialize `warp_results` array to identity element (0 for sum, -inf for max, +inf for min)
- **Also**: Guard the final reduction: only threads with `threadIdx.x < num_warps` participate

## P4: Atomic Contention Destroys Throughput
- **Symptom**: Grid-level atomic reduction is far slower than expected
- **Root cause**: Thousands of blocks atomically updating the same address causes serialization
- **Example**: 1024 blocks doing atomicAdd to one address: effective throughput drops by >100x
- **Fix**: Use hierarchical reduction (block-level first, then grid-level with few atomics)
- **Fix**: For multi-element reductions, spread atomics across different addresses (one per output element)

## P5: FP32 Atomic Rounding Causes Non-Determinism
- **Symptom**: Reduction results differ between runs with same input
- **Root cause**: atomicAdd for floating-point is not associative; order of additions varies between runs
- **Fix**: Use two-pass reduction (deterministic) if reproducibility is required
- **Fix**: Or use integer atomics on scaled-integer representation, then convert back

## P6: Not Maximizing Memory Bandwidth in Bandwidth-Bound Reduction
- **Symptom**: Reduction kernel achieving <50% of peak memory bandwidth
- **Root cause**: Scalar loads (4 bytes at a time) instead of vectorized loads
- **Fix**: Use `float4` loads (16 bytes) or `int4` loads; requires 16-byte-aligned input
- **Also**: Ensure each thread processes multiple elements (grid-stride loop) to amortize launch overhead

## P7: Grid-Stride Loop with Incorrect Boundary Handling
- **Symptom**: Wrong results for input sizes not divisible by grid size
- **Root cause**: Last iteration loads elements beyond array bounds
- **Fix**: Guard with `if (idx < N)` inside the grid-stride loop
- **Note**: For vectorized loads, also check that the tail elements (N % vector_width) are handled

## P8: Reduction of Half-Precision Values Loses Accuracy
- **Symptom**: Large numerical error in FP16 reduction of many elements
- **Root cause**: FP16 has limited range and precision; accumulating thousands of values overflows or loses mantissa bits
- **Fix**: Accumulate in FP32 internally; only convert final result to FP16
- **Pattern**: Load as __half2, convert to float2, accumulate in float, convert back at the end

## P9: Bank Conflicts in SMEM Reduction
- **Symptom**: Block-level SMEM reduction slower than expected
- **Root cause**: If warp results are written to consecutive 4-byte addresses, bank conflicts occur during the read phase when multiple threads access the same bank
- **Fix**: For <= 32 partial results (one per warp), each result maps to a different bank -- no conflict
- **Note**: This is only a concern if the SMEM reduction layout is unusual (e.g., 2D partial results)

## P10: Not Using Hardware Warp Reduction When Available
- **Symptom**: Using 5-step shuffle tree for integer reduction when `redux.sync` would be faster
- **Root cause**: Unaware of SM80+ hardware reduction instruction
- **Fix**: For 32-bit integer add/min/max/and/or/xor: use PTX `redux.sync.op.s32` or compiler intrinsic
- **Limitation**: Not available for floating-point types; still need shuffle for FP reductions

## P11: At this tiny problem size (4KB read, ~6µs), the kernel is deeply underutilized (0 (discovered in verification)

**Symptom**: At this tiny problem size (4KB read, ~6µs), the kernel is deeply underutilized (0.6% memory SOL) so the 11.8% speedup reflects launch/scheduling overhead sensitivity more than sustained throughput gains; the benefit may look different at scale.
**Source**: Level 3 sandbox verification (2026-04-06)

## P12: Long scoreboard stalls increased from 47% to 60% — once shuffle eliminates compute/barrier overhead, memory latency becomes the dominant bottleneck, so further gains require addressing memory access patterns rather than reduction logic (discovered in verification)

**Symptom**: Long scoreboard stalls increased from 47% to 60% — once shuffle eliminates compute/barrier overhead, memory latency becomes the dominant bottleneck, so further gains require addressing memory access patterns rather than reduction logic.
**Source**: Level 3 sandbox verification (2026-04-06)

## P13: At tiny input sizes the kernel launch latency (~5-10us) dwarfs all compute, so warp-shuffle vs shared-memory reduction differences are unmeasurable; both versions are ~1 (discovered in verification)

**Symptom**: At tiny input sizes the kernel launch latency (~5-10us) dwarfs all compute, so warp-shuffle vs shared-memory reduction differences are unmeasurable; both versions are ~1.5-3x slower than even the PyTorch eager baseline.
**Source**: Level 3 sandbox verification (2026-04-06)

## P14: Vectorized load optimizations are meaningless on tiny inputs where kernel launch latency dwarfs actual compute/memory time; the skill can only be verified on large reductions (millions of elements) that actually saturate memory bandwidth (discovered in verification)

**Symptom**: Vectorized load optimizations are meaningless on tiny inputs where kernel launch latency dwarfs actual compute/memory time; the skill can only be verified on large reductions (millions of elements) that actually saturate memory bandwidth.
**Source**: Level 3 sandbox verification (2026-04-06)

## P15: L1 hit rate dropped from 87 (discovered in verification)

**Symptom**: L1 hit rate dropped from 87.5% to 0.17% after optimization — coalesced streaming access bypasses L1 entirely, so any kernel that mixes coalesced bulk reads with small reuse-heavy accesses in the same launch may see L1 contention tradeoffs.
**Source**: Level 3 sandbox verification (2026-04-06)
