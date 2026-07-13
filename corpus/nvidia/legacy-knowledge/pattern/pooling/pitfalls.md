# Pooling Pattern -- Pitfalls

## P1: Non-Coalesced Memory Access in Channel-Last Format
- **Symptom**: Pooling kernel achieves very low memory bandwidth
- **Root cause**: NCHW layout means spatial neighbors are contiguous, but pooling across channels requires strided access
- **Fix**: Use NHWC layout where channel dimension is contiguous; spatial pooling then operates on strided but predictable patterns
- **Fix**: Or use SMEM tiling to stage coalesced loads before random access

## P2: Redundant Global Memory Loads for Overlapping Windows
- **Symptom**: Pooling with stride=1 and pool_size=3 is much slower than expected
- **Root cause**: Each output element independently loads its entire 3x3 window from global memory; 9 loads per output element with 8/9 being redundant (shared with neighbors)
- **Fix**: Use SMEM tiling to load input tile once, then all threads read from SMEM
- **Speedup**: Approximately pool_size^2 / 1 in global memory traffic reduction

## P3: Incorrect Boundary Handling for Average Pooling
- **Symptom**: Border output values are systematically too small
- **Root cause**: Dividing by full pool_h * pool_w even when some window elements are outside the tensor (zero-padded)
- **Example**: Corner element with 3x3 pool and no padding: only 1 valid element but dividing by 9
- **Fix**: Compute actual valid element count per output position; divide by that count

## P4: SMEM Halo Region Not Loaded
- **Symptom**: Incorrect output near tile boundaries when using SMEM tiling
- **Root cause**: Only the "core" tile is loaded to SMEM; the extra border elements (halo) needed by edge threads are missing
- **Fix**: SMEM tile must include halo: width = blockDim.x * stride + pool_w - stride (and similarly for height)
- **Fix**: Assign extra threads to load the halo region, or have boundary threads load additional elements

## P5: Forgetting __syncthreads After SMEM Load
- **Symptom**: Race condition producing incorrect pooling results at tile edges
- **Root cause**: Some threads start reading SMEM before all threads have finished loading
- **Fix**: Insert `__syncthreads()` between SMEM cooperative load and the pooling computation

## P6: Integer Division in Index Computation
- **Symptom**: Pooling kernel surprisingly slow despite simple computation
- **Root cause**: Computing `in_h = out_h * stride_h + kh` and then converting to linear index involves integer multiplication; but worse, computing `nc = idx / (H_out * W_out)` uses expensive integer division
- **Fix**: Precompute batch and channel indices from blockIdx.z; avoid division in hot loop
- **Fix**: For power-of-2 dimensions, use bitwise shift instead of division

## P7: Max Pooling Initializer Not -FLT_MAX
- **Symptom**: Max pooling incorrectly returns 0 for windows with all negative values
- **Root cause**: Accumulator initialized to 0 instead of -FLT_MAX
- **Fix**: Initialize `max_val = -FLT_MAX` (or `-INFINITY` for more permissive behavior)
- **Analogous**: Min pooling should initialize to +FLT_MAX

## P8: Global Pooling Implemented as Nested Loops Instead of Reduction
- **Symptom**: Global average pooling over large spatial dimensions (e.g., 224x224) is very slow
- **Root cause**: Single thread loops over all 50176 elements sequentially
- **Fix**: Use parallel reduction (warp shuffle + SMEM + block reduction) -- see pattern/reduction
- **Alternative**: Use cuDNN's pooling primitives which handle this efficiently

## P9: Not Using fmaxf/fminf (NaN Propagation Issue)
- **Symptom**: Max pooling produces NaN when any input is NaN
- **Root cause**: `a > b ? a : b` does not propagate NaN correctly on GPU
- **Fix**: Use `fmaxf(a, b)` which is NaN-safe (returns the non-NaN value, or NaN if both are NaN)
- **Note**: Behavior depends on the desired NaN semantics -- some frameworks propagate NaN intentionally

## P10: SMEM Tiling for Non-Overlapping Pooling (Wasted Effort)
- **Symptom**: SMEM tiling adds overhead without performance gain
- **Root cause**: When stride >= pool_size (no overlap), every input element is used by exactly one output element; no reuse opportunity
- **Fix**: For non-overlapping pooling, skip SMEM tiling; direct global memory loads are sufficient

## P11: Fewer executed instructions (9 (discovered in verification)

**Symptom**: Fewer executed instructions (9.2M → 8.4M) does not guarantee faster execution; increased memory stalls and write amplification (703KB → 1.6MB DRAM writes) can dominate, especially when L1 hit rate drops (73% → 68%).
**Source**: Level 3 sandbox verification (2026-04-06)

## P12: SMEM tiling introduces significant barrier stalls (13 (discovered in verification)

**Symptom**: SMEM tiling introduces significant barrier stalls (13.85%) that partially offset the memory latency savings; the occupancy limit from shared memory also dropped from 32 to 24 blocks per SM due to increased SMEM allocation (1KB→2.7KB).
**Source**: Level 3 sandbox verification (2026-04-06)

## P13: Vectorized loads bypass L1 cache entirely (hit rate dropped from 70 (discovered in verification)

**Symptom**: Vectorized loads bypass L1 cache entirely (hit rate dropped from 70.6% to 0%), increasing long-scoreboard stalls from 20% to 31%; this tradeoff is net positive here but could hurt kernels that reuse input data across threads.
**Source**: Level 3 sandbox verification (2026-04-06)

## P14: L1 and L2 cache hit rates dropped dramatically (L1: 87 (discovered in verification)

**Symptom**: L1 and L2 cache hit rates dropped dramatically (L1: 87.5%→0.6%, L2: 33.5%→1.2%) which is actually correct behavior — the baseline was repeatedly re-reading the same data through cache due to poor parallelism, while the optimized version streams data once with full coalescing. However, roofline efficiency slightly decreased (94.6%→87.2%) suggesting the warp-level approach introduces some overhead from shuffle instructions (instructions doubled: 5M→9.6M).
**Source**: Level 3 sandbox verification (2026-04-06)

## P15: L2 cache hit rate dropped significantly (89 (discovered in verification)

**Symptom**: L2 cache hit rate dropped significantly (89.4% → 56.5%) due to the intermediate buffer between passes; for tensors that barely fit L2, the two-pass approach may lose some benefit to increased DRAM traffic.
**Source**: Level 3 sandbox verification (2026-04-06)

## P16: Register usage increased from 27 to 31 per thread (occupancy limiter stays at 8 warps/SM from registers), and long scoreboard stalls rose from 19 (discovered in verification)

**Symptom**: Register usage increased from 27 to 31 per thread (occupancy limiter stays at 8 warps/SM from registers), and long scoreboard stalls rose from 19.7% to 22.9% — the reduced arithmetic no longer hides memory latency as effectively.
**Source**: Level 3 sandbox verification (2026-04-06)

## P17: L1 hit rate dropped from 87 (discovered in verification)

**Symptom**: L1 hit rate dropped from 87.5% to ~0%, indicating the optimized kernel bypasses L1 entirely — acceptable here since coalescing is perfect and data is streamed, but could hurt if the kernel were modified to re-read spatial data.
**Source**: Level 3 sandbox verification (2026-04-06)
