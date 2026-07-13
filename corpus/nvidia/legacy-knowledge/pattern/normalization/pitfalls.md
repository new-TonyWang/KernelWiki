# Normalization Pattern -- Pitfalls

## P1: Accumulating Statistics in Half-Precision
- **Symptom**: LayerNorm output has large numerical errors compared to reference
- **Root cause**: Summing thousands of FP16 values overflows FP16 range (max ~65504) or loses mantissa precision
- **Example**: Feature dim 4096, mean value ~0.5: sum = 2048, but individual FP16 additions lose bits
- **Fix**: Always accumulate sum and sum_sq in FP32; only convert back to FP16 for final output

## P2: Numerical Instability in Naive Variance Computation
- **Symptom**: Variance is negative (mathematically impossible) for large values
- **Root cause**: Computing `var = E[x^2] - (E[x])^2`; catastrophic cancellation when both terms are large and close
- **When it matters**: Large feature values (before normalization converges in early training)
- **Fix**: Use Welford's online algorithm for numerically stable variance
- **Mitigation**: For most trained models where values are small, the naive formula is acceptable

## P3: Missing Epsilon in rsqrt
- **Symptom**: NaN output when variance is exactly zero
- **Root cause**: `rsqrtf(0.0f)` = inf, then inf * 0 = NaN
- **Fix**: Always use `rsqrtf(var + eps)` with eps = 1e-5f or 1e-6f
- **Note**: This applies to RMSNorm too: `rsqrtf(mean_sq + eps)`

## P4: Using sqrt + division Instead of rsqrt
- **Symptom**: Normalization kernel is compute-bottlenecked on the statistics
- **Root cause**: Computing `1.0f / sqrtf(var + eps)` uses two expensive operations
- **Fix**: Use `rsqrtf(var + eps)` -- single hardware instruction, ~3x faster

## P5: Not Broadcasting Statistics Efficiently
- **Symptom**: Repeated global memory reads for mean/variance in the normalization pass
- **Root cause**: After computing statistics, they are stored to global memory and re-read
- **Fix**: In a fused kernel, broadcast statistics via shuffle (within warp) or SMEM (across warps)
- **Fix**: Keep statistics in registers throughout the normalize + scale + bias step

## P6: Wrong Reduction Dimension
- **Symptom**: Output has wrong shape or values
- **Root cause**: LayerNorm reduces over the last (feature) dimension; BatchNorm over the batch dimension; confusing the two
- **Detection**: Check that reduction produces one scalar per row (LayerNorm) or per channel (BatchNorm)
- **Fix**: Map threads to the correct dimension: for LayerNorm, threads within a warp handle elements along the feature axis

## P7: Forgetting Affine Parameters (gamma, beta)
- **Symptom**: Normalization gives correct statistics but wrong output
- **Root cause**: Omitting the learnable scale (gamma) and shift (beta) after normalization
- **Fix**: Output = normalized * gamma + beta, where gamma and beta are per-feature vectors
- **For RMSNorm**: Output = normalized * gamma (no beta typically)

## P8: Block-Level Synchronization Missing for Large Feature Dim
- **Symptom**: Race condition in statistics computation when multiple warps handle one row
- **Root cause**: Warp leaders write partial sums to SMEM but `__syncthreads()` is missing before the final reduction
- **Fix**: Insert `__syncthreads()` between partial result write and the final reduction read

## P9: Inefficient Memory Access Pattern for Column-Wise Normalization
- **Symptom**: BatchNorm or instance-norm kernel much slower than LayerNorm
- **Root cause**: Reducing over the batch dimension requires strided access (non-coalesced)
- **Fix**: Transpose data to make reduction dimension contiguous
- **Fix**: Or use multi-pass approach with block-level atomics for the reduction

## P10: Two-Kernel Approach When Fusion Would Suffice
- **Symptom**: Separate kernels for (1) compute statistics and (2) normalize
- **Root cause**: Developer assumed statistics must be materialized in global memory
- **Fix**: Fuse into single kernel: compute statistics, then immediately normalize in same kernel
- **Benefit**: Avoids reading the entire input tensor twice from HBM
- **Exception**: Very large feature dimensions where single-block statistics + normalization exceeds SMEM

## P11: L1 cache hit rate dropped from 50% to 33% (one-pass doesn't re-read cached data from pass 1), and barrier stalls increased from 4 (discovered in verification)

**Symptom**: L1 cache hit rate dropped from 50% to 33% (one-pass doesn't re-read cached data from pass 1), and barrier stalls increased from 4.1% to 6.8% due to more complex shared-memory reductions accumulating two quantities simultaneously.
**Source**: Level 3 sandbox verification (2026-04-06)

## P12: Long scoreboard stalls increased from 32% to 37% — removing shared-memory synchronization points exposes more raw memory latency, and roofline efficiency paradoxically dropped (25 (discovered in verification)

**Symptom**: Long scoreboard stalls increased from 32% to 37% — removing shared-memory synchronization points exposes more raw memory latency, and roofline efficiency paradoxically dropped (25.2% → 21.0%) even though wall-clock time improved, because the kernel completes in fewer cycles but utilizes each cycle less.
**Source**: Level 3 sandbox verification (2026-04-06)

## P13: Block-level normalization introduces ~10 (discovered in verification)

**Symptom**: Block-level normalization introduces ~10.8% barrier stalls from __syncthreads() between the per-warp partial reduction and the final warp reduction; this overhead grows with warp count and can become the bottleneck if the feature dimension doesn't justify the extra warps.
**Source**: Level 3 sandbox verification (2026-04-06)

## P14: Warp-per-row lowered L1 hit rate from 48% to 12 (discovered in verification)

**Symptom**: Warp-per-row lowered L1 hit rate from 48% to 12.5% and increased DRAM bytes read by 37% (67MB→92MB) due to changed access patterns; the synchronization and instruction savings dominate, but for memory-bound follow-up optimizations this cache regression becomes the new ceiling.
**Source**: Level 3 sandbox verification (2026-04-06)

## P15: Register usage increased from 16 to 20 per thread, dropping occupancy_limit_registers from 16 to 10 blocks — on smaller problems this could hurt latency; the optimization also increased total instructions executed by 60% (7 (discovered in verification)

**Symptom**: Register usage increased from 16 to 20 per thread, dropping occupancy_limit_registers from 16 to 10 blocks — on smaller problems this could hurt latency; the optimization also increased total instructions executed by 60% (7.5M→12M) and active cycles by 34%, meaning the speedup comes entirely from eliminating memory round-trips, not from fewer operations.
**Source**: Level 3 sandbox verification (2026-04-06)

## P16: rsqrtf vs 1/sqrtf is a source-level distinction that vanishes after compilation on modern CUDA toolchains (sm_70+); the skill is correct in principle but provides no measurable benefit because the compiler already applies this transformation (discovered in verification)

**Symptom**: rsqrtf vs 1/sqrtf is a source-level distinction that vanishes after compilation on modern CUDA toolchains (sm_70+); the skill is correct in principle but provides no measurable benefit because the compiler already applies this transformation.
**Source**: Level 3 sandbox verification (2026-04-06)
