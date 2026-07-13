# Attention Pattern -- Pitfalls

## P1: Numerical Instability in Online Softmax
- **Symptom**: NaN or inf values in attention output, especially with long sequences
- **Root cause**: Exponentials overflow when raw attention scores are large
- **Fix**: Always subtract the running max before exp: `exp(S_ij - m_i)`, never compute `exp(S_ij)` directly
- **Also**: Update running max BEFORE computing exp, not after

## P2: Forgetting Final Normalization After Tiled Accumulation
- **Symptom**: Output values are consistently too large
- **Root cause**: After all tiles are processed, O still needs division by the final sum `l`
- **Fix**: After the K/V loop completes: `O = O / l_final` per row

## P3: Incorrect Rescaling of Previous Accumulator
- **Symptom**: Subtle numerical errors that grow with sequence length
- **Root cause**: When max changes (m_new > m_prev), the existing O accumulator must be rescaled by `exp(m_prev - m_new)` but this step is omitted or applied incorrectly
- **Fix**: `O = exp(m_prev - m_new) * O_prev + exp(S_tile - m_new) * V_tile`
- **Note**: Both O_prev AND the new exp(S)*V must use the SAME m_new as reference

## P4: Exponential Bottleneck Ignored on Hopper/Blackwell
- **Symptom**: Attention kernel achieves only 35-50% of theoretical peak TFLOPS
- **Root cause**: SFU throughput for exp is 256x slower than tensor core throughput
- **Impact**: For head_dim=128, exp takes ~50% of matmul time (FP16) or ~100% (FP8)
- **Fix**: Overlap softmax with GEMM via pingpong scheduling or intra-warpgroup pipelining
- **Fix**: On Blackwell, use software polynomial approximation of exp to supplement MUFU

## P5: Register Pressure from Multiple Accumulators
- **Symptom**: Performance cliff or spilling in fused attention kernel
- **Root cause**: Attention needs accumulators for S (QK^T), O (PV), plus softmax statistics -- all simultaneously in registers
- **Impact**: Without register reallocation, FP32 accumulators cause massive spilling
- **Fix**: Use warp-specialization with setmaxnreg to give consumer warps 200+ registers
- **Fix**: On Blackwell, use TMEM for accumulators (256KB per SM, separate from register file)

## P6: Causal Mask Creates Load Imbalance
- **Symptom**: Attention with causal mask is slower per-token than bidirectional attention
- **Root cause**: Triangular mask means tiles in upper-right are skipped; CTAs assigned those tiles finish early
- **Fix**: Use persistent kernel with tile scheduler that accounts for causal mask (skip empty tiles)
- **Fix**: FlashAttention-4's tile scheduler distributes non-trivial tiles evenly across SMs

## P7: FP8 Quantization Error from Activation Outliers
- **Symptom**: FP8 attention output significantly diverges from FP16 reference
- **Root cause**: Activation outliers (0.1% of values) have much larger magnitude, causing loss of precision for remaining values after quantization
- **Fix**: Apply Hadamard transform (random orthogonal rotation) to Q and K before FP8 quantization
- **Result**: 2.6x error reduction; Hadamard is O(d log d) and memory-bandwidth bound, negligible cost

## P8: SMEM Bandwidth Bottleneck in Backward Pass
- **Symptom**: Backward attention achieves lower utilization than forward despite 2.5x more FLOPs
- **Root cause**: 5 MMAs share SMEM operand traffic; total SMEM bytes/cycle exceeds SM bandwidth
- **Example on B200**: Forward needs 768 SMEM cycles, backward needs 3328 SMEM cycles (for 2560 compute cycles)
- **Fix**: Store intermediates (P^T, dS^T) in TMEM instead of SMEM
- **Fix**: 2-CTA MMA mode halves per-CTA SMEM traffic for operand B

## P9: Warpgroup Synchronization Errors in Pingpong Schedule
- **Symptom**: Incorrect softmax results due to data race between warpgroups
- **Root cause**: Pingpong scheduling requires explicit `bar.sync` between warpgroups; missing barriers allow one warpgroup to read softmax statistics being modified by another
- **Fix**: Use named barriers to ensure warpgroup 1 completes GEMM before warpgroup 2 starts its softmax on the same data

## P10: MQA/GQA Head Packing Not Exploited
- **Symptom**: MQA/GQA attention slower than expected despite fewer KV heads
- **Root cause**: Naive implementation processes each Q head separately even though multiple Q heads share the same KV head
- **Fix**: Pack multiple Q heads into the same CTA, sharing KV loads across Q heads
- **Benefit**: Reduces global memory bandwidth for KV, improves arithmetic intensity

## P11: Persistent kernel with tile scheduling added significant overhead (short scoreboard stalls jumped from 0 (discovered in verification)

**Symptom**: Persistent kernel with tile scheduling added significant overhead (short scoreboard stalls jumped from 0.75% to 21.13%) that negated any benefit from skipping masked tiles or improving data reuse; the baseline naive kernel was already at 97.8% memory roofline, leaving almost no room for the optimization to help.
**Source**: Level 3 sandbox verification (2026-04-05)

## P12: Increasing shared memory from 1KB to 5KB per block dropped the shared-memory occupancy limit from 8 to 3 blocks/SM, and L2 hit rate degraded from 59% to 44% — the reduced occupancy may partially offset the DRAM write savings by reducing latency hiding (discovered in verification)

**Symptom**: Increasing shared memory from 1KB to 5KB per block dropped the shared-memory occupancy limit from 8 to 3 blocks/SM, and L2 hit rate degraded from 59% to 44% — the reduced occupancy may partially offset the DRAM write savings by reducing latency hiding
**Source**: Level 3 sandbox verification (2026-04-05)
