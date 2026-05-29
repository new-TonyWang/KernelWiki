# Attention Pattern -- Skills

## When to Apply
- Scaled dot-product attention: O = softmax(Q * K^T / sqrt(d)) * V
- Multi-head attention (MHA), multi-query attention (MQA), grouped-query attention (GQA)
- Any kernel that chains two GEMMs with softmax in between

## Core Insight: FlashAttention Tiling
Standard attention materializes the full NxN attention matrix in HBM (quadratic memory).
FlashAttention reorders the computation to work tile-by-tile, keeping intermediate results in SRAM:
- Tile over the K/V sequence dimension (outer loop) and Q sequence dimension (inner loop or parallelized)
- Use online softmax (track running max and sum) to compute exact softmax without full materialization
- Memory usage becomes O(N) instead of O(N^2)

## Skill 1: Online Softmax with Tiled Rescaling
- Maintain per-row running statistics: `m_prev` (running max) and `l_prev` (running sum of exp)
- For each new tile of S = Q * K^T:
  1. Compute `m_new = max(m_prev, rowmax(S_tile))`
  2. Rescale: `l_new = exp(m_prev - m_new) * l_prev + rowsum(exp(S_tile - m_new))`
  3. Update accumulator: `O = exp(m_prev - m_new) * O + exp(S_tile - m_new) * V_tile`
  4. Final normalization: `O = O / l_new`
- The rescaling step is the key innovation that avoids materializing the full attention matrix

### Conditional rescaling optimization (FlashAttention-4):
- Skip rescaling when `m_new - m_prev < threshold` (max didn't change much)
- In that case: `O += exp(S_tile - m_prev) * V_tile` (no O rescaling needed)
- Reduces non-MMA work on the critical path; decide at warp granularity to avoid divergence

## Skill 2: Overlapping GEMM and Softmax
- **Problem**: On H100, tensor cores do 989 TFLOPS FP16 but SFU (exp) does only 3.9 TFLOPS
- For head dim 128: exp can take ~50% of the time relative to matmul
- **Solution 1 (inter-warpgroup pingpong)**: 2 warpgroups alternate -- WG1 does GEMM while WG2 does softmax, then swap
  - Improvement: 570 -> 620 TFLOPS on H100 for FP16 attention
- **Solution 2 (intra-warpgroup pipelining)**: Within one warpgroup, issue WGMMA async, do softmax while waiting
  - Uses `warpgroup_commit_batch()` + softmax + `warpgroup_wait<0>()`
  - Improvement: 620 -> 640-660 TFLOPS, at the cost of higher register pressure
- **Solution 3 (software exp emulation, FlashAttention-4)**: On Blackwell, distribute exp across MUFU and FMA polynomial approximation
  - Cody-Waite range reduction: decompose 2^x = 2^n * 2^f
  - Polynomial approximation of 2^f via Horner's method on FMA units
  - Effective exp throughput nearly doubled

## Skill 3: Back-to-Back GEMM Fusion
- Attention requires S = Q * K^T then O = softmax(S) * V
- Fusing both GEMMs into one kernel avoids materializing S to HBM
- Challenge: GEMM1 output is in registers/TMEM; must convert to softmax input format
- WGMMA layout for accumulator C differs from softmax's preferred rowwise layout
- On Hopper: read accumulator from registers one row at a time for softmax
- On Blackwell: read from TMEM (one thread per row), simplifying the mapping

## Skill 4: FP8 Attention with Incoherent Processing
- FP8 doubles tensor core throughput (1978 TFLOPS on H100 vs 989 TFLOPS for FP16)
- Challenge: Activation outliers cause large quantization errors in FP8
- Solution: Multiply Q and K by random orthogonal matrix (Hadamard transform) to spread outliers
- Hadamard transform is O(d log d) per head, memory-bandwidth bound, fusible with rotary embedding
- Result: 2.6x smaller quantization error vs naive FP8

## Skill 5: Tile Size Selection for Attention
- Q tile (Br): typically 64 or 128 rows (matches WGMMA M dimension)
- KV tile (Bc): typically 64 or 128 columns (trades SMEM for parallelism)
- Head dimension (d): 64, 128, or 256 (larger d = more FLOPs per softmax overhead)
- SMEM budget: must hold Q_tile + K_tile + V_tile + O_tile + softmax statistics
- Register budget: accumulator for S (Br x Bc) + accumulator for O (Br x d) + softmax stats

## Skill 6: Persistent Kernel with Tile Scheduling for Attention
- Like GEMM, attention benefits from persistent kernels to avoid wave quantization
- Additional consideration: causal mask creates triangular workload (some tiles are empty)
- Tile scheduler must account for variable work per tile (skip fully masked tiles)
- FlashAttention-4 uses a specialized scheduler that balances work across SMs for causal attention

## Skill 7: Backward Pass Optimization
- Backward computes ~2.5x the FLOPs of forward (5 MMAs vs 2)
- Forward: S = QK^T, O = PV (2 GEMMs + softmax)
- Backward: dP, dS recomputation, dQ, dK, dV (5 GEMMs + elementwise)
- On Blackwell: backward is SMEM-bandwidth bottlenecked, not compute bottlenecked
- Key technique: store intermediates (P^T, dS^T) in TMEM to reduce SMEM traffic
- 2-CTA MMA mode halves SMEM traffic for operand B and reduces atomic reductions

## Architecture-Specific Notes
- **Ampere**: FlashAttention-2 with multistage cp.async pipeline; ~70% utilization achievable
- **Hopper**: FlashAttention-3 with warp-specialization + pingpong; ~75% utilization (740 TFLOPS FP16)
- **Blackwell**: FlashAttention-4 with UMMA + TMEM + software exp; ~71% utilization (1605 TFLOPS BF16)
  - Bottleneck shifts from compute to exp (forward) and SMEM bandwidth (backward)

## Cross-References
- pattern/gemm -- underlying GEMM optimization applies to the two matmuls
- pattern/gemm/warp-specialization -- pingpong scheduling pattern
- pattern/gemm/pipeline-overlap -- pipeline mechanics for attention mainloop
- optimization/compute/fast-math -- exp approximation techniques
- optimization/compute/half-precision-math -- FP8 quantization
