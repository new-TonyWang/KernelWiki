---
title: Normalization Pattern -- Skill Routing
pattern_class: cuda-core
op: normalization
status: draft
id: routing-normalization-ROUTING
type: operator-routing
vendor: nvidia
operator: normalization
architectures:
- sm90
- sm90a
languages:
- cuda-cpp
- triton
techniques:
- persistent-kernel
- pipeline-stages
- vectorized-loads
- cache-policy
- register-budgeting
- data-reuse
- kernel-fusion
- loop-unrolling
- shared-memory-optimization
- software-exp
kernel_types:
- fused-kernel
- gemm
- quantization
confidence: inferred
tags:
- persistent-kernel
- pipeline-stages
- vectorized-loads
- cache-policy
- register-budgeting
- data-reuse
- kernel-fusion
- loop-unrolling
- shared-memory-optimization
- software-exp
- fused-kernel
- gemm
- quantization
- cuda-cpp
- triton
---
# Normalization Pattern -- Skill Whitelist

This file lists the optimization skills applicable to a custom normalization kernel, in recommended application order. Only skills that currently exist under `wiki/nvidia/foundations/` (with a completed `skill.md`) are listed.

Normalization kernels are hybrid: a **reduction phase** (computing mean / variance / RMS) followed by an **elementwise scaling phase**. Skills are tagged by which phase they primarily affect.

## Primary skills (highest impact)

### 1. Warp Primitives (Shuffle Reduction) -- reduction phase

- **Skill path**: `wiki/nvidia/foundations/compute/warp-primitives/`
- **Why it matters for normalization**: the reduction phase (computing mean, variance, or RMS) is a sum reduction over the hidden dimension. Warp-level butterfly reduction via `__shfl_down_sync` replaces shared-memory tree reduction for the intra-warp stage, eliminating shared-memory round-trips. On H200, a full 5-step butterfly reduction takes approximately 145 cycles (measured).
- **When to apply**: always. Every normalization kernel's reduction phase must use warp shuffles for the intra-warp reduction. The canonical block-level reduction pattern uses shared memory only for inter-warp communication (warp leaders write partials to smem), then a final warp reduces those partials via shuffles.
- **Relevance to bottleneck triage**: Q4 in `reasoning/bottleneck-triage.md` -- "Thread 0 (or lane 0) does a serial reduction."

### 2. Global Memory Coalescing -- scaling phase

- **Skill path**: `wiki/nvidia/foundations/memory/coalescing/`
- **Why it matters for normalization**: the elementwise scaling phase reads every element of the input row, applies `(x - mean) * inv_std * gamma + beta`, and writes the result. This is a pure memory-bandwidth- bound operation. Coalesced (stride-1) loads and stores are essential for achieving peak bandwidth. On H200, non-coalesced access is 2.85x slower than coalesced access (measured).
- **When to apply**: always. Verify that consecutive threads access consecutive memory addresses in the scaling phase. For LayerNorm / RMSNorm with row-major layout [B*S, H], the standard pattern `input[row * H + threadIdx.x]` is coalesced. For BatchNorm with NCHW layout, spatial-axis access may not be coalesced; prefer NHWC.
- **Relevance to bottleneck triage**: Q4 in `reasoning/bottleneck-triage.md` -- "Reads/writes memory with stride > 1 per thread."

### 3. Vectorized Access -- scaling phase

- **Skill path**: `wiki/nvidia/foundations/memory/vectorized-access/`
- **Why it matters for normalization**: loading `float4` (128-bit) in the elementwise phase increases bytes-in-flight per thread and reduces the total number of load/store instructions. On H200, float4 vs scalar shows 2.59x bandwidth gain (measured).
- **When to apply**: when H is divisible by 4 (for float32) or by 8 (for float16/bfloat16), and the input pointer is 16-byte aligned. The grid-stride loop in the scaling phase becomes: `float4 v = reinterpret_cast<const float4*>(row_in)[i]; ...`
- **Relevance to bottleneck triage**: Q4 in `reasoning/bottleneck-triage.md` -- "Per-element scalar load/store when dtype * 4 fits a vector."

## Secondary skills (moderate impact)

### 4. Shared Memory Cache -- reduction + multi-pass phases

- **Skill path**: `wiki/nvidia/foundations/memory/shared-memory-cache/`
- **Why it matters for normalization**: LayerNorm and RMSNorm make two passes over each row — mean/variance first, then the scaling pass. Without staging the row in smem, the scaling pass reads every element from HBM a second time. A `__shared__ float row[H]` tile (for H that fits, typically H ≤ 8192 on H200 depending on block size) eliminates that second-round-trip and also serves as the inter-warp staging area for the warp-leader partial sums feeding skill #1 above.
- **When to apply**:
  - **Two-pass layout always benefits** from the row-in-smem cache when `H * sizeof(elem)` fits the smem budget. Sub-skill S1 from the sibling skill (temporal reuse) applies directly.
  - For the warp-leader partial-sum array (`__shared__ float warp_sums[32]`), no padding is needed (stride-1 access by warp_id is already bank-conflict-free).
  - For BatchNorm NCHW with spatial reduction, staging the HxW plane into smem can convert non-coalesced spatial reads into coalesced row-tile loads (sibling sub-skill S2).
- **When NOT to apply**: when H exceeds the per-block smem budget, fall back to multi-block partitioning + skill #11 (atomic-reduction) for the mean/variance combine. The row-cache stage is not possible in that regime.
- **Specific guidance**: the canonical block-level reduction pattern `smem[warpId] = warp_partial; __syncthreads(); if (warpId == 0) final_warp_reduce(smem[lane])` is both bank-conflict-free and requires `sizeof(T) * 32` of smem, leaving almost all of it for the row tile when H is the dominant claim.

### 5. Barrier Optimization -- between reduction and scaling phases

- **Skill path**: `wiki/nvidia/foundations/sync/barrier-optimization/`
- **Why it matters for normalization**: LayerNorm / RMSNorm has at least two `__syncthreads()` per row — one after the mean/variance reduction, another after the inv-std broadcast to all lanes before scaling. For multi-pass implementations (separate mean pass and variance pass), that doubles to four barriers per row. On H200 sm_90a, each `__syncthreads()` costs ~31–62 ns depending on block size (measured: `sources/experience/hw-probes/barrier-cost/`).
- **When to apply**:
  - **S1 narrow scope**: the broadcast of mean/inv-std from warp 0 to the other warps is typically block-scope and needs `__syncthreads`; no narrower scope is correct. The *intra-warp* reduction step, however, uses `__shfl_*_sync` intrinsics with embedded warp-sync — no `__syncwarp` needed.
  - **Single-pass online algorithm (Welford)**: computes mean + variance in a single pass, cutting the barrier count in half. Preferred over the two-pass variant when numerical stability is acceptable and the per-element compute of Welford is not the bottleneck.
  - **S2 arrive/wait split**: only helps when there is independent work between the mean/inv-std broadcast and the scaling phase — typically there is not (scaling uses the mean immediately). S2 is therefore not a fit for basic LayerNorm; it becomes relevant for fused LayerNorm + elementwise epilogue where the scaling loop itself runs in parallel with something independent (see pitfall P6 of the skill for the bare-cost caveat).
- **Relevance to bottleneck triage**: Q4 in `reasoning/bottleneck-triage.md` — "Multi-pass normalization kernel with `stall_barrier` above ~15 %."

### 6. Bank-Conflict Avoidance -- reduction phase

- **Skill path**: `wiki/nvidia/foundations/memory/bank-conflict/`
- **Why it matters for normalization**: the block-level reduction phase uses shared memory for inter-warp communication. When warp leaders write partial sums to `smem[warpId]`, bank conflicts can occur if multiple warps' lane-0 threads map to the same bank.
- **When to apply**: whenever the kernel uses `__shared__` memory for the reduction. Check that warp leaders write to distinct banks (they typically do if `warpId` values are consecutive, since each maps to a different bank). If the reduction uses a tree pattern in shared memory, verify sequential addressing to avoid conflicts.
- **Relevance to bottleneck triage**: Q4 in `reasoning/bottleneck-triage.md` -- "Narrow reduction into smem with heavy bank contention."

### 7. Fast Math Intrinsics -- scaling phase

- **Skill path**: `wiki/nvidia/foundations/compute/fast-math/`
- **Why it matters for normalization**: the scaling phase uses `rsqrtf(variance + eps)`. The `__frsqrt_rn` hardware intrinsic or `--use_fast_math` flag can accelerate this. LayerNorm/RMSNorm tolerance is typically relaxed enough (eps >= 1e-5) to allow fast-math approximations.
- **When to apply**: when the caller accepts reduced precision (fp32 with relaxed accuracy requirements). Do not apply for fp64 or when exact reproducibility is required.
- **Relevance to bottleneck triage**: Q4 in `reasoning/bottleneck-triage.md` -- "Heavy use of sinf/cosf/expf/logf on fp32 with relaxed precision ok."

### 8. Instruction-Level Parallelism (ILP) -- scaling phase

- **Skill path**: `wiki/nvidia/foundations/compute/ilp/`
- **Why it matters for normalization**: the elementwise scaling loop processes one element at a time by default. Unrolling with multiple independent computations per iteration hides FMA pipeline latency (4 cycles on H200). When combined with vectorized loads (float4), each iteration processes 4 elements with 4 independent FMA chains.
- **When to apply**: when H is large enough that each thread processes multiple elements (H > blockDim.x).
- **Relevance to bottleneck triage**: Q4 in `reasoning/bottleneck-triage.md` -- "Small tight loop inside kernel without #pragma unroll / ILP."

## Tertiary skills (situational)

### 9. Register Pressure Management

- **Skill path**: `wiki/nvidia/foundations/memory/register-pressure/`
- **Why it matters for normalization**: vectorized access + ILP + multiple reduction passes (mean, variance) increase register usage. High register pressure reduces occupancy, potentially limiting latency hiding. On H200, excessive register usage caused 4.84x slowdown from spilling (measured).
- **When to apply**: after applying skills 3, 5, and 6, check regs/thread via `-Xptxas=-v`. If regs/thread > 128, consider `__launch_bounds__` or splitting the kernel.

### 10. Compiler Hints

- **Skill path**: `wiki/nvidia/foundations/compute/compiler-hints/`
- **Why it matters for normalization**: `__launch_bounds__` controls register allocation ceiling; `#pragma unroll` exposes ILP in the grid-stride loops of both the reduction and scaling phases.
- **When to apply**: tuning the final kernel after functional correctness is established.

### 11. Async Copy (LDGSTS Prefetching)

- **Skill path**: `wiki/nvidia/foundations/memory/async-copy/`
- **Why it matters for normalization**: for kernels that make multiple passes over the same data (e.g., separate mean and variance passes in LayerNorm), prefetching the next tile into shared memory while computing on the current tile can hide memory latency. However, most normalization kernels are bandwidth-bound with simple arithmetic, so async copy provides limited benefit.
- **When to apply**: only when the normalization kernel is fused with compute-heavy operations (e.g., fused layernorm + GELU) and ncu shows "Stall Long Scoreboard" as the dominant bottleneck. For simple layernorm/rmsnorm, async copy is unlikely to help and may regress performance (see GTC25-S72683: trivial ops showed 10% regression).

### 12. Memory Ordering (correctness)

- **Skill path**: `wiki/nvidia/foundations/sync/memory-ordering/`
- **Why it matters for normalization**: multi-block normalization kernels (rare, for very large hidden dims) that communicate partial sums via global memory require correct memory ordering (e.g., `__threadfence()` before signaling completion to a coordinator block).
- **When to apply**: only for multi-block normalization kernels that use global-memory communication. Single-block kernels (the common case) do not need this skill.

### 13. Atomic Reduction Contention Control (multi-block reduction phase only)

- **Skill path**: `wiki/nvidia/foundations/sync/atomic-reduction/`
- **Why it matters for normalization**: when the hidden dimension H exceeds a single block's capacity (roughly H > 4096 for simple LayerNorm), the mean / variance reduction is split across multiple blocks per row and the per-block partial sums must be combined with a global atomic. Each block must commit **one atomic per block** (S1 pattern), NOT one per thread -- a per-thread atomic on a single per-row accumulator is both contention-pathological and numerically unsound in FP32 (on H200 the running-sum mantissa saturates after ~1e5 adds; atomic-reduction skill pitfall P7). For LayerNorm, accumulate in FP32 (or promote to FP64 at the block boundary) even when input/output are FP16/BF16, so precision loss at the atomic stage does not poison `mean` and `rstd`.
- **When to apply**: only when the per-row reduction is split across blocks. For the common case where one block handles one row (H <= 4096 on H200), use the in-block warp+shmem reduction and skip this skill.
- **Caveat**: most production LayerNorm kernels (Apex, Triton, cuDNN) choose their block layout specifically to keep one row per block and avoid the atomic path altogether. Only reach for this skill when row size forces multi-block partitioning.

### 14. L2 Access Policy (weight / gamma / beta residency across launches)

- **Skill path**: `wiki/nvidia/foundations/memory/l2-access-policy/`
- **Why it matters for normalization**: LayerNorm / RMSNorm / GroupNorm all re-read the **same gamma / beta parameter vectors** on every forward pass, and multi-pass variants (mean → variance pass over the input) re-read the full activation in the second pass. When those buffers sit in the 37.5 MiB–128 MiB range and the kernel is invoked repeatedly between other kernels on the same stream (the canonical inference loop), pinning them with `accessPolicyWindow` lets the second pass read from L2 rather than re-issuing DRAM traffic. Measured on H200 at WS = 80 MiB with tuned `hitRatio = set_aside / WS`: **+17.7%** effective BW over no policy.
- **When to apply**:
  - **Multi-pass normalization** (e.g. two-pass LayerNorm: mean pass + variance pass). The activation is hot *within* the kernel already, so a single-kernel harness shows no benefit (probe's WS = 40 MiB soft-null directly applies). Across two sequential kernels on the same stream, the second pass can persist the buffer from the first.
  - **Persistent-parameter kernels** (gamma / beta / RMSNorm scale) where parameters stay fixed across many token-step launches in inference. Size typically < 37.5 MiB — no special tuning needed; just a window with `hitRatio = 1.0` pins it for the whole stream.
  - **Fused norm+matmul epilogues** where the norm scale re-reads a weight tile already loaded by the gemm — if the tile exceeds L2 set-aside, use tuned `hitRatio = set_aside / tile_bytes`.
- **When NOT to apply**:
  - Single-kernel single-pass normalization with activation < L2. The activation is naturally hot through a grid-stride loop; no pinning benefit (measured neutral at WS = 4 / 40 MiB).
  - gamma / beta vectors that are tiny (< 1 MiB). Routine L2 caching already keeps them hot; window adds code complexity for zero gain.
- **Specific guidance for normalization**:
  - Pin **only** the parameters (gamma, beta, running_mean, running_var), not the activation stream. Activations are per-token ephemeral; tagging them persisting evicts reusable parameters. Use `hitProp=Persisting` for params, `missProp= Streaming` for everything else.
  - For the two-pass variant, pin the **activation** only between the two passes — set the window before pass 1, reset it after pass 2 via `cudaCtxResetPersistingL2Cache` (skill S3) so the next token-step starts clean.
- **Measured impact**: +17.7% effective BW at WS = 80 MiB on H200 when `hitRatio` is tuned. See `sources/experience/hw-probes/l2-residency.md`.
- **Relevance to bottleneck triage**: Q4 in `reasoning/bottleneck-triage.md` — normalization kernels whose second pass has DRAM bandwidth at saturation despite reading data just written by the first pass (no L2 reuse across launches).

### 15. Cache Load Hints (gamma/beta reads + two-pass activation re-reads)

- **Skill path**: `wiki/nvidia/foundations/memory/cache-load-hints/`
- **Why it matters for normalization**: the two-pass LayerNorm / RMSNorm pattern re-reads the same activation row in passes 1 (mean/var) and 2 (scaling); gamma / beta / running_mean / running_var parameters are also re-read across many token-step launches in inference. Both are L2-resident reuse patterns — exactly the regime where H200 measured **2.26× slowdown** if the loads are tagged `__ldcg` or `__ldcv`. The legacy KB's "use `__ldg` for gamma/beta" advice is harmless but redundant on sm_9.0a: with `const __restrict__` pointers the compiler auto-emits `ld.global.nc`, and explicit `__ldg` is within 0.1% of default at L2 regime.
- **When to apply**:
  - Declare gamma / beta / running_* parameters as `const float* __restrict__`. Compiler handles the rest. Skip explicit `__ldg`.
  - For the two-pass variant, default-load the activation row in pass 2 (it is L2-resident after pass 1). **Do NOT** use `__ldcg` — measured 2.26× slowdown.
  - Keep all loads default / `__ldca`. The hot data (row, gamma, beta) wants L1 staging for the multi-pass pattern.
- **When NOT to apply**:
  - Never use `__ldcg` / `__ldcv` on the activation or parameter buffers. The 2.26× L2-reuse penalty is the most common self-inflicted wound in normalization kernels.
  - `__ldcs` (evict-first) on gamma/beta: measured null in the un-contended case. Not worth the code noise.
- **Specific guidance for normalization**:
  - If you are composing with the `l2-access-policy` skill (priority 14) to pin gamma/beta across launches, keep the per-instruction loads at default. The window picks *which lines survive L2 pressure*; the default `ld.global.ca` picks *L1 + L2 staging*. They compose, and both being right is necessary for the fast path.
- **Measured impact on H200**: `__ldg` ≡ default on both DRAM and L2 regimes (within 0.5%); `__ldcg` / `__ldcv` cost 2.26× on L2-resident reuse. See `sources/experience/hw-probes/cache-hint.md`.
- **Relevance to bottleneck triage**: Q4 in `reasoning/bottleneck-triage.md` — normalization kernel with `__ldcg` on gamma/beta reported as slower than default. Correct response: revert to default, use `const __restrict__`, and compose with skill 14 (l2-access-policy) if cross-launch pinning is needed.

### 16. Half-Precision Math (fp16/bf16 for I/O; fp32 for the accumulator)

- **Skill path**: `wiki/nvidia/foundations/compute/half-precision-math/`
- **⚠ PRECISION IMPACT — correctness-critical for normalization**: normalization's mean / variance / RMS accumulates over the whole reduction axis. In fp16 a sum of 4096 terms of magnitude ~1 accumulates ~2 ULP of error (fp16 eps ≈ 4.88e-4 → 4096 × 4.88e-4 ≈ 2 per unit of sum), and the fp16 max finite of **6.55e4** is overrun by any hidden dim > ~2000 at activation magnitude > ~30. In bf16 the range is fp32-level but the mantissa is only 7 bits (eps ≈ 7.81e-3), so per-term error per add is ~16× worse. **Both formats are unsafe for the accumulator**; only the *storage* is safe.
- **Why it matters for normalization**: I/O halves (2× memory BW), compute gets 1.58–1.84× vs fp32 (measured) — all good for the memory-bound scaling phase. But the reduction phase (mean, variance) **must** accumulate in fp32, otherwise the computed mean / rstd is numerically unsound. This is the canonical "mixed-precision normalization" shape.
- **When to apply**:
  - **Storage in fp16 or bf16**: the activation row, gamma / beta parameters, and final output. The 2× memory BW is the dominant win.
  - **Reduction accumulator in fp32**: mean and variance partial sums. Promote via `__half2float` / `__bfloat162float` at the fp16/bf16 → fp32 boundary, accumulate in fp32, reduce in fp32 (see reduction pattern §§1-2 shuffle primitives), and produce `mean, rstd` in fp32.
  - **Scaling phase**: cast `mean, rstd` back at the store; the per-element compute `(x - mean) * rstd * gamma + beta` can run in fp16/bf16 directly (one-shot op per element, no accumulation).
  - **Packed `__hfma2` for gamma/beta-weighted scaling**: if the kernel is compute-bound on the scaling epilogue, consider packed half2 in the store loop. Measured H200 gain: **only 1.16× over scalar `__hfma`** (not 2×), so alignment and register cost must be justified.
  - **Fused RMSNorm + SiLU / GELU activation**: the transcendental is in the scaling phase. Use `h2exp` / `hrsqrt` with the caveat that sm_9.0a coverage of packed transcendentals is inferred (skill P10, not re-measured).
- **When NOT to apply**:
  - **Accumulating mean / variance in fp16**. Legacy P1 measured: trivial overflow for H > 2000. **This is a correctness bug, not a perf regression.**
  - **Accumulating mean / variance in bf16**. The 7-bit mantissa kills accuracy within ~0.8% per add — after 4096 adds the mean drifts by a full decimal digit.
  - **Native fp16 `atomicAdd` for multi-block variance combine**. Legacy P12: contention throughput worse than fp32, AND precision loss at each atomic step. Use hierarchical fp32 fan-in (`atomic-reduction` skill §S1), cast to fp16 only at the final store.
  - **bf16 when the per-layer eps is below ~1e-6**. bf16 min-normal is fp32-level so not denormal-bound, but the mantissa resolves only ~2 decimal digits — if `rstd` computed as `1 / sqrt(var + eps)` has `var ≈ eps`, the bf16 arithmetic rounds `var + eps` to just `var` (P4 adjacent).
- **Specific guidance for normalization**:
  - gamma / beta can be stored in fp16 or bf16 as long as their magnitudes fit (gamma typically ~1.0, beta typically ~0). Use `const __nv_bfloat16* __restrict__` parameters.
  - Compose with skill 15 (cache-load-hints) by default-loading — `const __restrict__` already emits the read-only path; do not reach for explicit `__ldg`.
  - Do NOT move to packed `__hfma2_relu` (fused FMA + ReLU) unless the kernel is already compute-bound after the mixed-precision shape; legacy P11 — memory-bound kernels see no wall-clock gain from the 4% instruction reduction.
- **Measured impact on H200 sm_9.0a** (see `sources/experience/hw-probes/half2-throughput.md`):
  - fp16 scalar FMA = 45 TFLOPS (1.58× fp32).
  - fp16 packed FMA = 52.6 TFLOPS (1.84× fp32, 1.16× fp16 scalar).
  - bf16 packed FMA = 46.5 TFLOPS (12% slower than fp16 packed).
  - 2× memory BW win from I/O halving is the dominant gain for the scaling phase.
- **Relevance to bottleneck triage**: Q1/Q4 in `reasoning/bottleneck-triage.md` — normalization kernel with fp16 I/O but computed `mean` or `rstd` showing numerical drift vs. a fp32 reference. Correct response: promote the accumulator to fp32 (skill §S3); keep I/O in fp16/bf16.

### 17. Branch Elimination (epsilon floor, abs-diff stats, clamp-style activations)

- **Skill path**: `wiki/nvidia/foundations/compute/branch-elimination/`
- **Why it matters for normalization**:
  - **Epsilon floor** on the variance denominator: `rstd = rsqrtf(fmaxf(var, eps))` is the canonical form — one `FMNMX` for the floor + one `MUFU.RSQ` for the rsqrt. Writing `rstd = rsqrtf(var + eps)` is equivalent in math but differs numerically for near-zero `var`; `rstd = rsqrtf(var < eps ? eps : var)` emits the same `FMNMX` as `fmaxf`, so the three forms are interchangeable at SASS — **pick the clearest-reading one, not the "branchless-looking" one**.
  - **MAE / abs-diff statistics** (e.g. `sum_abs_dev = sum(|x - mean|)` for median absolute deviation or quantile loss): use `fabsf(x - mean)` — 1 `LOP3.LUT` vs 2-3 ops for `if (x < mean) d = mean - x; else d = x - mean;`. Measured 2.36× speedup on H200.
  - **Clamp-style activations fused into normalization** (e.g. clipped ReLU, hardtanh, quantized layer output): use `fmaxf(fminf(x, hi), lo)` — two `FMNMX` ops. Do NOT simulate as `x < lo ? lo : (x > hi ? hi : x)` — same SASS but harder to read; do NOT simulate as arithmetic masks.
- **When to apply**:
  - Variance epsilon floor — always `fmaxf(var, eps)`.
  - GroupNorm's clamp-to-valid-range fused step — `fmaxf`/`fminf`.
  - Abs-deviation stats or gradient clipping fused into LayerNorm backward — `fabsf`.
- **When NOT to apply**:
  - Do NOT replace `if (cond) smem[lane] = partial` (used in reduction inter-warp staging) with arithmetic — the compiler predicates the guarded store to `FSEL + store` when the body is a single assignment; arithmetic simulation is strictly more ops (branch-elimination pitfall P1).
  - Do NOT bit-trick the epsilon floor with `__int_as_float(__float_as_int(var) | epsilon_bits)` — the round-trip is slower than `fmaxf` (branch-elimination pitfall P2).
- **Measured on H200**: `fmaxf` 1.56×, `fabsf` 2.36× over the `if/else` form (branchless-patterns probe).
- **Relevance to bottleneck triage**: Q4 in `reasoning/bottleneck-triage.md` — normalization kernel where variance denominator arithmetic shows 2-op `FSETP+FSEL` instead of `FMNMX` for the epsilon floor. Correct response: use `fmaxf(var, eps)`.
