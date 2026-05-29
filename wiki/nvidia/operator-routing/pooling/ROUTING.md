---
title: Pooling Pattern -- Skill Routing
pattern_class: cuda-core
op: pooling
status: draft
id: routing-pooling-ROUTING
type: operator-routing
vendor: nvidia
operator: pooling
---
# Pooling Pattern -- Skill Whitelist

This file lists the optimization skills applicable to a custom pooling kernel, in recommended application order. Only skills that currently exist under `knowledge/30-skill/` (with a completed `skill.md`) are listed.

## Applicable skills

### 1. Global Memory Coalescing (primary)

- **Skill path**: `30-skill/memory/coalescing/`
- **Why it matters for pooling**: each output element requires reading a kH x kW window from the input tensor. The output store pattern -- adjacent threads writing to adjacent output elements -- must be coalesced to achieve peak write bandwidth. The input loads are determined by the window offset and may or may not be coalesced depending on tensor layout and stride. For NCHW layout with the width dimension mapped to consecutive threads, the output writes are naturally coalesced (stride-1 in the W dimension). Input loads within a single row of the pooling window are also coalesced when threads read from the same row with adjacent W offsets.
- **When to apply**: always. Verify that the thread-to-output mapping produces coalesced stores, and that the innermost input-load loop accesses contiguous memory.
- **Relevance to bottleneck triage**: Q4 in `70-reasoning/bottleneck-triage.md` -- "Reads/writes memory with stride > 1 per thread."

### 2. Shared Memory Cache (primary, prerequisite for overlapping windows)

- **Skill path**: `30-skill/memory/shared-memory-cache/`
- **Why it matters for pooling**: adjacent output elements' receptive fields overlap by `kH-stride_H` rows and `kW-stride_W` columns. Without staging, each input element is re-read from global memory once per overlapping output element it participates in — for a 3×3 window with stride 1, that's 9 global reads per input pixel. The sibling skill's S1 sub-skill (tile reuse) turns this into a single coalesced global load per pixel + many smem reads.
- **When to apply**: whenever the pooling stride is smaller than the kernel size (`stride < kH || stride < kW`), giving reuse ≥ 2×. For non-overlapping windows (stride ≥ kernel size), reuse is 0 and the smem stage is pure overhead — prefer the "direct global read" path (see pitfall P5 in the sibling skill).
- **Specific guidance for pooling**:
  - Block tile size = `(TILE_H + kH - stride_H) × (TILE_W + kW - stride_W)`; plus `__syncthreads()` after the load.
  - Always declare smem with `+1` padding along the column axis (`__shared__ float tile[TILE_H + kH - 1][TILE_W + kW]`) to pre-empt bank conflicts during per-row scans of the window (skill P8 was measured on H200: unpadded `[32][32]` → 16.3 M bank conflicts, padded `[32][33]` → 33 K, a 488× drop).
- **Relevance to bottleneck triage**: Q4 in `70-reasoning/bottleneck-triage.md` -- "Redundant global reads of overlapping windows."

### 3. Bank-Conflict Avoidance (primary)

- **Skill path**: `30-skill/memory/bank-conflict/`
- **Why it matters for pooling**: when the kernel uses shared memory to stage overlapping input tiles (skill 2 above), the column-read access pattern during the window scan can cause 32-way bank conflicts. The canonical fix is `[TILE_H][TILE_W + 1]` padding.
- **When to apply**: whenever the kernel uses `__shared__` memory for input tiling. If the kernel does not use shared memory (e.g., direct global reads for small non-overlapping windows), this skill is not needed.
- **Relevance to bottleneck triage**: Q4 in `70-reasoning/bottleneck-triage.md` -- "Narrow reduction into smem with heavy bank contention."

### 4. Vectorized Access (secondary)

- **Skill path**: `30-skill/memory/vectorized-access/`
- **Why it matters for pooling**: when multiple channels or width elements can be loaded together (e.g., NHWC layout where consecutive channels are contiguous, or NCHW with stride-1 pooling along W), using float4/float2 vector loads increases bytes per load instruction and reduces total instruction count. This is most beneficial for the input-loading phase when the access is naturally aligned.
- **When to apply**: when the input tensor is 16-byte aligned (or 8-byte for float2), the innermost dimension fits a vector width, and the pooling stride allows contiguous vector reads. For kW=1 pooling (1D vertical pooling), each row load is a single scalar and vectorization does not help.
- **Relevance to bottleneck triage**: Q4 in `70-reasoning/bottleneck-triage.md` -- "Per-element scalar load/store when dtype * 4 fits a vector."

### 5. Warp Divergence (boundary-tile + stride-variable handling)

- **Skill path**: `30-skill/compute/warp-divergence/`
- **Why it matters for pooling**: two divergence sources in pooling: (a) the right/bottom edge tiles have out-of-bounds predicates where ~half the warp's lanes may fall off the image; (b) variable-stride or adaptive pooling inserts per-lane decisions on which input rows to read. Both are often short-body predicable — the problem is when nvcc fails to predicate because the body contains a `float` load address computation plus a dtype-cast-to-output-type.
- **When to apply**:
  - For boundary tiles, rewrite as arithmetic guard: `val = (x < W && y < H) ? input[y*W + x] : 0.0f;` (single selp on the loaded value). The guard should NOT wrap the load instruction itself — predicated loads still evaluate addresses (BP §13.2), which is fine for memory safety here.
  - For adaptive pooling, if the kernel is compute-bound, sort the input tiles by window size and launch one kernel per bucket (skill S4; amortization check per pitfall P8).
- **When NOT to apply**: boundary-tile predicates when kernel is already memory-bound (the common case for 2×2/3×3 pooling on large feature maps). Measured on H200: memory-bound version of the divergence probe was flat across p — rewriting boundary logic to be predicable would add code complexity without moving wall-clock.
- **Relevance to bottleneck triage**: Q4 in `70-reasoning/bottleneck-triage.md` — "Boundary-tile predicate forcing a real branch on a compute-bound pooling kernel."

### 6. Instruction-Level Parallelism (secondary)

- **Skill path**: `30-skill/compute/ilp/`
- **Why it matters for pooling**: the innermost loop of a pooling kernel iterates over the kH x kW window, performing a load + compare (max) or load + add (avg) per element. For larger windows (5x5, 7x7), unrolling this loop with `#pragma unroll` exposes independent operations to the instruction scheduler, hiding load latency behind computation. Multiple independent accumulator chains can be used when the window is processed in row-major order.
- **When to apply**: when kernel_size >= 3 and the inner loop is not already unrolled by the compiler. Check the generated SASS or PTX to verify whether the compiler unrolled without the pragma.
- **Relevance to bottleneck triage**: Q4 in `70-reasoning/bottleneck-triage.md` -- "Small tight loop inside kernel without #pragma unroll / ILP."

### 7. Compiler Hints

- **Skill path**: `30-skill/compute/compiler-hints/`
- **Why**: `__launch_bounds__` constrains register allocation for the pooling kernel, enabling the compiler to target a specific occupancy level. `#pragma unroll` on the window loops is the most impactful hint for pooling kernels.
- **When**: tuning the final kernel after functional correctness is established and the primary skills above have been applied.

### 8. Register Pressure Management

- **Skill path**: `30-skill/memory/register-pressure/`
- **Why**: larger pooling windows with unrolled loops and vectorized loads can increase register usage, potentially reducing occupancy. Monitor registers per thread via `-Xptxas=-v` after applying vectorized-access and ILP.
- **When**: after applying skills 3-4, if occupancy has dropped and latency has not improved proportionally.

## Skills NOT applicable to pooling

The following skills exist in the KB but are generally not relevant for standard pooling kernels:

- **Warp primitives** (`30-skill/compute/warp-primitives/`): pooling windows are typically small (2x2 to 7x7) and each output element is computed independently by a single thread. There is no cross-thread reduction within a warp (unlike global reduction kernels). If a future variant requires reducing across threads (e.g., very large adaptive windows), this skill would become relevant.

- **Fast math** (`30-skill/compute/fast-math/`): pooling involves only max/add/divide operations, which are exact in IEEE-754. There are no transcendental functions to approximate.

- **Async copy** (`30-skill/memory/async-copy/`): async global-to-shared copies (LDGSTS) are primarily beneficial for iterative, compute-heavy kernels. Standard pooling kernels load each input tile once and perform minimal compute (max or sum), so the copy-compute overlap window is too small to benefit.

- **Memory ordering** (`30-skill/sync/memory-ordering/`): pooling kernels are embarrassingly parallel across output elements. There is no inter-block communication or global atomics.

### 9. Branch Elimination (max-pooling comparator + boundary clamp)

- **Skill path**: `30-skill/compute/branch-elimination/`
- **Why it matters for pooling**:
  - **Max-pooling's inner comparator** is the canonical `fmaxf` use case. The pooling window accumulator is updated as `acc = fmaxf(acc, input[...])` — **1 `FMNMX` SASS op** per element. Writing `acc = (input[...] > acc) ? input[...] : acc` is functionally identical but emits 2 ops (`FSETP + FSEL`), **1.56× slower per update**.
  - **Adaptive pooling boundary clamp** (`idx_start = (out_idx * in_len) / out_len` + bounds check): `idx = min(max(idx, 0), in_len - 1)` compiles to two `IMNMX` integer min/max — one op per clamp. Do NOT rewrite as `idx = idx * (idx > 0) * (idx < in_len - 1)` — arithmetic simulation is strictly slower and only masks the value to 0, not to a valid bound.
  - **Zero-padding on boundary tiles** (`val = (in_idx < 0 || in_idx >= H) ? 0 : input[in_idx]`): compiler predicates this to a guarded load + FSEL; already one op per boundary check. Do NOT hand-rewrite.
- **When to apply**:
  - Max-pooling / max-over-window: use `fmaxf` for the accumulator update — measured 1.56× faster per step (branchless-patterns probe Group A).
  - Adaptive-pool bounds: use `min(max(...))` pattern — compiler lowers to `IMNMX` ops.
  - Avg-pooling with variable window (edge tiles): prefer the compiler's predicated masked-load over any hand-coded predicate mask.
- **When NOT to apply**:
  - Do NOT rewrite avg-pool's `if (in_bounds) { sum += val; count++; }` as `sum += val * in_bounds; count += in_bounds;` — the branchful form gets predicated to a pair of FSELs and the arithmetic form adds FMUL/FADD ops (branchless-patterns probe's cond-arith variant: 1.27× slower).
  - Do NOT hand bit-trick a boundary-fix `(idx < 0) ? 0 : idx` as `idx & ~(idx >> 31)` — for the pooling range this emits the same-or-more ops than the clean `max(0, idx)` intrinsic form.
- **Measured on H200**: `fmaxf` 1.56×, boundary clamp via `min(max(...))` single-op per dim. branchless-patterns probe.
- **Relevance to bottleneck triage**: Q4 in `70-reasoning/bottleneck-triage.md` — max-pooling kernel where SASS shows `FSETP + FSEL` pairs instead of `FMNMX`. Correct response: switch inner-loop comparator to `fmaxf`.
