---
title: Transpose Pattern -- Skill Routing
pattern_class: cuda-core
op: transpose
status: draft
source:
- path: spec
  anchor: Reference
id: routing-transpose-ROUTING
type: operator-routing
vendor: nvidia
operator: transpose
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L1484-L1540
architectures:
- sm90
- sm90a
languages:
- cuda-cpp
techniques:
- pipeline-stages
- vectorized-loads
- cache-policy
- register-budgeting
- data-reuse
- shared-memory-optimization
- communication-overlap
confidence: inferred
tags:
- pipeline-stages
- vectorized-loads
- cache-policy
- register-budgeting
- data-reuse
- shared-memory-optimization
- communication-overlap
- fused-kernel
- cuda-cpp
kernel_types:
- fused-kernel
---
# Transpose Pattern -- Skill Whitelist

This file lists the optimization skills applicable to a custom transpose kernel, in recommended application order. Only skills that currently exist under `wiki/nvidia/foundations/` (with a completed `skill.md`) are listed.

## Primary skills (critical)

### 1. Shared Memory Cache (the central mechanism)

- **Skill path**: `wiki/nvidia/foundations/memory/shared-memory-cache/`
- **Why it matters for transpose**: the naive transpose forces one side of the access pattern (load or store) to be non-coalesced. Sub- skill S2 of this skill (coalescing transform via smem) *is* the transpose mechanism: load coalesced into `__shared__`, `__syncthreads()`, read column-wise and write coalesced. The worked example is PG §2.2.4.2.1 (L1484-L1540).
- **When to apply**: always, for Q5a (2-D matrix transpose). For Q5c (AoS↔SoA conversion), smem is not needed — the conversion is a one-pass copy kernel.
- **Specific guidance**:
  - Default tile size is `[32][33]` for fp32 (32 threads/row × 32 rows per tile + 1 padding column).
  - For small matrices (< 512² fp32), drop to `[16][17]` or skip smem entirely — see pitfall P7 of the skill.
  - For fp16 / bf16, `[32][34]` (stride +2 in 16-bit units = +1 32-bit bank) is the canonical shape.
- **Measured on H200**: 4096² fp32 transpose at 1685 GB/s with padded smem, 985 GB/s without padding, 528 GB/s with naive no-smem. Probe: `sources/experience/hw-probes/smem-tile-reuse/`.
- **Relevance to bottleneck triage**: Q4 in `reasoning/bottleneck-triage.md` — "Non-coalesced store pattern with potential smem pivot."

### 2. Bank-Conflict Avoidance (critical companion to skill #1)

- **Skill path**: `wiki/nvidia/foundations/memory/bank-conflict/`
- **Why it matters for transpose**: the unpadded `[TILE][TILE]` smem tile causes 32-way bank conflicts on the column read phase (skill #1 above). Adding a `+1` padding column shifts each row by one bank and eliminates the conflict.
- **When to apply**: whenever the transpose kernel uses 2-D smem — which is "always" for Q5a.
- **Measured on H200** (sibling skill's probe): unpadded `[32][32]` → 16,326,828 bank conflicts per kernel; padded `[32][33]` → 33,449 (488× drop; 1.71× wall-clock speedup on top of S2).
- **Relevance to bottleneck triage**: Q4 in `reasoning/bottleneck-triage.md` — "Smem column access on stride-32 tile without +1 padding."

### 3. Layout Transform (required for Q5c, complementary to Q5a)

- **Skill path**: `wiki/nvidia/foundations/memory/layout-transform/`
- **Why it matters for transpose**: sub-skill S1 (AoS → SoA) is the right technique when the data is struct-shaped and only some fields are hot — this is *not* a matrix transpose but frequently shows up in transpose-pattern tasks (e.g. particle-system state conversion). Sub-skill S2 (`cudaMallocPitch`) ensures each row of a 2-D matrix starts at a coalescing-aligned address before the transpose kernel runs. Sub-skill S3 (in-kernel transpose) overlaps with skill #1 above — the mechanism lives in shared-memory-cache, the *amortization decision* lives here.
- **When to apply**:
  - Q5c (AoS↔SoA): use S1.
  - Rows not aligned to 128 B: use S2.
  - Considering whether to materialize the transpose: use S3 + amortization guidance.
- **Measured on H200**: AoS-read-one-field at 1208 GB/s effective; SoA-read-one-field at 2405 GB/s effective (1.99×); SoA-vectorized at 3613 GB/s (2.99× total). Conversion kernel amortizes at 3.65 downstream calls. Probe: `sources/experience/hw-probes/aos-vs-soa/`.
- **Relevance to bottleneck triage**: Q4 in `reasoning/bottleneck-triage.md` — "Stride-N gmem reads because of struct layout."

## Secondary skills (moderate impact)

### 4. Vectorized Access

- **Skill path**: `wiki/nvidia/foundations/memory/vectorized-access/`
- **Why it matters for transpose**: when the tile element fits a vector type (`float4`, `bfloat162`, `half2`), loading the tile in vectorized chunks reduces the load-instruction count proportionally. On H200 in the aos-vs-soa probe, adding float4 to SoA delivered another 1.5× on top of the stride-1 fix.
- **When to apply**: large-tile transposes (≥ 128² per tile) where the element dtype is 4-byte or 2-byte aligned. For fp32 tiles of `[32][32]`, float4 gives a 4× reduction in load instructions. For fp16 / bf16 tiles, consider reading as `float` (packing 2 elements) or `float2` (packing 4 elements).
- **When NOT to apply**: tile size < 128² or element dtype > 32 bits (e.g. int64) where float4 is not applicable.

### 5. Global Memory Coalescing (sanity check)

- **Skill path**: `wiki/nvidia/foundations/memory/coalescing/`
- **Why it matters for transpose**: skills #1-#4 arrange for both gmem sides of the transpose to be coalesced. This skill is the sanity check — after writing the kernel, verify with NCU that both the global load sector count and the global store sector count match the theoretical minimum `N² × sizeof(elem) / 32 B`.
- **When to apply**: always, as the final correctness check. NCU metrics: `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` and `l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum`.

## Tertiary skills (situational)

### 6. Compiler Hints

- **Skill path**: `wiki/nvidia/foundations/compute/compiler-hints/`
- **Why**: `__restrict__` on the input/output pointers enables the compiler to assume no aliasing; `__launch_bounds__` can raise the register ceiling for vectorized transpose kernels where register pressure is otherwise occupancy-limiting.
- **When**: tuning the final kernel after functional correctness.

### 7. Register Pressure Management

- **Skill path**: `wiki/nvidia/foundations/memory/register-pressure/`
- **Why**: vectorized transpose + large tile + fp16 unpack-repack can push registers per thread above the occupancy break-even. Monitor with `-Xptxas=-v` after applying skill #4.
- **When**: after applying skills #1-#4, if occupancy drops below 50 %.

## Skills NOT applicable to transpose

- **Warp primitives** (`wiki/nvidia/foundations/compute/warp-primitives/`): transpose is embarrassingly parallel across output elements. No cross-thread reduction or scan is involved. If a future variant requires e.g. argmax along a transposed axis, it is a reduction-after-transpose composition — use the reduction pattern's routing.
- **Fast math** (`wiki/nvidia/foundations/compute/fast-math/`): transpose performs zero floating-point arithmetic.
- **Async copy** (`wiki/nvidia/foundations/memory/async-copy/`): for a single-pass transpose the `cp.async` overlap window is the smem-stage interval (~20 % of runtime). The overhead of setting up the pipeline usually exceeds the overlap benefit. Exception: multi-stage pipelined transpose for very large matrices (> 32K²) where the tile stream itself is long enough to amortize. Not in the current scope.
- **Atomic reduction** (`wiki/nvidia/foundations/sync/atomic-reduction/`): no atomic operations appear in transpose (each output is written by exactly one thread).
- **Memory ordering** (`wiki/nvidia/foundations/sync/memory-ordering/`): no cross- block communication.
- **Branch elimination** (`wiki/nvidia/foundations/compute/branch-elimination/`): transpose's only conditional is the tail boundary check (`if (x < W && y < H) out[y*W + x] = smem[...]`). nvcc 12.9 predicates this simple-body bounds check to a guarded store (`FSEL` + predicated `STG`) with no inner-loop `BRA` — verified via `cuobjdump --dump-sass` on reference transpose kernels. No rewrite opportunity; no `fmaxf` / `fabsf` apply because transpose performs no arithmetic. Listed here only to document the explicit "not-applicable" decision.
