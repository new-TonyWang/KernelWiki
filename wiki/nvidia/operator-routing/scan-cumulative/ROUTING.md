---
title: Scan (Cumulative) Pattern -- Skill Routing
pattern_class: cuda-core
op: scan-cumulative
status: draft
id: routing-scan-cumulative-ROUTING
type: operator-routing
vendor: nvidia
operator: scan-cumulative
architectures:
- sm90
- sm90a
languages:
- cuda-cpp
techniques:
- pipeline-stages
- double-buffering
- vectorized-loads
- cache-policy
- kernel-fusion
- loop-unrolling
- shared-memory-optimization
- communication-overlap
kernel_types:
- fused-kernel
- attention
- quantization
confidence: inferred
tags:
- pipeline-stages
- double-buffering
- vectorized-loads
- cache-policy
- kernel-fusion
- loop-unrolling
- shared-memory-optimization
- communication-overlap
- fused-kernel
- attention
- quantization
- cuda-cpp
---
# Scan (Cumulative) Pattern -- Skill Whitelist

This file lists the optimization skills applicable to a custom scan (prefix sum / cumulative) kernel, in recommended application order. Only skills that currently exist under `wiki/nvidia/foundations/` (with a completed `skill.md`) are listed.

## Applicable skills

### 1. Warp Primitives (Shuffle Scan)

- **Skill path**: `wiki/nvidia/foundations/compute/warp-primitives/`
- **Why it matters for scan**: the intra-warp inclusive scan via `__shfl_up_sync` is the fundamental building block of every custom scan kernel. Unlike reduction (which uses `__shfl_xor_sync` or `__shfl_down_sync` for butterfly patterns), scan uses `__shfl_up_sync` in a Hillis-Steele pattern: for offsets 1, 2, 4, 8, 16, each thread fetches the value from `offset` lanes earlier and adds it to its own accumulator. This completes a warp-wide inclusive scan in 5 steps without touching shared memory.
- **When to apply**: always. Every custom scan kernel's intra-warp phase should use `__shfl_up_sync`. Falling back to shared-memory scan within a warp wastes bandwidth and adds `__syncwarp` overhead.
- **Relevance to bottleneck triage**: Q4 in `reasoning/bottleneck-triage.md` -- "Thread 0 (or lane 0) does a serial reduction."

### 2. Shared Memory Cache (prerequisite for block-level scan)

- **Skill path**: `wiki/nvidia/foundations/memory/shared-memory-cache/`
- **Why it matters for scan**: scan is fundamentally stateful — each output position depends on all preceding inputs — so a block-level scan must stage its working set in shared memory, then iteratively propagate partial results. Both the Hillis-Steele in-place variant (cuda-samples `scan1Inclusive`) and the Blelloch upsweep/downsweep variant rely on a single `__shared__` buffer as the scan workspace. The sibling skill's sub-skill S1 (smem as user-managed cache) is the exact mechanism; S3 (dynamic smem) is how the buffer is sized when block dim is chosen at launch.
- **When to apply**: always for block-level scans over > 32 elements. For warp-level scans (`__shfl_up_sync` ladder), smem is not needed. For grid-level scans, each block's smem holds its tile; the cross-block combine uses memory-ordering primitives (skill 6 below), not smem.
- **Specific guidance for scan**:
  - Hillis-Steele requires double-buffered smem when done in-place (`smem[2][N]`, swap the read/write index each pass) to avoid the read-after-write race.
  - Blelloch's upsweep/downsweep typically uses a single `smem[N]` buffer with power-of-two strides; those strides are the exact patterns that trigger bank conflicts (see skill 3 below).
  - For block-sizes that force > 48 KB of smem (large tile per block), use the sibling skill's S3 opt-in pattern (`cudaFuncAttributeMaxDynamicSharedMemorySize`).
- **Relevance to bottleneck triage**: Q4 in `reasoning/bottleneck-triage.md` -- "Multi-pass block-level algorithm without smem working set."

### 3. Barrier Optimization (scan is barrier-heavy)

- **Skill path**: `wiki/nvidia/foundations/sync/barrier-optimization/`
- **Why it matters for scan**: block-level scan algorithms (Hillis- Steele, Blelloch) are inherently barrier-heavy — each iteration of the scan tree must `__syncthreads()` between read and write of the smem buffer. A 1024-element block scan with Blelloch requires `2 × log2(1024) = 20` `__syncthreads()` calls per phase (upsweep
  + downsweep), plus one at the end for final write-back. On H200
  sm_90a at block size 1024, each `__syncthreads()` costs ~62 ns (measured: `sources/experience/hw-probes/barrier-cost/`), so barrier cost alone is ~1.2 μs per block — often a noticeable fraction of total kernel time for scans.
- **When to apply**:
  - **S1 narrow scope**: when the scan fits in a single warp (N ≤ 32 lanes), use `__shfl_up_sync` / `__shfl_down_sync` ladder with zero `__syncthreads()` calls. Saves all 20+ block barriers at the cost of only handling one warp's worth of scan; block- level scans then become "warp scan per warp, inter-warp aggregation via one `__syncthreads()`, then broadcast".
  - **Decoupled-lookback single-pass scan** (50-classical-algo, pending bucket G): replaces multiple kernel launches + barriers with one kernel that streams tiles, cutting most barrier traffic. Defer until the classical-algo layer is bootstrapped.
- **When NOT to apply**: if the scan already uses warp-shuffle primitives throughout and the barriers are only the unavoidable minimum, the barrier path is not the bottleneck — look at memory coalescing (skill #5) or ILP (skill #6) instead.
- **Relevance to bottleneck triage**: Q4 in `reasoning/bottleneck-triage.md` — "Scan kernel with `stall_barrier` > 20 %."

### 4. Bank-Conflict Avoidance

- **Skill path**: `wiki/nvidia/foundations/memory/bank-conflict/`
- **Why it matters for scan**: block-level scan kernels use shared memory to communicate warp totals between warps. The classic Hillis-Steele shared-memory scan (cuda-samples scan1Inclusive) performs alternating read-write passes on shared memory with power-of-two strides, which causes bank conflicts at certain offsets. The +1 padding technique or switching to a warp-shuffle based approach avoids this.
- **When to apply**: whenever the kernel uses `__shared__` memory for the inter-warp scan tree or for the primary scan data. Particular attention is needed when the shared memory array uses a stride that is a multiple of 32.
- **Relevance to bottleneck triage**: Q4 in `reasoning/bottleneck-triage.md` -- "Narrow reduction into smem with heavy bank contention."

### 5. Global Memory Coalescing

- **Skill path**: `wiki/nvidia/foundations/memory/coalescing/`
- **Why it matters for scan**: unlike reduction (read-only), scan kernels both read the full input and write the full output to global memory. Both load and store phases must be coalesced. The standard pattern `tid = blockIdx.x * blockDim.x + threadIdx.x; val = input[tid]; ... output[tid] = result;` is coalesced. Multi-dimensional scans along a non-contiguous axis require special attention.
- **When to apply**: always. Verify both load and store are stride-1 access. For 2-D scans along the non-contiguous axis, consider transposing the data first.
- **Relevance to bottleneck triage**: Q4 in `reasoning/bottleneck-triage.md` -- "Reads/writes memory with stride > 1 per thread."

### 6. Instruction-Level Parallelism (ILP)

- **Skill path**: `wiki/nvidia/foundations/compute/ilp/`
- **Why it matters for scan**: when each thread processes multiple input elements (the vectorized / multi-element-per-thread approach), the serial accumulation phase can benefit from multiple independent accumulator chains if the thread processes elements from non-adjacent segments. However, for a standard serial prefix sum within a thread's tile, ILP is limited because each step depends on the previous. ILP primarily helps the global-memory load phase when combined with vectorized access.
- **When to apply**: after vectorized access is in place and the load/compute overlap can be improved. Less impactful for scan than for reduction because the serial dependency chain is inherent.
- **Relevance to bottleneck triage**: Q4 in `reasoning/bottleneck-triage.md` -- "Small tight loop inside kernel without #pragma unroll / ILP."

### 7. Async Copy (LDGSTS Prefetching)

- **Skill path**: `wiki/nvidia/foundations/memory/async-copy/`
- **Why it matters for scan**: for iterative multi-tile scans where each block processes multiple tiles in sequence (e.g., a segmented scan with large segments), prefetching the next tile via `cuda::memcpy_async` while computing the current tile increases bytes-in-flight. Only beneficial when the scan involves non-trivial per-element compute (e.g., a scan of `sqrt(x) + exp(x)`). For simple sum-scan over a flat array, async copy typically adds overhead without benefit.
- **When to apply**: when the kernel processes multiple tiles per block AND per-element computation is non-trivial. Evaluate carefully.
- **Relevance to bottleneck triage**: Q4b in `reasoning/bottleneck-triage.md` -- bytes-in-flight analysis.

### 8. Memory Ordering (correctness)

- **Skill path**: `wiki/nvidia/foundations/sync/memory-ordering/`
- **Why it matters for scan**: the decoupled-lookback single-pass scan algorithm requires correct global memory ordering between blocks. Each block publishes its partial/full aggregate via global memory and subsequent blocks read it. Without proper `__threadfence()` or `cuda::atomic` operations with appropriate memory order, data races cause silent corruption. This is a **correctness** skill, not a performance skill.
- **When to apply**: whenever implementing a custom multi-block scan that communicates between blocks via global memory (e.g., decoupled lookback, or the three-kernel approach where block totals are exchanged).
- **Relevance to bottleneck triage**: Q4 in `reasoning/bottleneck-triage.md` -- "Data race or inconsistent cross-thread reads."

### 9. Branch Elimination (Kogge-Stone lane guards + max-scan / min-scan comparators)

- **Skill path**: `wiki/nvidia/foundations/compute/branch-elimination/`
- **Why it matters for scan**:
  - **Kogge-Stone / Hillis-Steele lane guards** (`if (lane_id >= offset) val = __shfl_up_sync(mask, val, offset) + val;`) are the canonical shuffle-scan pattern. The `if (lane_id >= offset)` guard is a simple-body predicate that compiler lowers to `FSEL` automatically — no inner-loop `BRA` is emitted. No rewrite needed.
  - **Max-scan / min-scan variants** (common in quantile estimation, backward compatibility prefix-max): the inner comparator should be `fmaxf(prev, cur)` — **1 `FMNMX` op** vs 2 for the equivalent ternary. Same 1.56× saving as reduction max.
  - **Decoupled-lookback status polling** (`if (status == FULL) break;`): the compiler **cannot** predicate this loop exit — it must emit a real `BRA`. Here the concern is not branch elimination but the lookback loop's warp vote and memory ordering (skill 8, memory-ordering).
  - **Lane-ID masks for sub-warp scans** (cooperative_groups tiled_partition): the partition's `__shfl_up_sync` already embeds the lane-selection mask; no C-level `if` needed.
- **When to apply**:
  - Max-prefix / min-prefix scan: switch inner comparator from `(a > b) ? a : b` to `fmaxf(a, b)`.
  - Any fused compute in the scan value (e.g. `val = fmaxf(val, 0.0f)` in a clamped prefix sum): one `FMNMX` vs two-op branch form.
- **When NOT to apply**:
  - Do NOT rewrite Hillis-Steele's `if (lane_id >= offset) val = ...` as arithmetic masking (`val = ((lane_id >= offset) ? shf(val, offset) : 0) + val`). The compiler predicates the original form to a single `FSEL`; the "arithmetic" version adds a compare + FMUL + FADD (measured 1.27× slower per step in branchless-patterns probe's cond-arith variant).
  - Do NOT bit-trick the lookback status check — status polling needs correct memory ordering (`memory-ordering` skill), not branch elimination.
- **Measured on H200**: `fmaxf` 1.56× per comparator step (branchless-patterns probe Group A).
- **Relevance to bottleneck triage**: Q4 in `reasoning/bottleneck-triage.md` — max-scan kernel showing `FSETP + FSEL` pairs instead of `FMNMX` in the shuffle-scan inner loop. Correct response: switch comparator to `fmaxf`.
