---
title: Atomic Reduction - Pitfalls
status: draft
evidence_level: spec
applies_to_pattern_class:
- cuda-core
applies_to_ops:
- reduction
- histogram
- scan
- synchronization
requires_sm: '>=7.0'
single_kernel_useful: true
source:
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L3435-L3436
  excerpt: If an atomic instruction executed by a warp reads, modifies, and writes
    to the same location in global memory for more than one of the threads of the
    warp, each read/modify/write to that location occurs and they are all serialized.
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L3641-L3645
  excerpt: 'Use the narrowest scope possible: block-scoped atomics are much faster
    than system-scoped atomics. Prefer weaker orderings. Consider memory location:
    shared memory atomics are faster than global memory atomics.'
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L23252-L23295
  excerpt: Atomic functions perform read-modify-write operations on shared data, making
    them appear to execute in a single step.
id: pitfall-atomic-reduction
type: pitfall
vendor: nvidia
---
## P1. Every thread writing to a single global address

**Symptom**: Kernel is orders of magnitude below peak throughput; NCU shows most warps stalled on atomic instructions (`stall_long_scoreboard` or `stall_mio_throttle` dominated by a single address).

**How to detect**:
- In NCU, look at "Warp State Statistics": a large fraction of `stall_mio_throttle` or `stall_long_scoreboard` attributable to a specific instruction line.
- SASS shows a `RED` / `ATOM` instruction on every warp iteration pointing at the same pointer.

**Fix**: Apply S1 (hierarchical reduction). Reduce inside the warp with `__shfl_down_sync`, reduce across warps with shared memory, then commit **one atomic per block** to the global accumulator. This drops the number of contending atomics from `N` (threads) to `G` (blocks) -- typically a 100x-10000x reduction in contention.

**Source**: Programming Guide L3435-L3436.

---

## P2. Using system scope when device scope suffices

**Symptom**: Atomics are 2-10x slower than the device-scope baseline.

**How to detect**:
- `cuda::atomic<T, cuda::thread_scope_system>` was declared, or a `*_system` atomic builtin is in use (e.g., `atomicAdd_system`), but no CPU thread or peer GPU actually participates in the atomic sequence.
- SASS shows atomic instructions with scope qualifier `.sys` instead of `.gpu`.

**Fix**: Downgrade the scope. For counters touched only by kernel threads on the same GPU, use `cuda::thread_scope_device`. For intra-block counters, use `cuda::thread_scope_block`. The programming guide (L3641-L3645) is explicit that block scope is "much faster" than system scope.

**Source**: Programming Guide L3641-L3645.

---

## P3. Using seq_cst when relaxed ordering is sufficient

**Symptom**: Unnecessary memory fences around every atomic operation; reduced instruction-level throughput.

**How to detect**:
- SASS shows a `MEMBAR.GPU` or `FENCE.SC` instruction surrounding an atomic whose only requirement is atomicity.
- `cuda::atomic::fetch_add` was called without specifying a memory order (defaults to `seq_cst`).

**Fix**: Pass `cuda::memory_order_relaxed` explicitly on every fetch_add / fetch_sub / fetch_or whose correctness does not depend on ordering relative to other memory operations. Reserve `acquire` / `release` for producer-consumer flag handoff.

**Source**: Programming Guide L3641-L3645.

---

## P4. Non-atomic read-modify-write on a shared counter

**Symptom**: Non-deterministic results; "lost updates" appear in counters and histograms; results become correct when the grid is shrunk to one thread or one block.

**How to detect**:
- Plain C++ `counter++` / `counter += ...` on a location that multiple threads write.
- Static analysis: any multi-writer `__device__` / `__shared__` variable that is updated without an atomic wrapper is suspect.

**Fix**: Route all concurrent increments through an atomic API: `atomicAdd`, `cuda::atomic<T>::fetch_add`, or the legacy `atomic<Op>` family. Plain compound assignment is **not** atomic even for word-sized types.

**Source**: Programming Guide L23252-L23295.

---

## P5. FP16/BF16 atomics silently flushing denormals

**Symptom**: Small gradient magnitudes in FP16 accumulation disappear; model training diverges or stagnates compared to an FP32 accumulation reference.

**How to detect**:
- `atomicAdd` overloads for `__half` / `__nv_bfloat16` are in the critical path.
- Numerical accuracy drops specifically when magnitudes are near subnormal range (~5.96e-8 for FP16).

**Fix**: Use the PTX form `atom.add.noftz.f16` / `atom.add.noftz.bf16` (PTX ISA L19647-L19680) which preserves denormal values, or accumulate in FP32 on the block side and cast only at the final commit. Note: the default C++ atomicAdd for __half is a library function and may or may not emit the `.noftz` variant depending on toolchain version.

**Source**: PTX ISA L19647-L19680.

---

## P6. Expecting scope/ordering wins on a kernel that is not atomic-bound

**Symptom**: After switching `thread_scope_system` to `thread_scope_device`, or `seq_cst` to `relaxed`, NCU metrics (memory stalls, instruction counts) improve visibly, but wall-clock time barely moves.

**How to detect**:
- NCU "Speed Of Light" shows the kernel was memory-bound or compute-bound before the change, with atomics occupying a small fraction of issued instructions.
- Instruction-count deltas in NCU look impressive but end-to-end latency is flat.

**Fix**: **Always check wall-clock time in addition to NCU micro-metrics**. Scope / ordering optimizations only pay off when atomic contention is a material fraction of the critical path. If the kernel is memory-bound, focus on coalescing / vectorized-access first; if it is compute-bound, focus on ILP / warp-primitives.

**Source**: Legacy Level-3 sandbox observation, carried forward as anecdotal. See `legacy_sandbox_path` in `skill.md`.

---

## P7 (new, measured). Naive FP32 accumulation into a single scalar silently loses precision

**Symptom**: A reduction kernel that sums N > ~10^5 FP32 values into a single global FP32 scalar via per-thread `atomicAdd(&out, x[i])` returns a result that is several percent off the double-precision reference, despite the atomic operation itself being correct.

**How to detect**:
- Compare the kernel output against a host-side `double` sum of the same input. A relative error > 1e-3 while the kernel's logic is "obviously correct" points here.
- The error grows with N and with the dynamic range of individual samples. The H200 probe measured `rel_err = 5.03e-02` on N = 33,554,432 FP32 samples with magnitudes 0 to ~3e-3.

**Cause**: Once the running sum exceeds ~5x10^4, the FP32 scalar has only 24 bits of mantissa, so successive per-sample adds of magnitude ~10^-4 round to zero. This is a property of the **accumulator**, not the atomic operation.

**Fix**: Apply S1 or grid-stride + S1. The warp-lane partials stay small; the per-warp shmem slots stay modest; only the final per-block partials (already in the thousands, and only O(blocks) of them) accumulate into the global scalar. In the same H200 probe, `hierarchical_s1` returned `rel_err = 7.5e-05` and `grid_stride_s1` returned `rel_err = 1.4e-05`. If even S1's precision is insufficient, accumulate per-warp or per-block partials in `double` before the final atomic.

**Source**: H200 probe 2026-04-20 (see `skill.md` `## Measured Characteristics`). This finding is not in the upstream programming guide because the programming guide describes the *atomic*, not the *numerical behaviour of the application's accumulator*.

---

## P8. Benchmarking atomic reductions on tiny inputs

**Symptom**: Reported speedups are dramatic (10x-100x) but both baseline and optimized kernels are nowhere near peak utilization (e.g., memory SOL <1%, compute SOL <1%). The speedup is driven by launch / algorithmic overhead, not by the reduction technique under test.

**How to detect**:
- Input size is small enough that `nvprof` / NCU reports the kernel as launch-overhead-dominated.
- Both baseline and optimized kernels are slower than a `torch.sum` reference for the same problem size.

**Fix**: Size the benchmark input so the reduction path occupies
>= 50% of H200 HBM bandwidth for S1/S4 comparisons. Follow
`reasoning/hardware-microbench.md` for the full protocol (warmup / repeat / clock policy / baseline choice). Do not promote a skill to `evidence_level: measured` based on a launch-overhead-bound benchmark.

**Source**: Legacy Level-3 sandbox observation, carried forward as anecdotal. See `legacy_sandbox_path` in `skill.md`.
