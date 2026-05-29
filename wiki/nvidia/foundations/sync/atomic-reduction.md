---
title: Atomic Reduction Contention Control
status: verified
evidence_level: measured
cuda_version_tested: '12.9'
driver_version_tested: 570.124.06
toolchain: nvcc 12.9.86 + ptxas 12.9
measured_on: H200-SXM
applies_to_pattern_class:
- cuda-core
applies_to_ops:
- reduction
- histogram
- scan
- synchronization
requires_sm: '>=7.0'
requires_features:
- scoped-atomics
single_kernel_useful: true
source:
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L3435-L3436
  excerpt: If an atomic instruction executed by a warp reads, modifies, and writes
    to the same location in global memory for more than one of the threads of the
    warp, each read/modify/write to that location occurs and they are all serialized,
    but the order in which they occur is undefined.
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L3524-L3610
  excerpt: 'Scoped atomics combine two key concepts: Thread Scope defines which threads
    can observe the effect of the atomic operation; Memory Ordering defines the ordering
    constraints relative to other memory operations.'
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L3641-L3645
  excerpt: 'Use the narrowest scope possible: block-scoped atomics are much faster
    than system-scoped atomics. Prefer weaker orderings: use stronger orderings only
    when necessary for correctness. Consider memory location: shared memory atomics
    are faster than global memory atomics.'
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L23252-L23295
  excerpt: Atomic functions perform read-modify-write operations on shared data, making
    them appear to execute in a single step. Using the Extended CUDA C++ atomic functions
    provided by libcu++ is recommended for efficiency, safety, and portability.
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/parallel-thread-execution/cuda_parallel-thread-execution_index.html.md
  anchor: L19647-L19680
  excerpt: atom — atomic reduction operations for thread-to-thread communication.
    atom{.sem}{.scope}{.space}.op{.level::cache_hint}.type d, [a], b; .scope = { .cta,
    .cluster, .gpu, .sys }.
artifacts:
  code: sources/experience/hw-probes/atomic-reduction-contention/artifacts/atomic_reduction_probe.cu
  build: sources/experience/hw-probes/atomic-reduction-contention/artifacts/build.sh
  introspection: sources/experience/hw-probes/atomic-reduction-contention/artifacts/device.json
  profile: ''
related_apis:
- atomicAdd
- atomicCAS
- cuda::atomic
- cuda::atomic_ref
- __shfl_down_sync
related_skills:
- memory-ordering
- warp-primitives
- bank-conflict
id: skill-atomic-reduction
type: skill
vendor: nvidia
tags:
- cuda-cpp
applies_to:
- general
---
## What

An **atomic reduction** combines values from many threads into a single location using atomic read-modify-write operations (`atomicAdd`, `atomicCAS`, `cuda::atomic::fetch_add`, the PTX `atom` / `red` instructions, and their libcu++ equivalents). Every concurrent atomic against the same address is **serialized** by the hardware: the programming guide (L3435-L3436) states that "each read/modify/write to that location occurs and they are all serialized, but the order in which they occur is undefined."

The cost model for atomic reductions therefore has three knobs the author must actively control:

1. **Contention**: how many threads target the same address per unit time. N threads hitting one address means ~N serial transactions.
2. **Scope** (`cta` / `cluster` / `gpu` / `sys`): which level of the memory hierarchy has to agree on the ordering. A narrower scope resolves at a closer cache level and is cheaper.
3. **Memory ordering** (`relaxed` / `acquire` / `release` / `acq_rel` / `seq_cst`): what fence semantics surround the atomic. Weaker orderings emit fewer fence instructions.

The three knobs combine into one rule of thumb from PG 3.2.4.1.2 (L3641-L3645): **narrowest scope, weakest ordering, resolved as close to the compute as possible**.

## Why

Atomic reductions are the default pattern whenever multiple threads must accumulate into a shared counter (sums, histograms, argmax, bounding-box reductions, free-list heads). When implemented naively -- every thread issuing one global atomicAdd to a single address -- they become the dominant cost of the kernel, because each atomic serializes at the L2 cache.

The programming guide (L3435-L3436) is explicit that concurrent atomics from the same warp are serialized. On realistic reductions this serialization extends across all active warps and all active blocks, so a single-global-address reduction behaves like an O(threads) sequential chain.

The per-scope latency difference is documented directly in PG 3.2.4.1.2 (L3641-L3645): block scope is "much faster" than system scope, shared-memory atomics are faster than global-memory atomics, weaker orderings skip memory fences. The techniques in this skill are all applications of those three statements.

## When to use

Apply these techniques whenever a kernel:

- Uses `atomicAdd` / `atomicMin` / `atomicMax` / `atomicCAS` against a global-memory location touched by many threads.
- Uses `cuda::atomic` with `thread_scope_device` or `thread_scope_system` where a narrower scope would suffice.
- Builds a histogram, a counter, or a reduction whose final value lives in a single scalar (or a small fixed array like a 256-bucket histogram).
- Uses a "first thread commits the block result" pattern with a single global atomic per block -- this is already the good pattern, but the block-local side of the reduction must also be efficient (shared-memory atomics or warp shuffles, not all-threads-to-one global atomic).

### S1. Hierarchical reduction (warp -> block -> grid)

Replace one global atomic per thread with **one global atomic per block**. Reduce inside a warp with `__shfl_down_sync` (zero memory traffic), reduce across warps through shared memory, then have a single thread of the block commit the block's partial sum:

```cuda
__global__ void hierarchicalSum(const float* input, float* output, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (tid < N) ? input[tid] : 0.0f;

    // Warp-level reduction via shuffle (no shared memory, no atomics)
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);

    __shared__ float warpSums[32];          // at most 32 warps per block
    int lane   = threadIdx.x & 31;
    int warpId = threadIdx.x >> 5;
    if (lane == 0) warpSums[warpId] = val;
    __syncthreads();

    // One warp reduces the per-warp partial sums
    if (warpId == 0) {
        int nWarps = (blockDim.x + 31) >> 5;
        val = (lane < nWarps) ? warpSums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        if (lane == 0)
            atomicAdd(output, val);          // exactly one global atomic per block
    }
}
```

Traffic profile: `blockDim.x` threads per block, one `atomicAdd` per block. If the grid has `G` blocks, only `G` atomics contend for the destination address -- typically 10^2-10^4 instead of 10^6-10^8.

### S2. Narrowest scope that is correct

Choose the scope based on who must observe the atomic, not on "what feels safe":

- `cuda::thread_scope_block` -- intra-block counters (partial sums in shared memory, per-block histograms).
- `cuda::thread_scope_device` -- inter-block counters on the same GPU. This is what plain `atomicAdd` already uses.
- `cuda::thread_scope_system` -- only required when a CPU thread or another GPU participates in the atomic sequence.

```cuda
#include <cuda/atomic>

// Block scope: resolves within the SM's L1/shared-memory hierarchy
__shared__ cuda::atomic<int, cuda::thread_scope_block> blockCounter;

// Device scope: resolves at L2 (the default for atomicAdd())
__device__ cuda::atomic<int, cuda::thread_scope_device> deviceCounter;
```

The programming guide (L3641-L3645) states that block-scoped atomics are "much faster" than system-scoped atomics; in a single-kernel H200 context system scope is almost never what you want.

### S3. Weakest ordering that is correct

If the only requirement is "this counter is incremented atomically," use `memory_order_relaxed`. The stronger orderings (`acquire` / `release` / `acq_rel` / `seq_cst`) emit fence instructions that pessimize surrounding memory traffic.

```cuda
// Pure counter: atomicity only, no ordering required
counter.fetch_add(1, cuda::memory_order_relaxed);

// Producer-consumer flag: release on producer side, acquire on consumer
flag.store(true, cuda::memory_order_release);               // producer
while (!flag.load(cuda::memory_order_acquire)) { /* wait */ }  // consumer
```

For a deeper treatment of memory ordering semantics (what each order actually guarantees), see the sibling `memory-ordering` skill; this skill is scoped to the contention/throughput side.

### S4. Shared-memory atomics before global-memory atomics

When the reduction's output set is small and fits in shared memory (the classic case is a 256-bin histogram), do the entire block-local aggregation with shared-memory atomics and commit only the final per-bin result to global memory:

```cuda
__global__ void histogram256(const uint8_t* data, int n, unsigned* hist) {
    __shared__ unsigned sHist[256];
    if (threadIdx.x < 256) sHist[threadIdx.x] = 0;
    __syncthreads();

    // Block-local histogram lives in shared memory (one 32-bit atomic per sample)
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += gridDim.x * blockDim.x) {
        atomicAdd(&sHist[data[i]], 1u);
    }
    __syncthreads();

    // Commit to global memory: 256 atomics per block regardless of n
    if (threadIdx.x < 256)
        atomicAdd(&hist[threadIdx.x], sHist[threadIdx.x]);
}
```

PG 3.2.4.1.2 (L3641-L3645) states shared-memory atomics are faster than global-memory atomics; the kernel above exploits that by moving the per-sample atomic into shared memory and issuing only `256` global atomics per block instead of `n/blockDim.x`.

## When NOT to use

- **When the reduction is not actually atomic-bound.** If the kernel is memory-bandwidth-bound (load-heavy, atomics are <5% of issued instructions), optimizing the atomic path is invisible on wall-clock. Profile first: check whether `stall_long_scoreboard` or `stall_mio_throttle` on atomic instructions is actually material before rewriting.
- **When the output is large and dense.** If every thread writes to a distinct address (e.g., elementwise `out[tid] = ...`), no atomic is needed; use a plain store. Atomic instructions on an uncontended address are still more expensive than a plain store.
- **When correctness requires stronger ordering.** `relaxed` is not a default for producer-consumer flags, lock-free queues, or any pattern where one thread's write must be visible before another thread's read. See the `memory-ordering` skill.
- **For device-wide reductions with a library path available.** CUB's `DeviceReduce::Sum` / `BlockReduce` already implements S1+S4 with hardware-tuned block sizes; custom code is only justified when a library is unavailable or the reduction is fused with surrounding ops.

## Measured Characteristics

Measured on H200-SXM (sm_90a, CUDA 12.9, driver 570.124.06), summing `N = 33,554,432` floats (128 MB) to one scalar. Full record: [sources/experience/hw-probes/atomic-reduction-contention/2026-04-20-atomic-reduction.md](../../../sources/experience/hw-probes/atomic-reduction-contention/2026-04-20-atomic-reduction.md).

| Kernel             | Median ms | Eff. BW GB/s | DRAM SoL | Warp cyc/issue |
| ------------------ | --------: | -----------: | -------: | -------------: |
| `naive_atomic`     |   58.8769 |         2.28 |    0.05% |     **39,148** |
| `hierarchical_s1`  |    0.2376 |       564.89 |   11.74% |           65.6 |
| `grid_stride_s1`   | **0.0393**|  **3415.56** | **75.58%** |         67.4 |

- **S1 hierarchical beats naive by 247.8x**; with a grid-stride loop on top of S1, the factor rises to **1498x** and the kernel becomes essentially HBM-bound (75.58% DRAM Speed-of-Light out of H200's ~4.8 TB/s peak).
- The NCU signature of the atomic-contention pathology on `naive_atomic` is *Warp Cycles Per Issued Instruction ~ 39K* with DRAM SoL near 0, not a conventional memory-bound stall profile.
- Legacy KernelPilot-sandbox runs for this skill (`KernelPilot/experience/2026-04-04-optimization-synchronization-atomic-reduction-*`) were performed on ~4 KB inputs and reported underutilized memory/compute SoL; they are retained as anecdotal only, **not** as measured evidence. The H200 probe above supersedes them for all `evidence_level: measured` claims in this skill.

## Correctness finding: naive atomic reduction is also numerically wrong

The H200 probe uncovered a second reason to avoid the naive single-global-atomic pattern beyond performance. Summing the same input with `naive_atomic` produced a result 5.03% below the double-precision reference (`rel_err = 5.029e-02`, correctness FAIL). The root cause is **FP32 accumulation loss** in the single global scalar: once the running sum grows past ~5x10^4, subsequent per-sample adds of magnitude ~10^-4 to ~10^-3 fall below the 24-bit mantissa's resolution and are quantized to zero.

Hierarchical S1 cures this as a side effect: warp-lane partial sums stay in O(32) units, block partial sums stay in O(8,192) units, and only the final per-block partials (already in the thousands) land in the global scalar. Each accumulation stage therefore mixes values of comparable magnitude, preserving precision. In the same probe, `hierarchical_s1` returned rel_err 7.5e-05 and `grid_stride_s1` returned rel_err 1.4e-05.

**Takeaway**: the naive pattern is not merely slow -- it is **incorrect** for any sum larger than ~10^5 FP32 samples. This is the strongest possible argument for S1 in pure-float accumulation contexts; for higher-precision accumulation see the `hardware-microbench.md` follow-up on FP32-in-FP64 reduction.

## Principles

1. **Minimize contention before minimizing per-op cost.** Dropping the per-block atomic count from 1024 to 1 is a larger win than switching `seq_cst` to `relaxed`.
2. **Scope narrows the cache level; ordering narrows the fence.** Both are independent, and both contribute to the final cost.
3. **Shared memory is a first-class reduction staging area.** S1 and S4 compose: block-local reduction in shared memory, then one global atomic per block.
4. **Library first.** CUB `DeviceReduce` applies all four techniques under the hood; only hand-write when the library path is proven insufficient (library-fallback contract in `wiki/nvidia/operator-routing/`).

## Open questions

- Q1. On H200, what is the measured per-atomic cost at scope `cta` / `gpu` / `sys` for a `fetch_add<int>` with relaxed ordering? (Feeds `sources/experience/hw-probes/atomic-reduction/scope-latency/`.)
- Q2. Does `atom.add.v2.f32` (vectorized FP32 atomic, PTX ISA L19647-L19680) beat two scalar `atomicAdd` calls on H200, and at what alignment?
- Q3. For FP16 gradient accumulation, does `atom.add.noftz.f16` (PTX ISA L19647-L19680) preserve enough precision vs. accumulating in FP32 on the block side and committing in FP16 at block commit?

## Legacy references

- `legacy_sandbox_path`: `KernelPilot/experience/2026-04-04-optimization-synchronization-atomic-reduction-skill{1,2,3,4}.md`. Treat as anecdotal only; environment (toolchain / driver / clock policy) was not recorded, and input sizes were too small to exercise the reduction path.
