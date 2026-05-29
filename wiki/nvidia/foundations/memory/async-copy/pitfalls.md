---
title: Asynchronous Data Copies - Pitfalls
status: draft
evidence_level: spec
applies_to_pattern_class:
- cuda-core
applies_to_ops:
- elementwise
- reduction
- normalization
- stencil
- scan
requires_sm: '>=8.0'
single_kernel_useful: true
source:
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L11266-L11269
  excerpt: LDGSTS supports copying 4, 8, or 16 bytes. Copying 4 or 8 bytes always
    happens in the so called L1 ACCESS mode... copying 16-bytes enables the L1 BYPASS
    mode.
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L11267-L11268
  excerpt: The pointers need to be aligned to 4, 8, or 16 bytes depending on the size
    of the data being copied. Best performance is achieved when the alignment of both
    shared memory and global memory is 128 bytes.
- path: <path-removed>
  anchor: ldgsts-apis-and-modes
  excerpt: L1 BYPASS -- sizeof(datatype) and alignment must be 16 bytes. producer_acquire()
    and producer_commit() must be called from converged code.
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L11269
  excerpt: LDGSTS must provide a signal when the operation is complete... if you use
    LDGSTS to prefetch some data that will be shared with other threads, a __syncthreads()
    is necessary after synchronizing with the LDGSTS completion mechanism.
id: pitfall-async-copy
type: pitfall
vendor: nvidia
---
## P1: Alignment requirements for L1 BYPASS mode

**Symptom**: LDGSTS falls back to L1 ACCESS mode (4-byte loads through L1), adding unnecessary L1 cache pollution and missing the potential bandwidth benefit of L1 BYPASS.

**Cause**: L1 BYPASS mode requires all of the following (programming guide L11266, GTC25-S72683 slide interval_0093):
- Copy size is exactly 16 bytes.
- Both source (global memory) and destination (shared memory) pointers are 16-byte aligned.
- The copy is expressed through `cuda::aligned_size_t<16>(size)` or an equivalent 16-byte type.

If any requirement is violated, the hardware silently falls back to L1 ACCESS mode.

**Example**:
```cuda
// BAD: 4-byte copy -- always L1 ACCESS mode
cuda::memcpy_async(&smem[tid], &gmem[idx], sizeof(float), pipe);

// GOOD: 16-byte copy with alignment hint -- L1 BYPASS mode
// Only BLOCK_DIM/4 threads each copy 16 bytes
if (tid < BLOCK_DIM / 4) {
    cuda::memcpy_async(&smem[tid * 4], &gmem[offset + tid * 4],
                       cuda::aligned_size_t<16>(4 * sizeof(float)), pipe);
}
```

**Fix**: Restructure the copy so that each participating thread copies 16 bytes. Use `memcpy_threads = BLOCK_DIM / 4` threads, each copying 4 floats (16 bytes). Ensure global memory allocations are 128-byte aligned (cudaMalloc guarantees at least 256-byte alignment). Ensure shared memory arrays start at 16-byte boundaries.

**Detection**: Check the SASS disassembly for `LDGSTS.BYPASS` vs `LDGSTS` instructions. Alternatively, compare L1 hit rates in Nsight Compute: L1 BYPASS mode should show no L1 hits for the async copy traffic.

## P2: `producer_commit` from diverged code

**Symptom**: Excessive `LDGDEPBAR` (dependency barrier) instructions in the compiled SASS, or pipeline stages that complete out of order.

**Cause**: `pipe.producer_commit()` (and `__pipeline_commit()`) is a warp-level instruction. If it is called inside a divergent branch, the compiler emits one commit per conditional path, because it cannot prove that all threads in the warp take the same path (GTC25-S72683, slide interval_0119).

**Example**:
```cuda
// BAD: commit inside conditional code
if (idx < N) {
    cuda::memcpy_async(&smem[tid], &gmem[idx], sizeof(float), pipe);
    pipe.producer_commit();  // compiler may emit multiple commits
}

// GOOD: commit in converged code outside the conditional
if (idx < N) {
    cuda::memcpy_async(&smem[tid], &gmem[idx], sizeof(float), pipe);
}
pipe.producer_commit();  // single commit for the entire warp
```

**Fix**: Always place `producer_acquire()` and `producer_commit()` (or `__pipeline_commit()`) in code paths where the entire warp converges. Move them outside of `if` statements.

## P3: NUM_STAGES not a compile-time constant

**Symptom**: Pipeline operations compile but performance is poor; the compiler generates unnecessary bookkeeping instructions.

**Cause**: `cuda::pipeline_consumer_wait_prior<N>(pipe)` requires `N` to be a compile-time constant (template parameter). If `NUM_STAGES` is a runtime variable, the template cannot be instantiated and the code either fails to compile or falls back to a less efficient wait mechanism (GTC25-S72683, slide interval_0130).

**Example**:
```cuda
// BAD: runtime stage count
int num_stages = compute_stages();
cuda::pipeline_consumer_wait_prior<num_stages - 1>(pipe);  // compile error

// GOOD: compile-time constant
constexpr int NUM_STAGES = 2;
cuda::pipeline_consumer_wait_prior<NUM_STAGES - 1>(pipe);  // OK
```

**Fix**: Define `NUM_STAGES` as a `constexpr` or template parameter. If different stage counts are needed for different shapes, use a `switch`-`case` or `if constexpr` dispatch over a small set of values.

## P4: Missing `__syncthreads()` in producer-consumer pattern

**Symptom**: Threads read stale or partially-written data from shared memory. Correctness failures that are non-deterministic.

**Cause**: When only a subset of threads performs the async copies (e.g., `BLOCK_DIM/4` threads for 16-byte L1 BYPASS copies) but all threads consume the data, the pipeline wait only guarantees that the *copying threads* see the data as ready. Other threads in the block need an explicit `__syncthreads()` after the pipeline wait to ensure visibility (programming guide L11269, GTC25-S72683 slide interval_0142).

**Example**:
```cuda
const int memcpy_threads = BLOCK_DIM / 4;

// Producer: only first memcpy_threads threads copy
if (tid < memcpy_threads) {
    cuda::memcpy_async(&smem[tid * 4], &gmem[offset + tid * 4],
                       cuda::aligned_size_t<16>(4 * sizeof(float)), pipe);
}
pipe.producer_commit();

// BAD: missing __syncthreads() -- non-copying threads may see stale data
cuda::pipeline_consumer_wait_prior<NUM_STAGES - 1>(pipe);
float val = smem[tid];  // thread tid >= memcpy_threads reads stale data

// GOOD: add __syncthreads() after the wait
cuda::pipeline_consumer_wait_prior<NUM_STAGES - 1>(pipe);
__syncthreads();
float val = smem[tid];  // all threads see the correct data
```

**Fix**: Insert `__syncthreads()` between the pipeline wait and the compute step whenever a **subset** of threads performs the copies. If all threads copy their own element (the simpler pattern), this is not needed because each thread only reads data it copied.

## P5: TMA peeling loop without `invoke_one`

**Symptom**: TMA instructions execute sequentially across all active threads in a warp instead of once, wasting bandwidth and cycles.

**Cause**: TMA is warp-uniform: it should be issued by exactly one thread per warp. When wrapped in `if (threadIdx.x == 0) { ... }`, the compiler does not know that only one thread enters the branch, so it emits a **peeling loop** that iterates the TMA instruction for each active lane (GTC25-S72683, slides interval_0193 and interval_0196).

**Example**:
```cuda
// BAD: compiler emits peeling loop
if (threadIdx.x == 0) {
    cuda::memcpy_async(smem_ptr, global_ptr,
                       cuda::aligned_size_t<16>(num_bytes), bar);
}

// GOOD: invoke_one tells the compiler only one thread executes
namespace cg = cooperative_groups;
cg::invoke_one(cg::coalesced_threads(), [&] {
    cuda::memcpy_async(smem_ptr, global_ptr,
                       cuda::aligned_size_t<16>(num_bytes), bar);
});
```

**Fix**: Use `cooperative_groups::invoke_one(cg::coalesced_threads(), ...)` to wrap TMA instructions. This is a Hopper+ concern and applies when using TMA 1D/ND, not LDGSTS.

## P6: Applying async copies to trivially simple kernels

**Symptom**: The async-copy version of a kernel is **slower** than the vanilla version.

**Cause**: For very simple compute (e.g., `c[i] = a[i] * b[i]`), the overhead of shared memory staging -- `commit`, `wait_prior`, `release` instructions plus SMEM reads -- exceeds the latency-hiding benefit. Our H200 probe confirmed this: the 2-stage LDGSTS kernel was ~10% slower than vanilla for `a*b` compute.

GTC25-S72683 explicitly states: "kernels that read GMEM -> write SMEM -> compute -> write back see only a minor improvement just by switching to async copies -- unless they can batch loads. The big wins are on iterative kernels that can prefetch data for future iterations, especially low-occupancy compute-heavy kernels."

**Fix**: Before adding async copies, check:
1. Is the kernel actually bottlenecked by "Stall Long Scoreboard" in Nsight Compute warp-state statistics? If not, async copies will not help.
2. Is the compute intensity high enough to amortize the staging overhead? GTC data shows that `sqrt` chains give 1.3x uplift, but trivial `a*b` gives no benefit.
3. Can the kernel genuinely prefetch -- i.e., does it have at least 2 iterations where future data can be loaded while current data is consumed?

## P7: Forgetting to drain the pipeline epilogue

**Symptom**: The last 1-2 stages' worth of data is never consumed, producing incorrect results for the tail elements.

**Cause**: The prologue fills `NUM_STAGES` stages, and the main loop prefetches one stage ahead while consuming one. At the end of the input, there are `NUM_STAGES - 1` stages still in flight. If the loop terminates without waiting for and consuming these remaining stages, the tail data is lost.

**Fix**: After the main loop, add an epilogue that waits for and consumes the remaining stages:

```cuda
// Epilogue: drain remaining stages
for (int s = 0; s < NUM_STAGES - 1 && remaining > 0; ++s) {
    __pipeline_wait_prior(0);  // wait for all
    // consume stage
    __pipeline_commit();       // keep pipeline counter balanced
}
```

Alternatively, structure the main loop to include both the prologue and epilogue stages in its iteration count.
