# Reduction Pattern -- Skills

## When to Apply
- Sum, max, min, mean, norm, dot product over one or more dimensions
- Partial reductions (reduce one dimension of a multi-dimensional tensor)
- Any operation that collapses N elements to 1 (or fewer) using an associative operator

## Algorithm-Level Design

### Three-Level Reduction Architecture
A well-optimized reduction kernel uses three hierarchy levels:
1. **Warp-level** (32 threads): shuffle-based reduction, zero SMEM overhead
2. **Block-level** (multiple warps): warp results combined via shared memory
3. **Grid-level** (multiple blocks): block results combined via atomics or a second kernel pass

## Skill 1: Warp-Level Reduction with Shuffle
- Use `__shfl_down_sync` (tree reduction) or `__shfl_xor_sync` (butterfly reduction)
- Tree reduction for sum: 5 steps for 32 threads (log2(32) = 5)
- Butterfly pattern: each step XORs lane indices, good for all-reduce (every lane gets result)

### Tree reduction pattern:
```cuda
// val holds each thread's partial result
for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
}
// Thread 0 now holds the warp reduction result
```

### Butterfly all-reduce pattern:
```cuda
for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, offset);
}
// All threads now hold the same reduced value
```

### Hardware warp reduction (SM80+):
- PTX `redux.sync.add.s32` performs warp-level reduction in hardware
- Faster than shuffle tree but limited to 32-bit integer operations (.add/.min/.max/.and/.or/.xor)
- Use for integer reductions; for FP reductions, shuffle is still needed

## Skill 2: Block-Level Reduction via Shared Memory
- Each warp reduces internally using shuffles, then warp-0 thread writes result to SMEM
- After `__syncthreads()`, first warp loads all partial results from SMEM and does a final warp-level reduction

### Pattern (for blockDim.x threads):
```cuda
__shared__ float warp_results[32]; // max 32 warps per block

// Step 1: warp-level reduction
float val = thread_data;
for (int off = 16; off > 0; off >>= 1)
    val += __shfl_down_sync(0xffffffff, val, off);

// Step 2: warp leaders write to SMEM
if (threadIdx.x % 32 == 0)
    warp_results[threadIdx.x / 32] = val;
__syncthreads();

// Step 3: first warp reduces warp results
if (threadIdx.x < blockDim.x / 32) {
    val = warp_results[threadIdx.x];
    for (int off = 16; off > 0; off >>= 1)
        val += __shfl_down_sync(0xffffffff, val, off);
}
```

## Skill 3: Grid-Level Reduction Strategies

### Strategy A: Atomic accumulation
- Each block atomically adds its partial result to a global accumulator
- `atomicAdd` for FP32/FP64 (native); for FP16/BF16 use `atomicAdd` on SM80+
- Pro: single kernel launch. Con: atomic contention if many blocks reduce to same location
- Best for: reducing a large tensor to a single scalar

### Strategy B: Two-pass reduction
- Pass 1: each block writes partial result to a temporary buffer
- Pass 2: a single block reduces all partial results
- Pro: no atomic contention. Con: two kernel launches
- Best for: very large reductions where atomic contention is severe

### Strategy C: Cooperative groups grid-level sync
- Use `cooperative_groups::this_grid().sync()` for grid-wide barrier
- Allows single-kernel two-pass without separate launch
- Con: requires cooperative launch, limits grid size to max occupancy

## Skill 4: Vectorized Loads for Bandwidth-Bound Reductions
- Reduction is almost always memory-bandwidth bound (low arithmetic intensity)
- Use vectorized loads (`float4`, `int4`, `__half2`) to maximize bandwidth utilization
- Load 128 bits at a time, reduce locally, then do the hierarchical reduction
- Requires input to be properly aligned (128-bit alignment for float4)

## Skill 5: Sequential Addressing to Avoid Warp Divergence
- Classic mistake: interleaved addressing causes divergent branches in early tree reduction
- Sequential addressing: active threads are always contiguous in each reduction step
- With shuffle-based reduction, this is handled automatically (no explicit addressing needed)

## Skill 6: Partial Reductions (Reduce Along One Axis)
- Reducing rows of a matrix: each warp/block handles one or more rows
- Reducing columns: more challenging due to strided access
  - Transpose first, then reduce rows (layout-transform + row reduction)
  - Or use atomics: each block processes a tile, atomically accumulates column sums
  - Or use warp shuffle in the column direction with proper indexing

## Skill 7: cuBLAS Reduction Operations
- `cublas<t>nrm2`: Euclidean norm (reduction to scalar)
- `cublas<t>dot`: Dot product (reduction of elementwise product)
- `cublas<t>asum`: Sum of absolute values
- `cublasI<t>amax` / `cublasI<t>amin`: Index of max/min element
- Use `cublasNrm2Ex` / `cublasDotEx` for mixed-precision reductions

## Cross-References
- optimization/compute/warp-primitives -- shuffle instruction details
- optimization/synchronization/atomic-reduction -- atomic operation optimization
- optimization/memory/coalescing -- coalesced loads for input data
- optimization/memory/vectorized-access -- vector loads for bandwidth
- pattern/normalization -- reduction is a sub-operation of normalization
