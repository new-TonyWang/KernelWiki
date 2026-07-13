# Occupancy Tuning -- Skills

```yaml
status: draft
source:
  - "Best Practices Guide 11.1 (Occupancy)"
  - "Best Practices Guide 11.1.1 (Calculating Occupancy)"
  - "Best Practices Guide 11.3 (Thread and Block Heuristics)"
  - "Best Practices Guide 11.4 (Effects of Shared Memory)"
  - "Programming Guide 2.2.7 (Kernel Launch and Occupancy)"
  - "Programming Guide 3.2.2.2 (Hardware Multithreading)"
  - "Programming Guide 5.4.3.2 (Launch Bounds)"
cross_ref:
  - optimization/memory/register-pressure
  - optimization/memory/shared-memory-cache
  - optimization/compute/instruction-level-parallelism
related_apis:
  - cudaOccupancyMaxActiveBlocksPerMultiprocessor
  - cudaOccupancyMaxPotentialBlockSize
  - cudaFuncSetAttribute
  - __launch_bounds__
  - setmaxnreg
unlocks:
  - Higher warp-level latency hiding through more active warps
  - Better utilization of SM resources
conflicts_with:
  - optimization/memory/register-pressure  # Higher occupancy may force register spilling
  - optimization/memory/shared-memory-cache  # More shared memory per block reduces occupancy
```

---

## S1: Use the Occupancy API for Dynamic Block Size Selection

**When to Use:** The optimal block size is not known at compile time, or the kernel is launched on devices with varying compute capabilities.

**How to Apply:**
1. Call `cudaOccupancyMaxPotentialBlockSize` to get the recommended block size for maximum occupancy.
2. Derive the grid size from the total work and the recommended block size.
3. Launch the kernel with these values.

**Code Template:**
```cpp
int minGridSize, blockSize;
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, myKernel, 0, 0);

int gridSize = (numElements + blockSize - 1) / blockSize;
myKernel<<<gridSize, blockSize>>>(...);
```

> Source: BP 11.1.1 -- "An application can also use the Occupancy API from the CUDA Runtime, e.g. cudaOccupancyMaxActiveBlocksPerMultiprocessor, to dynamically select launch configurations based on runtime parameters."

---

## S2: Apply __launch_bounds__ for Forward Compatibility

**When to Use:** Every production kernel should specify launch bounds to guarantee at least one block can run per SM, preventing "too many resources requested for launch" errors on future architectures.

**How to Apply:**
1. Annotate the kernel with `__launch_bounds__(maxThreadsPerBlock)` at minimum.
2. Optionally provide `minBlocksPerMultiprocessor` to guide the compiler on register allocation.
3. Profile to determine the right `minBlocksPerMultiprocessor` value.

**Code Template:**
```cpp
// Single argument: guarantees kernel can launch with up to 256 threads
__global__ void __launch_bounds__(256)
myKernel(float* data) {
    // kernel body
}

// Two arguments: hint compiler to target at least 4 blocks per SM
__global__ void __launch_bounds__(256, 4)
myKernel(float* data) {
    // compiler will limit registers so >=4 blocks can be resident
}
```

> Source: BP 11.1 -- "developers should include the single argument __launch_bounds__(maxThreadsPerBlock) which specifies the largest block size that the kernel will be launched with. Failure to do so could lead to 'too many resources requested for launch' errors."

---

## S3: Choose Block Size as Multiple of Warp Size

**When to Use:** Always -- this is a fundamental heuristic for all CUDA kernels.

**How to Apply:**
1. Set threads per block to a multiple of 32 (warp size).
2. Start experimentation in the 128--256 range.
3. Use at least 64 threads per block; prefer multiple smaller blocks per SM over one large block if the kernel calls `__syncthreads()` frequently.

**Code Template:**
```cpp
// Good: multiple of 32, in the 128-256 sweet spot
myKernel<<<numBlocks, 256>>>(...);

// Bad: wastes computation on under-populated warps
myKernel<<<numBlocks, 100>>>(...);  // last warp has only 4 active threads
```

> Source: BP 11.3 -- "Threads per block should be a multiple of warp size to avoid wasting computation on under-populated warps and to facilitate coalescing."

---

## S4: Trade Occupancy for Per-Thread Resources via ILP

**When to Use:** When the kernel has high instruction-level parallelism (ILP) and profiling shows that reducing occupancy to gain more registers per thread improves performance.

**How to Apply:**
1. Profile baseline occupancy and register usage with `--ptxas-options=-v`.
2. Experiment with `__launch_bounds__(maxThreads, minBlocks)` where `minBlocks` is lower, giving the compiler freedom to use more registers.
3. Verify that the performance improvement from fewer register spills outweighs the loss in latency hiding.

**Code Template:**
```cpp
// Allow compiler to use more registers (fewer blocks per SM)
__global__ void __launch_bounds__(128, 2)
computeIntensive(float* __restrict__ out, const float* __restrict__ in) {
    // High-ILP code with many independent operations
    float a = in[idx];
    float b = in[idx + stride];
    float c = in[idx + 2*stride];
    float d = in[idx + 3*stride];
    // ... all independent computation chains
    out[idx] = a + b + c + d;
}
```

> Source: BP 11.3 -- "A lower occupancy kernel will have more registers available per thread than a higher occupancy kernel, which may result in less register spilling to local memory; in particular, with a high degree of exposed instruction-level parallelism (ILP) it is, in some cases, possible to fully cover latency with a low occupancy."

---

## S5: Probe Occupancy Sensitivity with Dynamic Shared Memory

**When to Use:** To determine whether a kernel is sensitive to occupancy without modifying the kernel code.

**How to Apply:**
1. Launch the kernel with increasing amounts of dynamic shared memory (third parameter of `<<<...>>>`).
2. More dynamic shared memory reduces occupancy by consuming more of the SM shared memory budget.
3. Plot performance vs. occupancy to find the diminishing-returns threshold.

**Code Template:**
```cpp
for (int extraSmem = 0; extraSmem < maxSmemPerBlock; extraSmem += 1024) {
    myKernel<<<gridSize, blockSize, extraSmem, stream>>>(...);
    // measure and record performance
}
// Find the knee in the performance curve
```

> Source: BP 11.4 -- "A useful technique to determine the sensitivity of performance to occupancy is through experimentation with the amount of dynamically allocated shared memory, as specified in the third parameter of the execution configuration."

---

## S6: Query and Respect SM Resource Limits

**When to Use:** When designing kernels that need to maximize SM utilization across different GPU architectures.

**How to Apply:**
1. Query device properties: `maxBlocksPerMultiProcessor`, `sharedMemPerMultiprocessor`, `regsPerMultiprocessor`, `maxThreadsPerMultiProcessor`.
2. Calculate the limiting resource for your kernel (registers, shared memory, or threads).
3. Adjust block size and resource usage to maximize occupancy within the tightest constraint.

**Code Template:**
```cpp
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);

// Query kernel resource usage
cudaFuncAttributes attrs;
cudaFuncGetAttributes(&attrs, myKernel);
int regsPerThread = attrs.numRegs;

// Calculate occupancy
int maxBlocksPerSM;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM, myKernel, blockSize, dynamicSmem);

float occupancy = (float)(maxBlocksPerSM * blockSize) / prop.maxThreadsPerMultiProcessor;
printf("Occupancy: %.1f%%\n", occupancy * 100);
```

> Source: PG 2.2.7 -- "The occupancy of a CUDA kernel is the ratio of the number of active warps to the maximum number of active warps supported by the SM."

---

## Cascading Opportunities

- After tuning occupancy, check `register-pressure` -- reducing register count increases occupancy but may cause spilling.
- Shared memory usage directly affects occupancy; combine with `shared-memory-cache` tuning.
- Higher occupancy enables better latency hiding, which benefits `data-prefetch` multi-buffering patterns.

## Conflicts

- **register-pressure:** Higher occupancy demands fewer registers per thread, potentially increasing spills to local memory.
- **shared-memory-cache:** Larger shared memory allocations per block reduce the number of concurrent blocks per SM.
- **ILP strategy:** Some kernels perform better at lower occupancy with more registers per thread when they have high ILP.

## Principles

1. **Occupancy is necessary but not sufficient:** Low occupancy always hurts, but increasing occupancy beyond a threshold may not help (BP 11.1).
2. **Register allocation granularity matters:** Registers are rounded up to the nearest 256 per warp; small changes in register count can cause large occupancy drops (BP 11.1.1).
3. **Block count > SM count:** The grid should have many more blocks than SMs to keep the GPU fully occupied as blocks complete at different rates (BP 11.3).

## Open Questions

- Q1: How does `setmaxnreg` (Hopper+) interact with `__launch_bounds__` for warpgroup-level register tuning?
- Q2: What is the optimal occupancy threshold for memory-bound vs. compute-bound kernels on Blackwell (CC 10.0)?
- Q3: Does `cudaOccupancyMaxPotentialBlockSize` account for cluster-level constraints on CC 9.0+?
