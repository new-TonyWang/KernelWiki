# Dynamic Parallelism -- Pitfalls

## P1: Attempting cudaDeviceSynchronize in CDP2

**Symptom:** Compilation error or undefined behavior when calling `cudaDeviceSynchronize()` from device code under CDP2.

**Detection:** Compile error with CUDA 12.0+ targeting CC 9.0+. On older devices, may compile but with unexpected behavior.

**Fix:** Use `cudaStreamTailLaunch` to process child grid results. Do not rely on device-side synchronization.

> Source: PG 4.18.3.1 -- "With the removal of cudaDeviceSynchronize(), it is no longer possible to access the modifications made by the threads in the child grid from the parent grid."

---

## P2: Parent Grid Trying to Read Child Grid Results Directly

**Symptom:** Race condition; parent reads stale data because child grid modifications are not visible to the parent.

**Detection:** Intermittent incorrect results that depend on timing. Works sometimes, fails other times.

**Fix:** Launch a tail kernel into `cudaStreamTailLaunch` to consume child results.

```cpp
// BAD: parent tries to read child results
__global__ void parent(float* data) {
    childKernel<<<...>>>(data);
    // data[tid] is NOT guaranteed to reflect child modifications
    float result = data[threadIdx.x];  // RACE!
}

// GOOD: use tail launch
__global__ void parent(float* data) {
    childKernel<<<...>>>(data);
    tailKernel<<<..., 0, cudaStreamTailLaunch>>>(data);
}
```

> Source: PG 4.18.3.1 -- "the only way to access the modifications made by the threads in the child grid before the parent grid exits is via a kernel launched into the cudaStreamTailLaunch stream."

---

## P3: Excessive CDP Launches Causing Overhead

**Symptom:** Performance is worse with CDP than with a flat single-kernel approach due to per-launch overhead.

**Detection:** Nsight Systems shows many very short child kernels with significant gaps between them.

**Fix:**
1. Batch work so child grids are larger.
2. Consider persistent kernel or work-stealing patterns instead of CDP for fine-grained work.
3. Set a minimum work threshold before launching a child grid.

> Source: PG 4.18 -- child grid launches have overhead that must be amortized.

---

## P4: Passing Shared or Local Memory Pointers to Child Grids

**Symptom:** Child grid accesses invalid memory; crash or undefined behavior.

**Detection:** `cudaErrorIllegalAddress` from child kernel. Address sanitizer flags.

**Fix:** Only pass global memory pointers to child grids. Shared and local memory are not accessible across grid boundaries.

> Source: PG 4.18.3 -- "Child grids can never access the local or shared memory of parent grids."

---

## P5: Using CDP1 on CC 9.0+ Devices

**Symptom:** Compilation failure or runtime error when using CDP1 APIs on Hopper or later.

**Detection:** Error when compiling with `-DCUDA_FORCE_CDP1_IF_SUPPORTED` for CC 9.0+.

**Fix:** Migrate to CDP2 (default since CUDA 12.0). Replace `cudaDeviceSynchronize()` with tail launch pattern.

> Source: PG 4.18.1.1 -- "CDP2 is the only version of CUDA dynamic parallelism available on devices of CC 9.0 and higher."
