---
status: unverified
verified_date: null
verified_on: null
performance_data:
  before: null
  after: null
source:
  - Programming Guide, Section 5.4.8.3 (Low-Level Load and Store Functions)
  - Programming Guide, Section 2.2.3.6 (Caches)
  - Best Practices Guide, Section 12.2 (Memory Instructions)
cross_ref:
  - PTX ISA, cache operators (.ca, .cg, .cs, .cv, .lu, .wb, .wt)
related_apis:
  - __ldg, __ldcg, __ldca, __ldcs, __ldlu, __ldcv, __stcg, __stcs, __stwb, __stwt, ld.global.nc, prefetch, .ca, .cg, .cs
related_experience: []
unlocks:
  - "memory/l2-cache-control: cache hints complement stream-level L2 persistence"
  - "memory/data-prefetch: cache hints control where prefetched data lands"
conflicts_with: []
---

# Cache Load Hints

## Skill 1: Use __ldg() for Read-Only Data via Texture Cache Path

### When to Use
- Data is read-only and not modified by the kernel
- Want to utilize the read-only (texture) cache path for additional bandwidth
- On older architectures (pre-CC 5.0), this was more critical; on modern GPUs it provides non-coherent cache path

### How to Apply
1. Replace `data[idx]` with `__ldg(&data[idx])`
2. The compiler may already generate `ld.global.nc` for `const __restrict__` pointers
3. Supports all fundamental types, vector types, and half/bf16

### Code Template
```cuda
__global__ void readonly_kernel(const float* __restrict__ data,
                                float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Explicit read-only cache load
        float val = __ldg(&data[idx]);
        out[idx] = val * 2.0f;
    }
}
```

### Source
Programming Guide, Section 5.4.8.3 (Low-Level Load and Store Functions)

## Skill 2: Use __ldcg() to Cache Only in L2 (Bypass L1)

### When to Use
- Data will not be reused by the same SM but may be reused by other SMs
- Want to preserve L1 cache for more frequently accessed data

### How to Apply
1. Replace load with `__ldcg(&data[idx])`
2. Maps to PTX `ld.global.cg` (cache-global, L2 only)
3. Frees L1 capacity for other data

### Code Template
```cuda
__global__ void l2_only_load(const float* data, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = __ldcg(&data[idx]);  // Cache in L2 only
        out[idx] = val + 1.0f;
    }
}
```

### Source
Programming Guide, Section 5.4.8.3 (Low-Level Load and Store Functions)

## Skill 3: Use __ldcs() for Streaming (One-Time Use) Data

### When to Use
- Data is accessed once and not expected to be reused
- Want to hint the cache to evict this data first

### How to Apply
1. Replace load with `__ldcs(&data[idx])`
2. Maps to PTX `ld.global.cs` (cache-streaming, evict-first policy)
3. Useful for large sequential scans where cache pollution is a concern

### Code Template
```cuda
__global__ void streaming_load(const float* data, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = __ldcs(&data[idx]);  // Streaming access
        out[idx] = val * 3.0f;
    }
}
```

### Source
Programming Guide, Section 5.4.8.3 (Low-Level Load and Store Functions)

## Skill 4: Use __stwb() and __stwt() for Store Cache Control

### When to Use
- __stwb(): Default write-back; data may be read back by same SM
- __stwt(): Write-through to system memory; useful for data consumed by CPU or other GPU

### How to Apply
1. Replace `data[idx] = val` with `__stwb(&data[idx], val)` or `__stwt(&data[idx], val)`
2. __stwt maps to PTX `st.global.wt` (write-through)
3. __stcg maps to PTX `st.global.cg` (L2-only store)

### Code Template
```cuda
__global__ void write_through_kernel(float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float result = compute_something(idx);
        __stwt(&out[idx], result);  // Write-through to system memory
    }
}
```

### Source
Programming Guide, Section 5.4.8.3 (Low-Level Load and Store Functions)

## Skill 5: Configure L1/Shared Memory Split

### When to Use
- Kernel benefits from larger L1 cache (no shared memory use)
- Or kernel needs maximum shared memory (heavy shared memory use)

### How to Apply
1. Use `cudaFuncSetCacheConfig()` to prefer L1 or shared memory
2. This is a hint; the runtime may override it

### Code Template
```cuda
// Prefer more L1 cache (no shared memory in this kernel)
cudaFuncSetCacheConfig(myKernel, cudaFuncCachePreferL1);

// Or prefer more shared memory
cudaFuncSetCacheConfig(myGemmKernel, cudaFuncCachePreferShared);

// Or equal split
cudaFuncSetCacheConfig(myKernel, cudaFuncCachePreferEqual);
```

### Source
Programming Guide, Section 3.2.6 (Configuring L1/Shared Memory Balance)

## Skill 6: Use __ldlu() for Last-Use Eviction Hint

### When to Use
- Data is loaded once and will not be needed again
- Want to explicitly mark data for eviction after use

### How to Apply
1. Replace load with `__ldlu(&data[idx])`
2. Maps to PTX `.lu` (last-use) cache operator
3. Hints the cache to evict this line after the load

### Code Template
```cuda
__global__ void last_use_load(const float* temp_data, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = __ldlu(&temp_data[idx]);  // Evict after use
        out[idx] = val + 1.0f;
    }
}
```

### Source
Programming Guide, Section 5.4.8.3 (Low-Level Load and Store Functions)

## Cascading Opportunities (unlocks)
After applying cache hints:
1. Check memory/l2-cache-control -- stream-level L2 policy complements per-load hints
2. Check memory/data-prefetch -- combine prefetch with appropriate cache level hints

## Conflicts
- None significant; cache hints are advisory and non-conflicting

## Principles
- **P1**: .ca (cache-all) is the default for loads: caches in both L1 and L2.
- **P2**: .cg (cache-global) bypasses L1, caches only in L2. Use for data reused across SMs but not within an SM.
- **P3**: .cs (cache-streaming) marks data as evict-first. Use for large sequential scans.
- **P4**: __ldg() uses the non-coherent texture cache path (ld.global.nc). On modern GPUs, `const __restrict__` may generate this automatically.
- **P5**: Store hints (.wb write-back, .wt write-through) control when data becomes visible to other processors.

## Open Questions (for Level 3 verification)
- Q1: On Hopper, does __ldcg() provide measurable L1 capacity improvement for mixed-access kernels?
- Q2: Does __ldg() still provide benefit over default loads with `const __restrict__` on sm_80+?
- Q3: What is the performance impact of __ldcs() on a streaming reduction kernel?
