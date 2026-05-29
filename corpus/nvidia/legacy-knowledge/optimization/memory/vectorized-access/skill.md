---
status: unverified
verified_date: null
verified_on: null
performance_data:
  before: null
  after: null
source:
  - Best Practices Guide, Section 10.2.1 (Coalesced Access)
  - Best Practices Guide, Section 10.2.3.4 (Async Copy)
  - Programming Guide, Section 2.2.4.1 (Coalesced Global Memory Access)
cross_ref:
  - PTX ISA, ld.global.v2/v4, st.global.v2/v4
related_apis:
  - ld.global.v2, ld.global.v4, st.global.v2, st.global.v4, atom.add.vec.f32
related_experience: []
unlocks:
  - "memory/coalescing: vectorized loads widen the effective transaction, reducing total transaction count"
  - "memory/shared-memory-cache: vectorized loads from global to shared improve fill bandwidth"
  - "memory/data-prefetch: 16-byte copies enable L1 bypass mode in async copy"
conflicts_with:
  - "memory/register-pressure: vector types (float4) consume multiple registers per thread"
---

# Vectorized Access

## Skill 1: Use float4/int4 for 128-bit Coalesced Loads

### When to Use
- Kernel is memory-bandwidth bound on global memory
- Each thread processes multiple contiguous elements (e.g., 4 floats)
- Data arrays are 16-byte aligned (guaranteed by cudaMalloc)

### How to Apply
1. Cast the global pointer to a vector type pointer (float4*, int4*)
2. Each thread loads one vector element instead of four scalar elements
3. Ensure thread count times vector_width covers the array length

### Code Template
```cuda
__global__ void vectorized_copy(const float* __restrict__ in,
                                float* __restrict__ out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_n = n / 4;
    if (tid < vec_n) {
        float4 val = reinterpret_cast<const float4*>(in)[tid];
        reinterpret_cast<float4*>(out)[tid] = val;
    }
}
```

### Source
Best Practices Guide, Section 10.2.1 (Coalesced Access)

## Skill 2: Use float2/int2 for 64-bit Vectorized Access

### When to Use
- Data type is half-precision (half2) or smaller types packed into 8 bytes
- 16-byte alignment cannot be guaranteed for sub-arrays
- Processing pairs of elements per thread

### How to Apply
1. Cast pointer to float2* or int2*
2. Load one 64-bit word per thread
3. Unpack into scalar components for computation

### Code Template
```cuda
__global__ void vec2_add(const float* __restrict__ a,
                         const float* __restrict__ b,
                         float* __restrict__ c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_n = n / 2;
    if (tid < vec_n) {
        float2 va = reinterpret_cast<const float2*>(a)[tid];
        float2 vb = reinterpret_cast<const float2*>(b)[tid];
        float2 vc;
        vc.x = va.x + vb.x;
        vc.y = va.y + vb.y;
        reinterpret_cast<float2*>(c)[tid] = vc;
    }
}
```

### Source
Programming Guide, Section 2.2.4.1 (Coalesced Global Memory Access)

## Skill 3: Vectorized Async Copy with 16-byte Width for L1 Bypass

### When to Use
- Copying data from global memory to shared memory (compute capability 8.0+)
- Want to avoid polluting L1 cache with streaming data
- Each thread copies at least 16 bytes

### How to Apply
1. Use `__pipeline_memcpy_async()` or `cuda::memcpy_async()` with 16-byte aligned sources
2. The hardware uses L1 bypass mode when copy size is 16 bytes per thread
3. Commit and wait on pipeline before using shared memory data

### Code Template
```cuda
__global__ void async_vec_copy(const float* __restrict__ global_in) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x * 4 + tid * 4;

    // 16-byte async copy: bypasses L1, directly global -> shared
    __pipeline_memcpy_async(&smem[tid * 4],
                            &global_in[gid],
                            sizeof(float4));
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    // Use smem[tid * 4 .. tid * 4 + 3]
}
```

### Source
Best Practices Guide, Section 10.2.3.4 (Asynchronous Copy from Global to Shared)

## Skill 4: Vectorized Atomic Operations with atom.add.vec

### When to Use
- Accumulating results into global memory arrays where multiple threads write to nearby locations
- Using float32 data with compute capability 9.0+ (Hopper)
- Reducing scatter-write pressure in epilogue stages

### How to Apply
1. Pack 2 or 4 float values into float2/float4
2. Use PTX inline assembly for `atom.add.vec.f32` (.v2 or .v4)
3. Ensure destination pointer is aligned to vector width

### Code Template
```cuda
__device__ void atomic_add_vec2(float* addr, float val0, float val1) {
    float tmp0, tmp1;
    asm volatile(
        "atom.add.v2.f32 {%0, %1}, [%2], {%3, %4};"
        : "=f"(tmp0), "=f"(tmp1)
        : "l"(addr), "f"(val0), "f"(val1)
        : "memory"
    );
}
```

### Source
PTX ISA, atom.add.vec.f32

## Skill 5: Align Data Structures for Vectorized Access

### When to Use
- Custom data structures are accessed in bulk from global memory
- Array-of-structures layout is used
- Want to guarantee the compiler generates vector loads

### How to Apply
1. Use `__align__(16)` on structures to force 16-byte alignment
2. Pad structures to be exactly 8 or 16 bytes
3. Verify with `--ptxas-options=-v` that vector loads are generated

### Code Template
```cuda
struct __align__(16) Particle {
    float x, y, z, w;  // 16 bytes, aligned
};

__global__ void load_particles(const Particle* __restrict__ particles,
                               float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Compiler generates a single 128-bit load
        Particle p = particles[idx];
        out[idx] = p.x + p.y + p.z + p.w;
    }
}
```

### Source
Best Practices Guide, Section 10.2.1 (Coalesced Access)

## Cascading Opportunities (unlocks)
After applying vectorized access:
1. Check memory/coalescing -- vectorized loads naturally produce fewer, wider transactions
2. Check memory/data-prefetch -- 16-byte async copies enable L1 bypass for better cache utilization
3. Check memory/shared-memory-cache -- vectorized loads into shared memory reduce staging overhead

## Conflicts
- memory/register-pressure: float4 variables occupy 4 registers each; heavy vectorization can increase register pressure and reduce occupancy

## Principles
- **P1**: Maximize bytes-used-per-transaction ratio. A 128-bit load uses 100% of a 16-byte fetch; vectorization reduces instruction count.
- **P2**: Alignment is critical. Misaligned vector loads degrade to multiple scalar transactions. Use cudaMalloc (256-byte aligned) or __align__ attributes.
- **P3**: Vector width should match data access pattern. If each thread naturally needs 4 elements, use float4; do not force vectorization when it introduces complexity.

## Open Questions (for Level 3 verification)
- Q1: On Hopper (sm_90), does float4 load still produce a single 128-bit transaction, or does the hardware split it?
- Q2: What is the measured bandwidth difference between float4 loads vs 4x float loads on A100 for a simple copy kernel?
- Q3: Does __builtin_assume_aligned help the compiler generate vector loads when pointer alignment is known but not declared?
