---
status: unverified
verified_date: null
verified_on: null
performance_data:
  before: null
  after: null
source:
  - "Best Practices Guide, Section 11.2 (Hiding Register Dependencies)"
  - "Best Practices Guide, Section 11.3 (Thread and Block Heuristics)"
  - "Programming Guide, Section 3.2.2.2 (Hardware Multithreading)"
cross_ref:
  - "Best Practices Guide, Section 11.1 (Occupancy)"
  - "Best Practices Guide, Section 10.2.7.1 (Register Pressure)"
related_apis:
  - fma.rn.f32
  - fma.rn.f64
  - "#pragma unroll"
  - mov.b128
  - lop3.b32
related_experience: []
unlocks:
  - "occupancy-tuning: ILP can substitute for occupancy in hiding latency"
  - "compiler-hints: #pragma unroll exposes ILP to the compiler"
conflicts_with:
  - "register-pressure: increasing ILP requires more live registers"
  - "occupancy-tuning: high ILP may tolerate lower occupancy, but excessive register use lowers occupancy"
---

# Instruction Level Parallelism

## Skill 1: Loop Unrolling to Expose Independent Instructions
### When to Use
When a loop body contains independent operations that can execute in parallel across pipeline stages, but the compiler's default unrolling is insufficient. Especially useful when the trip count is known at compile time.

### How to Apply
1. Place `#pragma unroll` (or `#pragma unroll N`) immediately before the loop.
2. Ensure loop iterations are independent (no cross-iteration data dependencies).
3. Full unrolling eliminates loop overhead and exposes all independent instructions.

### Code Template
```cuda
__global__ void vector_add_ilp(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Each thread processes 4 elements to expose ILP
    for (int i = idx; i < n / 4; i += stride) {
        float a0 = a[4*i+0], a1 = a[4*i+1], a2 = a[4*i+2], a3 = a[4*i+3];
        float b0 = b[4*i+0], b1 = b[4*i+1], b2 = b[4*i+2], b3 = b[4*i+3];
        c[4*i+0] = a0 + b0;
        c[4*i+1] = a1 + b1;
        c[4*i+2] = a2 + b2;
        c[4*i+3] = a3 + b3;
    }
}
```

### Source
Best Practices Guide, Section 11.2 (Hiding Register Dependencies); Programming Guide, Section 5.4.9.1 (#pragma unroll)

## Skill 2: Multiple Accumulator Registers to Break Dependency Chains
### When to Use
When a reduction or dot-product loop has a serial dependency chain through a single accumulator, limiting throughput to one FMA per ~4 cycles due to register read-after-write latency.

### How to Apply
1. Split the accumulation across N independent accumulators (e.g., `sum0`, `sum1`, `sum2`, `sum3`).
2. In each iteration, accumulate into a different register.
3. After the loop, combine the partial sums.
4. This allows the hardware to issue multiple FMAs per cycle since they target different destination registers.

### Code Template
```cuda
__device__ float dot_product_ilp4(const float* a, const float* b, int n) {
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

    #pragma unroll 4
    for (int i = 0; i < n; i += 4) {
        sum0 = fmaf(a[i+0], b[i+0], sum0);
        sum1 = fmaf(a[i+1], b[i+1], sum1);
        sum2 = fmaf(a[i+2], b[i+2], sum2);
        sum3 = fmaf(a[i+3], b[i+3], sum3);
    }

    return (sum0 + sum1) + (sum2 + sum3);
}
```

### Source
Best Practices Guide, Section 11.2 (Hiding Register Dependencies)

## Skill 3: Per-Thread Multi-Element Processing (ILP over TLP)
### When to Use
When occupancy is limited by register or shared memory constraints, and you want to hide arithmetic latency by giving each thread more independent work instead of increasing the number of warps.

### How to Apply
1. Assign each thread N elements instead of 1 (e.g., thread processes 4 or 8 values).
2. Load all values into registers, process independently, then store.
3. This trades occupancy (fewer threads needed) for ILP within each thread.
4. With sufficient ILP, even 25-33% occupancy can achieve near-peak throughput.

### Code Template
```cuda
// Each thread processes ITEMS_PER_THREAD elements
constexpr int ITEMS_PER_THREAD = 8;

__global__ void elementwise_ilp(const float* __restrict__ in,
                                 float* __restrict__ out, int n) {
    int base = (blockIdx.x * blockDim.x + threadIdx.x) * ITEMS_PER_THREAD;
    float vals[ITEMS_PER_THREAD];

    // Load phase: independent loads issued together
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++)
        vals[i] = in[base + i];

    // Compute phase: independent operations
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++)
        vals[i] = rsqrtf(vals[i] + 1.0f);

    // Store phase
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++)
        out[base + i] = vals[i];
}
```

### Source
Best Practices Guide, Section 11.3 (Thread and Block Heuristics) -- "with a high degree of exposed instruction-level parallelism (ILP) it is, in some cases, possible to fully cover latency with a low occupancy"

## Skill 4: Interleave Memory and Compute Instructions
### When to Use
When a kernel alternates between memory loads and arithmetic, and the two can overlap. Issuing loads early lets the memory system work while the compute pipeline processes previous data.

### How to Apply
1. Issue loads for the next iteration's data before computing on the current iteration's data.
2. This is software pipelining: load(k+1), compute(k), store(k-1).
3. Requires enough registers to hold both current and prefetched data.

### Code Template
```cuda
__device__ void software_pipeline(const float* a, float* out, int n) {
    float curr, next;
    next = a[0];  // Prefetch first element

    for (int i = 0; i < n - 1; i++) {
        curr = next;         // Use prefetched value
        next = a[i + 1];    // Prefetch next (overlaps with compute below)
        out[i] = curr * curr + 1.0f;  // Compute on current
    }
    out[n - 1] = next * next + 1.0f;  // Process last element
}
```

### Source
Best Practices Guide, Section 11.2 (Hiding Register Dependencies); PTX ISA, fma.rn.f32 (software pipelining of dependent chains)

## Cascading Opportunities (unlocks)
- **occupancy-tuning**: With high ILP, the kernel can tolerate lower occupancy while still hiding latency.
- **compiler-hints**: `#pragma unroll` is the primary mechanism for the compiler to expose ILP.

## Conflicts
- **register-pressure**: More ILP means more live variables in registers. Over-unrolling can cause register spills that negate the ILP benefit.
- **occupancy-tuning**: If ILP increases register pressure beyond the threshold, occupancy drops; find the balance point.

## Principles
- Arithmetic instruction latency on sm_70+ is approximately 4 cycles. To hide this with ILP alone (no TLP), you need at least 4 independent instructions in flight.
- The compiler automatically unrolls small loops with known trip counts. `#pragma unroll` controls this explicitly.
- ILP and TLP (occupancy) are complementary strategies for latency hiding. The ideal kernel uses both.
- A warp scheduler can issue one instruction per cycle; having independent instructions ready prevents pipeline stalls.

## Open Questions
- What is the optimal ILP factor (number of elements per thread) for different arithmetic intensities on Hopper vs Ampere?
- Can the compiler automatically detect and break accumulator dependency chains, or must this always be manual?
