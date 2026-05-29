---
status: unverified
verified_date: null
verified_on: null
performance_data:
  before: null
  after: null
source:
  - Best Practices Guide, Section 10.2.4 (Local Memory)
  - Best Practices Guide, Section 10.2.7.1 (Register Pressure)
  - Programming Guide, Section 2.2.3.3 (Registers)
  - Programming Guide, Section 2.2.3.4 (Local Memory)
  - Programming Guide, Section 5.4.3.2 (Launch Bounds)
  - Programming Guide, Section 5.4.3.3 (Maximum Number of Registers per Thread)
cross_ref:
  - Best Practices Guide, Section 11.1 (Occupancy)
  - Best Practices Guide, Section 11.2 (Hiding Register Dependencies)
related_apis:
  - __launch_bounds__, __maxnreg__, -maxrregcount, setmaxnreg.inc, setmaxnreg.dec, ld.local, st.local, cudaFuncGetAttributes
related_experience: []
unlocks:
  - "latency/occupancy-tuning: reducing register usage increases occupancy"
  - "memory/data-prefetch: async copy bypasses registers, reducing register pressure"
conflicts_with:
  - "memory/vectorized-access: float4 variables consume 4 registers each"
  - "compute/instruction-level-parallelism: ILP requires more live variables, increasing register demand"
---

# Register Pressure

## Skill 1: Use __launch_bounds__ to Control Register Allocation

### When to Use
- Kernel uses too many registers, limiting occupancy
- Want the compiler to target a specific occupancy level

### How to Apply
1. Add `__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)` to the kernel
2. The compiler will limit registers so that the specified blocks can co-reside
3. Use the single-argument form `__launch_bounds__(maxThreadsPerBlock)` for forward compatibility

### Code Template
```cuda
#define THREADS_PER_BLOCK 256
#define MIN_BLOCKS_PER_SM 2

__global__ void
__launch_bounds__(THREADS_PER_BLOCK, MIN_BLOCKS_PER_SM)
myKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Compiler targets register usage allowing 2 blocks of 256 threads per SM
        data[idx] = data[idx] * 2.0f;
    }
}
```

### Source
Programming Guide, Section 5.4.3.2 (Launch Bounds)

## Skill 2: Use __maxnreg__ for Fine-Grained Register Control

### When to Use
- Need exact control over maximum registers per thread
- __launch_bounds__ is not precise enough

### How to Apply
1. Add `__maxnreg__(N)` to the kernel definition
2. Cannot be combined with `__launch_bounds__`
3. The compiler cap will be N registers per thread; excess spills to local memory

### Code Template
```cuda
__global__ void
__maxnreg__(32)
myKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}
```

### Source
Programming Guide, Section 5.4.3.3 (Maximum Number of Registers per Thread)

## Skill 3: Use -maxrregcount Compiler Option for File-Wide Control

### When to Use
- Want to control register usage for all kernels in a file
- Rapid experimentation with different register limits

### How to Apply
1. Compile with `nvcc -maxrregcount=N`
2. All kernels in the file will use at most N registers
3. Kernels with `__maxnreg__` are exempt

### Code Template
```bash
# Compile with max 64 registers per thread
nvcc -maxrregcount=64 -o my_app my_kernel.cu
```

### Source
Best Practices Guide, Section 10.2.7.1 (Register Pressure)

## Skill 4: Diagnose Register Spilling to Local Memory

### When to Use
- Kernel is slower than expected despite correct memory access patterns
- Suspect register spills (local memory access)

### How to Apply
1. Compile with `--ptxas-options=-v` to see per-kernel register and local memory usage
2. Examine PTX output (-ptx or -keep) for `.local` declarations and `ld.local`/`st.local` instructions
3. High lmem usage indicates spilling

### Code Template
```bash
# Check register and local memory usage
nvcc --ptxas-options=-v my_kernel.cu 2>&1 | grep -E "registers|lmem"
# Output example: Used 42 registers, 16 bytes lmem
```

### Source
Best Practices Guide, Section 10.2.4 (Local Memory)

## Skill 5: Reduce Register Pressure by Limiting Live Variables

### When to Use
- Kernel has many intermediate variables alive simultaneously
- Register count exceeds the target for desired occupancy

### How to Apply
1. Recompute values instead of storing them if computation is cheap
2. Break large kernels into smaller device functions
3. Reduce array sizes declared in registers; use shared memory for larger arrays
4. Replace float4 with sequential float operations if register-limited

### Code Template
```cuda
// High register pressure: many live variables
__global__ void high_pressure(float* data, int n) {
    float a = data[threadIdx.x];
    float b = data[threadIdx.x + 1];
    float c = data[threadIdx.x + 2];
    float d = data[threadIdx.x + 3];
    float e = a + b + c + d;
    float f = a * b - c * d;
    // a, b, c, d, e, f all live simultaneously
    data[threadIdx.x] = e + f;
}

// Reduced pressure: recompute instead of storing
__global__ void low_pressure(float* data, int n) {
    float a = data[threadIdx.x];
    float b = data[threadIdx.x + 1];
    float sum_ab = a + b;
    a = data[threadIdx.x + 2];  // Reuse register
    b = data[threadIdx.x + 3];
    data[threadIdx.x] = sum_ab + a + b + (sum_ab - a * b);
}
```

### Source
Best Practices Guide, Section 10.2.7.1 (Register Pressure)

## Skill 6: Dynamic Register Adjustment with setmaxnreg (SM 9.0+)

### When to Use
- Warp specialization patterns where producer and consumer warps need different register counts
- Compute capability 9.0+ (Hopper)

### How to Apply
1. Use PTX inline assembly for `setmaxnreg.inc.sync.aligned.u32` to increase registers
2. Use `setmaxnreg.dec.sync.aligned.u32` to decrease registers
3. All threads in the warpgroup must execute this instruction

### Code Template
```cuda
__device__ void increase_registers() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 64;");
}

__device__ void decrease_registers() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 64;");
}
```

### Source
PTX ISA, setmaxnreg

## Cascading Opportunities (unlocks)
After controlling register pressure:
1. Check latency/occupancy-tuning -- more blocks can co-reside, improving latency hiding
2. Check memory/data-prefetch -- async copy reduces register demand vs synchronous copy

## Conflicts
- memory/vectorized-access: float4 loads use 4 registers; vectorization increases register demand
- compute/instruction-level-parallelism: maximizing ILP requires many live variables in registers

## Principles
- **P1**: On CC 7.0, each SM has 65,536 registers. For 100% occupancy (2048 threads), max 32 registers per thread.
- **P2**: Register allocations are rounded up to the nearest 256 registers per warp.
- **P3**: Local memory (register spill) resides in off-chip global memory, accessed with the same latency as global memory. It is cached in L1/L2.
- **P4**: Lower occupancy is not always bad: more registers per thread can reduce spilling and increase ILP, sometimes improving performance.
- **P5**: Compile with `--ptxas-options=-v` to always check register usage.

## Open Questions (for Level 3 verification)
- Q1: What is the optimal register count for a typical GEMM kernel targeting 50% occupancy on H100?
- Q2: How much does local memory spilling cost in practice with L1 caching on sm_80+?
- Q3: Does setmaxnreg on Hopper provide measurable benefit for warp-specialized GEMM?
