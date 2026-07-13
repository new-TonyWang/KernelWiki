---
status: unverified
verified_date: null
verified_on: null
performance_data:
  before: null
  after: null
source:
  - "Best Practices Guide, Section 12.1.5 (Loop Counters Signed vs. Unsigned)"
  - "Best Practices Guide, Section 13.2 (Branch Predication / pragma unroll)"
  - "Best Practices Guide, Section 20.1 (nvcc Compiler Switches)"
  - "Programming Guide, Section 5.4.1.4 (__restrict__ Pointers)"
  - "Programming Guide, Section 5.4.1.5 (__grid_constant__ Parameters)"
  - "Programming Guide, Section 5.4.9 (Compiler Optimization Hints)"
  - "Programming Guide, Section 5.4.3.2 (Launch Bounds)"
  - "Programming Guide, Section 5.4.3.3 (Maximum Number of Registers per Thread)"
cross_ref:
  - "Best Practices Guide, Section 12.1.10 (Precision-related Compiler Flags)"
  - "Programming Guide, Section 2.5.4.3 (Optimization Options)"
  - "Programming Guide, Section 2.5.4.4 (Link-Time Optimization LTO)"
related_apis:
  - __restrict__
  - __grid_constant__
  - __launch_bounds__
  - __maxnreg__
  - __builtin_assume_aligned
  - __builtin_assume
  - __builtin_expect
  - __builtin_unreachable
  - "#pragma unroll"
  - cudaFuncSetAttribute
related_experience: []
unlocks:
  - "instruction-level-parallelism: #pragma unroll and __restrict__ enable the compiler to expose ILP"
  - "register-pressure: __launch_bounds__ and __maxnreg__ control register allocation"
  - "occupancy-tuning: launch bounds directly influence occupancy"
conflicts_with:
  - "operator-fusion: __fadd_rn / __fmul_rn prevent FMA fusion"
---

# Compiler Hints

## Skill 1: __restrict__ Pointers for Alias Elimination
### When to Use
When multiple pointer parameters point to non-overlapping memory regions. This allows the compiler to cache loads in registers, eliminate redundant loads, and reorder instructions.

### How to Apply
1. Add `__restrict__` to all pointer parameters that do not alias each other.
2. All pointer parameters should be restricted for the optimizer to be fully effective.
3. For `__global__` function `const` pointers marked with `__restrict__`, loads compile to `ld.global.nc` (read-only texture cache path), reducing L1 pressure.

### Code Template
```cuda
__global__ void saxpy(const float* __restrict__ x,
                       const float* __restrict__ y,
                       float* __restrict__ out,
                       float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a * x[i] + y[i];  // Compiler caches x[i], y[i] freely
    }
}
```

### Source
Programming Guide, Section 5.4.1.4 (__restrict__ Pointers)

## Skill 2: __launch_bounds__ for Register and Occupancy Control
### When to Use
When you need to control the register budget to achieve a target occupancy, or to prevent "too many resources requested for launch" errors.

### How to Apply
1. Annotate the `__global__` function with `__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)`.
2. The compiler adjusts register allocation to fit the specified occupancy target.
3. Use `__CUDA_ARCH__` for architecture-dependent bounds.

### Code Template
```cuda
#define THREADS 256
#if __CUDA_ARCH__ >= 900
    #define MIN_BLOCKS 3
#else
    #define MIN_BLOCKS 2
#endif

__global__ void
__launch_bounds__(THREADS, MIN_BLOCKS)
my_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = rsqrtf(data[idx] + 1.0f);
    }
}
```

### Source
Programming Guide, Section 5.4.3.2 (Launch Bounds)

## Skill 3: #pragma unroll for Loop Optimization
### When to Use
When you want explicit control over loop unrolling beyond the compiler's default heuristic. The compiler unrolls small constant-trip-count loops automatically, but `#pragma unroll` gives manual control.

### How to Apply
1. Place `#pragma unroll` immediately before the loop for full unrolling.
2. Use `#pragma unroll N` for partial unrolling by factor N.
3. Use `#pragma unroll 1` to disable unrolling when the compiler over-unrolls.

### Code Template
```cuda
__device__ float reduce_array(const float* arr, int n) {
    float sum = 0.0f;

    // Fully unroll: all iterations become independent instructions
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        sum += arr[i];
    }

    // Partially unroll by factor 4
    #pragma unroll 4
    for (int i = 8; i < n; i++) {
        sum += arr[i];
    }

    return sum;
}
```

### Source
Programming Guide, Section 5.4.9.1 (#pragma unroll); Best Practices Guide, Section 13.2 (Branch Predication)

## Skill 4: __builtin_assume_aligned for Vectorized Memory Access
### When to Use
When you know a pointer is aligned to a specific boundary but the compiler cannot prove it. This enables the compiler to generate wider (vectorized) load/store instructions.

### How to Apply
1. Call `__builtin_assume_aligned(ptr, alignment)` where alignment is a power of two.
2. The returned pointer carries the alignment guarantee for subsequent accesses.
3. Common alignments: 16 (for float4), 32 (for WMMA fragments).

### Code Template
```cuda
__global__ void vectorized_copy(const float* __restrict__ in,
                                 float* __restrict__ out, int n) {
    const float* aligned_in = (const float*)__builtin_assume_aligned(in, 16);
    float* aligned_out = (float*)__builtin_assume_aligned(out, 16);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n / 4) {
        // Compiler can now generate ld.global.v4.f32
        float4 val = ((const float4*)aligned_in)[idx];
        ((float4*)aligned_out)[idx] = val;
    }
}
```

### Source
Programming Guide, Section 5.4.9.2 (__builtin_assume_aligned)

## Skill 5: Signed Loop Counters for Better Optimization
### When to Use
Always prefer signed integers for loop counters. Unsigned overflow has defined semantics in C that restrict compiler optimizations; signed overflow is undefined, giving the compiler more optimization freedom.

### How to Apply
1. Declare loop counters as `int` (signed) rather than `unsigned int`.
2. This enables strength reduction, induction variable optimization, and loop vectorization.

### Code Template
```cuda
__global__ void strided_access(const float* __restrict__ in,
                                float* __restrict__ out,
                                int n, int stride, int offset) {
    // Good: signed int loop counter
    for (int i = 0; i < n; i++) {
        out[i] = in[offset + stride * i];
    }

    // Bad: unsigned int prevents strength reduction
    // for (unsigned int i = 0; i < n; i++) { ... }
}
```

### Source
Best Practices Guide, Section 12.1.5 (Loop Counters Signed vs. Unsigned)

## Skill 6: __builtin_assume and __builtin_expect for Branch Optimization
### When to Use
When you can provide the compiler with runtime invariants or branch prediction hints that it cannot infer statically.

### How to Apply
1. `__builtin_assume(condition)`: tells the compiler the condition is always true, enabling dead code elimination and range analysis.
2. `__builtin_expect(expr, value)`: hints that `expr` will likely equal `value`, improving branch prediction code generation.
3. `__builtin_unreachable()`: marks unreachable code, enabling elimination of dead branches.

### Code Template
```cuda
__global__ void kernel_with_hints(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Tell compiler n is always positive and a multiple of 256
    __builtin_assume(n > 0);
    __builtin_assume(n % 256 == 0);

    if (idx < n) {
        // Hint: this branch is almost always taken
        if (__builtin_expect(data[idx] > 0.0f, 1)) {
            data[idx] = rsqrtf(data[idx]);
        }
    }
}
```

### Source
Programming Guide, Section 5.4.9.3 (__builtin_assume), Section 5.4.9.4 (__builtin_expect), Section 5.4.9.5 (__builtin_unreachable)

## Skill 7: __grid_constant__ for Read-Only Kernel Parameters
### When to Use
When a kernel parameter is read-only and shared across all threads. Without this annotation, the compiler may create per-thread copies of the parameter.

### How to Apply
1. Annotate the `__global__` function parameter with `__grid_constant__`.
2. The parameter must be `const`-qualified with a non-reference type.
3. All threads see the same address, eliminating per-thread parameter copies.

### Code Template
```cuda
struct KernelParams {
    int n;
    float alpha;
    float beta;
};

__global__ void kernel(const __grid_constant__ KernelParams params,
                        float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < params.n) {
        data[idx] = params.alpha * data[idx] + params.beta;
    }
}
```

### Source
Programming Guide, Section 5.4.1.5 (__grid_constant__ Parameters)

## Skill 8: nvcc Optimization Flags
### When to Use
When you want to control compilation behavior at the command line level for performance tuning.

### How to Apply
Key flags:
- `-maxrregcount=N`: cap register usage per kernel (file-level).
- `--ptxas-options=-v` or `-Xptxas=-v`: report per-kernel register/shared/constant memory usage.
- `-extra-device-vectorization`: enable extra vectorization passes.
- `-dlto`: enable device link-time optimization for cross-TU inlining.

### Code Template
```bash
# Report resource usage and cap registers at 64
nvcc -Xptxas=-v -maxrregcount=64 -o kernel kernel.cu

# Enable device LTO for better cross-function optimization
nvcc -dlto -o kernel kernel.cu

# Extra device vectorization
nvcc -extra-device-vectorization -o kernel kernel.cu
```

### Source
Best Practices Guide, Section 20.1 (nvcc); Programming Guide, Section 2.5.4.3 (Optimization Options)

## Cascading Opportunities (unlocks)
- **instruction-level-parallelism**: `#pragma unroll` and `__restrict__` enable the compiler to reorder and parallelize instructions.
- **register-pressure**: `__launch_bounds__` and `__maxnreg__` directly control register allocation.
- **occupancy-tuning**: Launch bounds determine the occupancy target for the compiler.

## Conflicts
- **operator-fusion**: Using `__fadd_rn`/`__fmul_rn` intrinsics prevents FMA fusion. Only use when separate rounding is intentionally required.

## Principles
- Compiler hints are zero-cost at runtime; they only affect code generation.
- `__restrict__` is the single most impactful pointer annotation for kernel performance.
- `__launch_bounds__` should be used on every production kernel to ensure forward compatibility and prevent launch failures on future hardware.
- Always verify the effect of compiler hints by comparing SASS output or profiling with Ncu.

## Open Questions
- How does `-dlto` (device link-time optimization) interact with `__launch_bounds__` across translation units?
- What is the performance impact of `#pragma nv_abi preserve_n_data` on inter-procedural register allocation?
