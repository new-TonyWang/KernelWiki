---
title: Compiler Hints
status: draft
evidence_level: measured
applies_to_pattern_class:
- cuda-core
applies_to_ops:
- elementwise
- reduction
- normalization
- scan
requires_sm: '>=3.0'
requires_features: []
single_kernel_useful: true
cuda_version_tested: '12.9'
driver_version_tested: 570.124.06
toolchain: nvcc 12.9 + ptxas 12.9
measured_on: H200-SXM
source:
- path: spec
  anchor: Reference
artifacts:
  code: sources/experience/hw-probes/compiler-hints/artifacts/launch_bounds_probe.cu
  build: nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo -Xptxas=-v -o launch_bounds_probe
    launch_bounds_probe.cu
  introspection: ''
  profile: ''
related_apis:
- __launch_bounds__
- __maxnreg__
- __restrict__
- cudaFuncSetAttribute
related_skills:
- ilp
- warp-primitives
- fast-math
id: skill-compiler-hints
type: skill
vendor: nvidia
tags:
- cuda-cpp
applies_to:
- general
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L22816-L22905
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L22292-L22365
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L24765-L24810
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L22890-L22905
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1064-L1099
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L2318-L2324
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L4094-L4130
---
## What

Compiler hints are source-level annotations and compiler flags that guide CUDA code generation without changing program semantics. They influence register allocation, occupancy, memory-access patterns, and loop structure. The key compiler hints covered by this skill are:

**Launch configuration directives:**

- `__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)` -- Annotates a `__global__` function with the maximum block size and an optional minimum resident-block count. The compiler derives a register ceiling L from these parameters and adjusts code generation accordingly. Compiles to the `.maxntid` and `.minnctapersm` PTX directives. A third optional parameter `maxBlocksPerCluster` controls cluster launch limits (compiles to `.maxclusterrank`).

- `__maxnreg__(N)` -- Directly caps the per-thread register count for a kernel. Compiles to the `.maxnreg` PTX directive. Cannot be used together with `__launch_bounds__` on the same kernel.

- `--maxrregcount=N` -- nvcc flag that sets a per-file register cap for all `__global__` functions. Ignored for kernels that have `__maxnreg__`.

**Pointer and memory hints:**

- `__restrict__` -- Declares that a pointer does not alias other pointers, enabling the compiler to cache loads in registers, reorder memory operations, and eliminate redundant loads. When applied to `const` pointers in `__global__` functions, loads compile to `ld.global.nc` (read-only data cache path) instead of `ld.global`.

**Loop control:**

- `#pragma unroll` -- Controls loop unrolling. Without an argument, fully unrolls loops with known trip counts. With an integer argument N, unrolls by factor N. With argument 1, disables unrolling.

**Runtime configuration:**

- `cudaFuncSetAttribute` -- Host-side API to configure per-kernel properties at runtime, including `cudaFuncAttributeMaxDynamicSharedMemorySize` (opt-in for >48KB dynamic shared memory) and `cudaFuncAttributePreferredSharedMemoryCarveout` (L1/shared memory partitioning).

**Diagnostic flags:**

- `-Xptxas=-v` (or `--ptxas-options=-v`) -- Reports per-kernel register count, shared memory usage, constant memory usage, and spill stores/loads. This is the primary tool for diagnosing register pressure. The `--resource-usage` flag provides similar output.

## Why

Register allocation is the central tradeoff in CUDA kernel optimization. Each SM on H200 (sm_90a) has 65,536 32-bit registers shared among all resident threads. Using fewer registers per thread allows more concurrent warps (higher occupancy), which improves latency hiding. Using more registers per thread can reduce instruction count (avoiding recomputation) and eliminate register spills to local memory, but reduces occupancy.

The compiler makes this tradeoff heuristically. Compiler hints let the programmer override or guide these heuristics based on knowledge the compiler lacks -- specifically, the intended launch configuration and whether pointer aliasing exists.

On H200 with `__launch_bounds__(256, 4)`:
- Register cap = 65536 / (256 * 4) = 64 registers per thread
- The compiler can use up to 64 registers to reduce instruction count
- If the kernel naturally needs >64 registers, the compiler spills to local memory (which has global memory latency)

Without launch_bounds, the compiler uses its own heuristics to balance register usage against occupancy. The heuristic can be either too conservative (using fewer registers than optimal, increasing instruction count) or too aggressive (using too many registers, reducing occupancy).

`__restrict__` eliminates aliasing analysis overhead and enables the read-only cache path (`ld.global.nc`), which uses the texture cache and avoids polluting the L1 data cache with read-only data.

## When to use

- **`__launch_bounds__` with one argument** -- Always include on production kernels as a forward-compatibility safeguard. The programming guide recommends: "developers should include the single argument `__launch_bounds__(maxThreadsPerBlock)` which specifies the largest block size with which the kernel will launch. Failure to do so could result in 'too many resources requested for launch' errors."

- **`__launch_bounds__` with two arguments** -- When profiling shows that occupancy is limited by register usage and you know the minimum number of blocks needed per SM. Set `minBlocksPerMultiprocessor` to the desired occupancy floor. The compiler will then adjust register allocation to guarantee that many blocks can co-reside.

- **`__restrict__`** -- On all pointer parameters in `__global__` and `__device__` functions where the programmer can guarantee no aliasing. Especially valuable when the kernel performs multiple reads from the same addresses (common sub-expressions) or when read-only data should use the texture cache path.

- **`#pragma unroll`** -- On inner loops with small known trip counts where the compiler's default unrolling is insufficient. Also on reduction loops, ILP-generating loops, and loops over warp shuffle steps.

- **`cudaFuncSetAttribute` for dynamic shared memory** -- Required when a kernel uses more than 48KB of dynamic shared memory. Must be called before the kernel launch.

- **`-Xptxas=-v`** -- During development, always compile with this flag to monitor register and shared memory usage. Sudden register count increases across code changes indicate potential performance regressions.

## When NOT to use

- **Overly tight `__launch_bounds__`** -- Setting `minBlocksPerMultiprocessor` too high forces the compiler to reduce registers below what the kernel naturally needs, causing register spilling. Spills go to local memory (global memory latency), which can be far worse than reduced occupancy. Check `-Xptxas=-v` output for "spill stores" and "spill loads" after adding launch_bounds.

- **`__restrict__` with aliased pointers** -- If two pointer arguments actually can point to overlapping memory, adding `__restrict__` produces undefined behavior. In-place operations (where input and output buffers overlap) must not use `__restrict__` on both pointers.

- **`#pragma unroll` on large or variable-trip loops** -- Full unrolling of large loops increases code size, which can cause instruction cache misses and increase register pressure. Use `#pragma unroll N` with a specific factor instead of unbounded unrolling.

- **`__maxnreg__` and `__launch_bounds__` together** -- These are mutually exclusive on the same kernel. The compiler rejects the combination.

- **Blindly maximizing occupancy** -- Higher occupancy does not always mean higher performance. A kernel that is compute-bound may benefit from more registers (lower occupancy but less recomputation) rather than fewer registers (higher occupancy but more instructions). The best practices guide notes: "Higher occupancy does not always equate to higher performance -- there is a point above which additional occupancy does not improve performance."

## Classical example: vector operation with compiler hints

```cuda
#define BLOCK_SIZE 256

__global__ void
__launch_bounds__(BLOCK_SIZE, 4)
fused_scale_bias(const float* __restrict__ input,
                 const float* __restrict__ scale,
                 const float* __restrict__ bias,
                 float*       __restrict__ output,
                 int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // __restrict__ enables the compiler to:
    //   1. Load input[i], scale[i], bias[i] via ld.global.nc (read-only cache)
    //   2. Cache the loads in registers without worrying about aliasing
    //   3. Reorder loads freely for ILP
    if (i < n) {
        float x = input[i];
        float s = scale[i];
        float b = bias[i];
        output[i] = x * s + b;
    }
}

// Host-side: opt-in for large dynamic shared memory
// cudaFuncSetAttribute(fused_scale_bias,
//     cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
```

Compile with diagnostics:
```bash
nvcc -arch=sm_90a -O3 -std=c++17 -Xptxas=-v -o kernel kernel.cu
```

Reading the `-Xptxas=-v` output:
```
ptxas info: Used 14 registers, 0 bytes spill stores, 0 bytes spill loads
```
- `Used 14 registers`: per-thread register count
- `0 bytes spill stores/loads`: no register spilling (good)
- If spill counts are non-zero, the kernel is over-pressured and `__launch_bounds__` may need loosening

## Measured Characteristics

- launch-bounds register-effect probe: On H200 (sm_90a, CUDA 12.9), a register-heavy kernel (32 accumulators, 3 rounds of cross-dependent FMA) showed that `__launch_bounds__(256, 4)` caused the compiler to allocate **56 registers** (vs 48 without launch bounds), with zero spills in both cases. The compiler increased register usage to reduce instruction count, staying within the 64-register occupancy cap. Latency was identical (0.120 ms) because both variants achieved 100% occupancy at block size 256. This confirms the programming guide statement that the compiler "may increase register usage up to L in order to reduce the number of instructions."
