---
title: Compiler Hints - Pitfalls
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
single_kernel_useful: true
source:
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L22833-L22841
  excerpt: If the initial register usage exceeds L, the compiler reduces it until
    it is less than or equal to L. This usually results in increased local memory
    usage and/or a higher number of instructions.
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L22904
  excerpt: The __launch_bounds__() and __maxnreg__() qualifiers cannot be applied
    to the same kernel together.
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L22346
  excerpt: Since register pressure is a critical issue in many CUDA codes, the use
    of restricted pointers can negatively impact performance by reducing occupancy.
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L22863-L22887
  excerpt: When MyKernel is invoked with the maximum number of threads per block...
    __CUDA_ARCH__ is undefined in host code
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L22324
  excerpt: Note that all pointer arguments must be restricted for the compiler optimizer
    to be effective.
id: pitfall-compiler-hints
type: pitfall
vendor: nvidia
---
## P1: Register spilling from overly tight launch_bounds

**Symptom**: Kernel performance degrades after adding `__launch_bounds__` despite higher theoretical occupancy. `-Xptxas=-v` output shows non-zero "spill stores" and "spill loads."

**Cause**: The `minBlocksPerMultiprocessor` parameter is set too high, forcing the register cap L below what the kernel naturally requires. The compiler must spill excess registers to local memory, which resides in global memory (L1/L2 cached but still much slower than registers).

**Example**: On H200 (sm_90a) with `__launch_bounds__(256, 8)`, the register cap is 65536 / (256 * 8) = 32 per thread. A kernel that naturally needs 56 registers would spill 24 registers' worth of data to local memory.

**Detection**: Always compile with `-Xptxas=-v` and check:
```
ptxas info: Used 32 registers, 96 bytes spill stores, 96 bytes spill loads
```
Any non-zero spill count is a red flag. The cost of spilling typically outweighs the benefit of higher occupancy.

**Fix**: Reduce `minBlocksPerMultiprocessor` until spill counts reach zero. Use the occupancy calculator or `cudaOccupancyMaxActiveBlocksPerMultiprocessor` to find the actual block count, then set `minBlocksPerMultiprocessor` to that value or one below.

## P2: Combining __launch_bounds__ and __maxnreg__ on the same kernel

**Symptom**: Compilation error.

**Cause**: The programming guide explicitly states: "The `__launch_bounds__()` and `__maxnreg__()` qualifiers cannot be applied to the same kernel together." They are mutually exclusive approaches to controlling register allocation.

**Fix**: Choose one approach. `__launch_bounds__` is generally preferred because it communicates occupancy intent (the compiler derives the register cap from the occupancy target). `__maxnreg__` directly controls register count but gives the compiler no occupancy information.

## P3: __CUDA_ARCH__ undefined in host code for launch configuration

**Symptom**: Kernel launches with the wrong thread count when `__launch_bounds__` parameters are architecture-conditional.

**Cause**: The programming guide pattern uses `__CUDA_ARCH__` to set different launch bounds per architecture:
```cuda
#if __CUDA_ARCH__ >= 900
    #define MY_KERNEL_MAX_THREADS  512
#else
    #define MY_KERNEL_MAX_THREADS  256
#endif
```
But `__CUDA_ARCH__` is undefined in host code, so `kernel<<<grid, MY_KERNEL_MAX_THREADS>>>()` in host code always uses the else branch (256), even when targeting sm_90a.

**Fix**: Either use a compile-time constant that does not depend on `__CUDA_ARCH__`, or determine the thread count at runtime via `cudaGetDeviceProperties`:
```cuda
int threadsPerBlock = (prop.major >= 9) ? 512 : 256;
kernel<<<grid, threadsPerBlock>>>(...);
```

## P4: __restrict__ applied to aliased pointers

**Symptom**: Silent incorrect results. The kernel produces wrong output for certain input patterns, especially when input and output buffers overlap.

**Cause**: `__restrict__` is a promise to the compiler that the pointer does not alias any other pointer. If this promise is violated (e.g., in-place operations where `input == output`), the compiler may reorder loads and stores in ways that produce incorrect results. This is undefined behavior -- no compiler warning, no runtime error.

**Fix**: Only use `__restrict__` when you can guarantee non-aliasing. For in-place kernels, do not mark both input and output as `__restrict__`. Review all call sites to ensure no aliasing.

## P5: Partial __restrict__ annotation is ineffective

**Symptom**: No performance improvement despite adding `__restrict__` to some pointer parameters.

**Cause**: The programming guide states: "Note that all pointer arguments must be restricted for the compiler optimizer to be effective." If only some pointers are marked `__restrict__`, the compiler cannot safely assume non-aliasing between the annotated and unannotated pointers.

**Fix**: Annotate all pointer parameters in the function, or none. For `__global__` functions, also use `const` on read-only pointers to enable the `ld.global.nc` read-only cache path.

## P6: __restrict__ increasing register pressure

**Symptom**: Adding `__restrict__` to all pointer parameters causes register count to increase (visible in `-Xptxas=-v` output), reducing occupancy and potentially harming performance.

**Cause**: `__restrict__` enables the compiler to cache more loads in registers and perform common sub-expression elimination. This optimization trades memory accesses for register pressure. The programming guide warns: "The result is a reduced number of memory accesses and computations, balanced by an increase in register pressure from caching loads and common sub-expressions in registers."

**Detection**: Compare register counts with and without `__restrict__` using `-Xptxas=-v`. If the increase crosses an occupancy threshold (causing fewer blocks to fit on an SM), the net effect may be negative.

**Fix**: If register pressure is already high, consider not using `__restrict__` on some parameters, or combine with `__launch_bounds__` to cap the register count. Profile both variants to determine which is faster.

## P7: Launching with more threads than maxThreadsPerBlock

**Symptom**: Kernel launch fails at runtime with "invalid configuration argument" or "too many resources requested for launch."

**Cause**: `__launch_bounds__(maxThreadsPerBlock)` sets a hard cap on the block size. The runtime enforces this: launching with more threads per block than the declared maximum results in a launch failure.

**Detection**: Check `cudaFuncAttributes.maxThreadsPerBlock` at runtime. In the probe, `heavy_no_lb` reported `maxTPB=1024` while `heavy_lb_256_4` reported `maxTPB=256`.

**Fix**: Ensure the launch configuration's block size does not exceed the declared `maxThreadsPerBlock`. When using architecture-conditional bounds, be especially careful with the host-side launch config (see P3).

## P8: Forgetting cudaFuncSetAttribute for large dynamic shared memory

**Symptom**: Kernel launch fails with "invalid configuration argument" when requesting more than 48KB of dynamic shared memory.

**Cause**: By default, kernels are limited to 48KB of dynamic shared memory. Using more requires an explicit opt-in via `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)` before the launch. This is a runtime requirement, not a compile-time one.

**Fix**: Call `cudaFuncSetAttribute` with the required shared memory size before the first launch that needs it:
```cuda
int maxbytes = 98304;  // 96 KB
cudaFuncSetAttribute(MyKernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
MyKernel<<<gridDim, blockDim, maxbytes>>>(...);
```

## P9: Over-unrolling with #pragma unroll

**Symptom**: Kernel performance degrades or register count spikes after adding `#pragma unroll` to a loop.

**Cause**: Full unrolling of loops with large trip counts creates many copies of the loop body, increasing code size (instruction cache pressure) and register pressure (many live variables from all iterations simultaneously). The instruction cache on H200 is limited, and code that exceeds it incurs costly re-fetches.

**Detection**: Compare register counts and instruction counts with and without the pragma using `-Xptxas=-v`.

**Fix**: Use `#pragma unroll N` with a specific factor (e.g., 4 or 8) instead of unbounded `#pragma unroll`. The factor should balance ILP benefits against code size. For loops with variable or large trip counts, `#pragma unroll 1` explicitly disables unrolling.

## P10: Assuming launch_bounds always reduces register count

**Symptom**: Confusion when `__launch_bounds__(maxTPB, minBlocks)` results in the compiler using more registers, not fewer.

**Cause**: When both parameters are specified, the compiler may intentionally increase register usage up to the limit L to "reduce the number of instructions and better hide the latency of single-threaded instructions." This was observed in the probe: `heavy_lb_256_4` used 56 registers vs 48 for `heavy_no_lb`.

**This is not a bug** -- it is the documented behavior. The compiler trades occupancy headroom for instruction efficiency when it knows the occupancy floor.

**Fix**: No action needed if spill counts remain zero. Verify with `-Xptxas=-v` that no spilling occurred. If register usage exceeds L, the compiler will spill and that is where problems start (see P1).
