---
title: Register Pressure -- Pitfalls
parent_skill: register-pressure
id: pitfall-register-pressure
type: pitfall
vendor: nvidia
architectures:
- sm90
- sm90a
languages:
- cuda-cpp
techniques:
- pipeline-stages
- register-budgeting
- loop-unrolling
- shared-memory-optimization
kernel_types:
- quantization
confidence: inferred
tags:
- pipeline-stages
- register-budgeting
- loop-unrolling
- shared-memory-optimization
- quantization
- cuda-cpp
---
## Pitfall 1: Blindly maximizing occupancy via --maxrregcount causes catastrophic spilling

**Symptom**: Kernel runs much slower after adding `--maxrregcount=N` despite higher reported occupancy from `cudaOccupancyMaxActiveBlocksPerMultiprocessor`.

**Root cause**: Forcing a low register cap (e.g., 32) on a kernel that naturally needs more (e.g., 64) causes the compiler to spill excess values to local memory. Local memory resides in the global memory space (L1/L2/HBM), with latencies 100-1000x worse than register access.

**Detection**: Compile with `-Xptxas=-v` and check for non-zero "spill stores" and "spill loads". Also check `cudaFuncGetAttributes` for `localSizeBytes > 0`.

**Measured example**: On H200, a 48-accumulator kernel at `--maxrregcount=32` showed 660 bytes spill stores + 784 bytes spill loads per thread, causing a 4.84x slowdown despite doubling occupancy from 50% to 100% (see probe record).

**Fix**: Remove or increase the `--maxrregcount` value. Use `-Xptxas=-v` to find the natural register count, then set the cap at or above that value. Alternatively, use `__launch_bounds__` which lets the compiler make occupancy-aware decisions rather than imposing a hard cap.

## Pitfall 2: __launch_bounds__ with excessive minBlocksPerMultiprocessor

**Symptom**: Adding `__launch_bounds__(256, 8)` causes spilling on a kernel that ran without spills when no launch bounds were specified.

**Root cause**: With `minBlocksPerMultiprocessor=8` and `maxTPB=256`, the register ceiling is L = 65536 / (256 * 8) = 32 registers per thread. If the kernel naturally needs more than 32 registers, the compiler must spill.

**Detection**: Compare `-Xptxas=-v` output before and after adding launch bounds. Look for non-zero spill counts in the annotated variant.

**Fix**: Reduce `minBlocksPerMultiprocessor` to a value where L exceeds the kernel's natural register demand. For a kernel needing 64 registers with block size 256: 65536 / (256 * L_target) >= 64, so max blocks = 4. Use `__launch_bounds__(256, 4)` or `__launch_bounds__(256, 3)`.

## Pitfall 3: Silent register pressure increase from loop unrolling

**Symptom**: Adding `#pragma unroll` or increasing the unroll factor causes a performance regression, even though the loop body is simple.

**Root cause**: Full unrolling of a loop with N iterations creates N copies of all loop-body variables, multiplying register pressure by up to N. If this pushes the kernel past an occupancy cliff (e.g., from 4 blocks/SM to 3 blocks/SM), the occupancy drop hurts more than the unrolling helped.

**Detection**: Compare register counts with `-Xptxas=-v` before and after the unroll pragma. Watch for occupancy cliffs by computing `floor(65536 / (regs_per_thread * block_size / 32))` warps.

**Fix**: Use `#pragma unroll K` with a bounded unroll factor instead of unbounded `#pragma unroll`. For FP32 on H200, K=4 matches the FMA pipeline depth and is usually sufficient.

## Pitfall 4: Large local arrays silently placed in local memory

**Symptom**: A kernel declares a small array (e.g., `float buf[64]`) inside the kernel function and shows unexpectedly high local memory usage.

**Root cause**: The programming guide states that "Arrays for which [the compiler] cannot determine that they are indexed with constant quantities" are placed in local memory. If the array index is runtime-variable, the entire array goes to local memory regardless of register availability. Even with constant indexing, large arrays exceed the register budget and are placed in local memory.

**Detection**: Check `-Xptxas=-v` for local memory usage or `localSizeBytes` from `cudaFuncGetAttributes`.

**Fix**: Replace runtime-indexed arrays with explicit scalar variables when the access pattern is known at compile time. For genuinely variable-indexed buffers, consider using shared memory instead.

## Pitfall 5: Confusing register count with register pressure

**Symptom**: A kernel uses 64 registers/thread and 0 spills, but the programmer adds `--maxrregcount=32` because "64 registers is too many."

**Root cause**: Register count alone does not indicate a problem. A kernel using 64 registers with 0 spills at 50% occupancy may be performing optimally. Register pressure is only a problem when either (a) the kernel is spilling, or (b) the kernel is latency-bound and would benefit from higher occupancy.

**Detection**: Before reducing registers, measure whether the kernel is compute-bound or memory-bound. If compute-bound, more registers (and more ILP) may help. If memory-bound, more occupancy (fewer registers) may help -- but only if it does not cause spilling.

**Fix**: Profile first, tune second. Use `cudaOccupancyMaxActiveBlocksPerMultiprocessor` to check current occupancy, and `-Xptxas=-v` to check spill status. Only reduce registers if profiling shows the kernel is latency-bound with low occupancy and no spilling would result from the reduction.

## Pitfall 6: --maxrregcount is per-file, not per-kernel

**Symptom**: A file contains two kernels -- one lightweight (needs 16 regs) and one heavy (needs 80 regs). Setting `--maxrregcount=48` helps the heavy kernel's occupancy but the lightweight kernel was already fine.

**Root cause**: `--maxrregcount` applies to all `__global__` functions in a compilation unit. It cannot be targeted at individual kernels.

**Detection**: Check `-Xptxas=-v` output for all kernels in the file.

**Fix**: Use `__maxnreg__(N)` or `__launch_bounds__` per-kernel instead of the file-wide `--maxrregcount`. Alternatively, split kernels into separate compilation units with different flags.

## Pitfall 7: __maxnreg__ and __launch_bounds__ are mutually exclusive

**Symptom**: Compilation error when both `__maxnreg__(N)` and `__launch_bounds__(M, K)` are applied to the same kernel.

**Root cause**: The programming guide explicitly states "The `__launch_bounds__()` and `__maxnreg__()` qualifiers cannot be applied to the same kernel together."

**Fix**: Choose one mechanism. `__launch_bounds__` is generally preferred because it lets the compiler balance register usage against the stated occupancy target. `__maxnreg__` is a harder cap for when you need exact register control.
