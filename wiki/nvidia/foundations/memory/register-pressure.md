---
title: Register Pressure
status: draft
evidence_level: measured
applies_to_pattern_class:
- cuda-core
applies_to_ops:
- elementwise
- reduction
- normalization
- scan
requires_sm: '>=9.0'
requires_features: []
single_kernel_useful: true
cuda_version_tested: '12.9'
driver_version_tested: 570.124.06
toolchain: nvcc 12.9 + ptxas 12.9
measured_on: H200-SXM
source:
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1064-L1065
  excerpt: Register pressure occurs when there are not enough registers available
    for a given task.
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1093-L1094
  excerpt: One of several factors that determine occupancy is register availability...
    if each thread block uses many registers, the number of thread blocks that can
    be resident on a multiprocessor is reduced.
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L1215
  excerpt: Using this option to reduce the number of registers a kernel can use may
    result in more thread blocks being scheduled on the SM concurrently, but may also
    result in more register spilling.
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L1219-L1223
  excerpt: Local memory is thread local storage... Any variable if the kernel uses
    more registers than available, that is register spilling.
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L22818-L22837
  excerpt: the compiler uses heuristics to minimize register usage while keeping register
    spilling and instruction count to a minimum.
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1089
  excerpt: Higher occupancy does not always equate to higher performance-there is
    a point above which additional occupancy does not improve performance.
artifacts:
  code: sources/experience/hw-probes/register-pressure/artifacts/reg_pressure_probe.cu
  build: nvcc -arch=sm_90a -O3 -std=c++17 -Xptxas=-v -o probe_default reg_pressure_probe.cu
  introspection: ''
  profile: ''
related_apis:
- cudaFuncGetAttributes
- cudaOccupancyMaxActiveBlocksPerMultiprocessor
- __launch_bounds__
- __maxnreg__
related_skills:
- compiler-hints
- ilp
id: skill-register-pressure
type: skill
vendor: nvidia
tags:
- cuda-cpp
applies_to:
- general
---
## What

Register pressure is the condition where a kernel's demand for per-thread registers exceeds what is available without sacrificing either occupancy or performance. On H200 (sm_90a), each SM has 65,536 32-bit registers shared among all resident threads. The maximum per-thread register count is 255. Register allocation determines the fundamental tradeoff between occupancy (how many warps can co-reside on an SM) and per-thread capability (how many values each thread can hold without spilling to slow local memory).

The three key mechanisms at play:

1. **Occupancy is bounded by register usage.** With block size 256 and 64 registers/thread, only 4 blocks fit: 65536 / (256 * 64) = 4 blocks = 1024 threads = 32 warps = 50% occupancy. With 32 registers/thread, 8 blocks fit = 2048 threads = 64 warps = 100% occupancy.

2. **Register spilling goes to local memory.** When the compiler cannot fit all live variables into the available register budget, it spills values to local memory. Despite the name "local", this storage resides in the global memory space (device DRAM), cached through L1 and L2. The programming guide states: "Local memory is thread local storage similar to registers and managed by NVCC, but the physical location of local memory is in the global memory space."

3. **Spill cost dominates occupancy gain.** A register access costs zero extra cycles. A spill load/store traverses the L1/L2/HBM hierarchy with latencies ranging from tens to hundreds of cycles. Our probe shows that doubling occupancy via forced spilling caused a 4.84x slowdown.

The compiler manages register allocation automatically using heuristics that balance register usage, spill cost, and instruction count. The programmer can influence this via `__launch_bounds__`, `__maxnreg__`, and `--maxrregcount` (see the compiler-hints skill).

Register allocation also interacts with ILP: more independent accumulator chains require more registers. The ILP skill discusses this tradeoff -- specifically, 4 accumulators for FP32 on H200 is the sweet spot for pipeline saturation, using approximately 4 extra registers per chain.

## Why

Register pressure matters because registers are the fastest storage on the GPU (zero extra latency) and the most constrained shared resource determining occupancy.

On H200 with 65,536 registers per SM and 2,048 maximum threads per SM:
- Perfect occupancy (100%) allows only 65536 / 2048 = **32 registers per thread**.
- Most non-trivial kernels need 40-80 registers per thread, meaning occupancy is often register-limited to 50-75%.
- This is by design: the best practices guide notes "Higher occupancy does not always equate to higher performance -- there is a point above which additional occupancy does not improve performance."

The register-vs-occupancy tradeoff has an optimal range that depends on the kernel's workload:

- **Compute-bound kernels** benefit from more registers (less recomputation, more ILP capacity) even at reduced occupancy, because ILP within each warp hides arithmetic latency.
- **Memory-bound kernels** benefit from higher occupancy to hide memory latency through warp-level parallelism (TLP), so fewer registers per thread is preferable -- provided no spilling occurs.
- **The boundary condition is spilling.** Any occupancy gain that comes at the cost of register spilling is almost always a net loss.

## When to use

Register pressure management is relevant in these situations:

- **After observing non-zero spill stores/loads in `-Xptxas=-v` output.** Any spill count > 0 is a signal. Small spills (< 16 bytes) may be tolerable; large spills (> 100 bytes per thread) are a performance emergency.

- **When `cudaOccupancyMaxActiveBlocksPerMultiprocessor` reports fewer blocks than expected.** If the register count per thread is the binding constraint (rather than shared memory or block size), register pressure management can increase occupancy.

- **When adding ILP accumulators or loop unrolling causes a performance regression.** The additional registers consumed by ILP may push the kernel past an occupancy cliff (e.g., from 4 blocks/SM to 3 blocks/SM).

- **When a kernel has many live variables.** Stencil computations, matrix operations with tile buffering, or kernels processing multiple elements per thread can easily exceed 64 registers.

## When NOT to use

- **Do not blindly minimize registers.** The goal is not "fewest registers" but "best overall throughput." A kernel at 50% occupancy with zero spills will often outperform the same kernel at 100% occupancy with spills.

- **Do not add `--maxrregcount` without checking spill output.** This flag is a blunt instrument that affects all kernels in a compilation unit. If it causes spilling, it will hurt performance. Always verify with `-Xptxas=-v`.

- **Do not over-constrain `__launch_bounds__`.** Setting `minBlocksPerMultiprocessor` too high forces the compiler to reduce register usage, potentially causing spilling. See the compiler-hints skill for proper usage.

- **Do not sacrifice ILP to save registers unless profiling shows register pressure is the bottleneck.** Reducing from 4 accumulators to 1 saves ~3 registers but costs a 4x throughput reduction in compute-bound phases (see the ILP skill).

## Techniques to reduce register pressure

1. **`__launch_bounds__(maxTPB, minBlocks)`** -- Tells the compiler the expected launch configuration so it can compute a register ceiling L = regsPerSM / (maxTPB * minBlocks). The compiler will reduce register usage to stay at or below L, spilling only if necessary. (Link: compiler-hints skill)

2. **`__maxnreg__(N)` or `--maxrregcount=N`** -- Directly caps the per-thread register count. Cannot be combined with `__launch_bounds__` on the same kernel. Only use when you know the target register count will not cause excessive spilling.

3. **Reduce ILP factor** -- If a kernel uses N independent accumulator chains and is register-limited, reducing N frees N registers at the cost of reduced per-thread throughput. The sweet spot on H200 is typically 4 chains for FP32 (matching the 4-cycle FMA pipeline depth).

4. **Break large kernels into smaller sequential stages** -- A monolithic kernel with many live variables across phases forces all variables to be live simultaneously. Splitting into multiple kernel launches (persistent-kernel or multi-pass) allows each stage to use fewer registers. The overhead is kernel launch latency (typically 3-10 us).

5. **Manual register tiling via shared memory** -- Explicitly store intermediate values in shared memory instead of registers. This trades register pressure for shared memory pressure and adds shared memory access latency (~20-30 cycles on H200), but shared memory is far faster than local memory spills (~100+ cycles for L2/HBM).

6. **Use smaller data types** -- `__half` or `__nv_bfloat16` values can be packed two per 32-bit register, halving register consumption for accumulator arrays. This also doubles throughput for supported operations.

## How to detect register pressure

### Compile-time: `-Xptxas=-v`

```bash
nvcc -arch=sm_90a -O3 -std=c++17 -Xptxas=-v -o kernel kernel.cu
```

Output format:
```
ptxas info : Function properties for _Z<mangled>
    <N> bytes stack frame, <S> bytes spill stores, <L> bytes spill loads
ptxas info : Used <R> registers, ...
```

- `Used R registers`: per-thread register count
- `S bytes spill stores` / `L bytes spill loads`: non-zero = spilling
- Additional warning flags: `-Xptxas=-warn-spills` (warn on any spill), `-Xptxas=-warn-lmem-usage` (warn on any local memory use)

### Runtime: `cudaFuncGetAttributes`

```cpp
cudaFuncAttributes attr;
cudaFuncGetAttributes(&attr, myKernel);
printf("regs=%d, localSize=%zu\n", attr.numRegs, attr.localSizeBytes);
```

- `attr.numRegs`: register count per thread
- `attr.localSizeBytes`: local memory per thread (> 0 implies spilling)

### Runtime: occupancy query

```cpp
int numBlocks;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &numBlocks, myKernel, blockSize, dynamicSmem);
float occupancy = (float)(numBlocks * blockSize) /
                  deviceProp.maxThreadsPerMultiProcessor;
```

## Reading the occupancy math on H200

H200 (sm_90a) register file parameters:
- 65,536 registers per SM
- 255 max registers per thread
- 2,048 max threads per SM (64 warps)
- Register allocation granularity: 256 registers per warp

Occupancy formula (register-limited case):
```
warps_by_regs = floor(65536 / (regs_per_thread * 32))
  -- but regs_per_thread is rounded up to nearest multiple of 8
     (256 regs per warp / 32 threads per warp = 8 reg granularity)
occupancy = min(warps_by_regs, 64) / 64
```

Example register-occupancy table (block size 256):

| regs/thread | warps possible | blocks/SM | occupancy |
|---|---|---|---|
| 32 | 64 | 8 | 100% |
| 40 | 51 | 6 | 75% |
| 48 | 42 | 5 | 62.5% |
| 56 | 36 | 4 | 50% |
| 64 | 32 | 4 | 50% |
| 80 | 25 | 3 | 37.5% |
| 96 | 21 | 2 | 25% |
| 128 | 16 | 2 | 25% |
| 160 | 12 | 1 | 12.5% |
| 255 | 8 | 1 | 12.5% |

Note: the actual occupancy depends on the interplay of register count, shared memory per block, and block size. Registers are often the binding constraint.

## Measured Characteristics

- Register pressure spill tradeoff probe: On H200 (sm_90a, CUDA 12.9), a 48-accumulator FMA kernel compiled with default register allocation used **64 registers/thread** with **zero spills** (0.1738 ms median, 4 blocks/SM, 50% occupancy). The same kernel compiled with `--maxrregcount=32` used **32 registers/thread** but generated **660 bytes of spill stores** and **784 bytes of spill loads** per thread (0.8414 ms median, 8 blocks/SM, 100% occupancy). Result: **4.84x slowdown** despite doubled occupancy, confirming that spill cost vastly outweighs occupancy gain when the spill volume is large.
