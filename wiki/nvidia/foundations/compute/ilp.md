---
title: Instruction-Level Parallelism (ILP)
status: draft
evidence_level: measured
applies_to_pattern_class:
- cuda-core
applies_to_ops:
- reduction
- scan
- elementwise
- normalization
requires_sm: '>=3.0'
requires_features: []
single_kernel_useful: true
cuda_version_tested: '12.9'
driver_version_tested: 570.124.06
toolchain: nvcc 12.9 + ptxas 12.9
measured_on: H200-SXM
source:
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1113
  excerpt: The latency of most arithmetic instructions is typically 4 cycles on devices
    of compute capability 7.0.
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1124
  excerpt: with a high degree of exposed instruction-level parallelism (ILP) it is,
    in some cases, possible to fully cover latency with a low occupancy
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1229-L1232
  excerpt: '32-bit floating-point add, multiply, multiply-add: throughput 128 ops/clock/SM
    on sm_9.0'
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L3452
  excerpt: At each instruction issue cycle, a warp scheduler selects a warp with threads
    ready to execute its next instruction
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L24765-L24805
  excerpt: '#pragma unroll: The compiler unrolls small loops with a known trip count
    by default.'
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1064-L1065
  excerpt: Register pressure occurs when there are not enough registers available
    for a given task.
artifacts:
  code: 80-experience/hw-probes/ilp/artifacts/ilp_fma_probe.cu
  build: nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo -o ilp_fma_probe ilp_fma_probe.cu
  introspection: ''
  profile: ''
related_apis: []
related_skills:
- warp-primitives
- fast-math
id: skill-ilp
type: skill
vendor: nvidia
tags:
- cuda-cpp
applies_to:
- general
---
## What

Instruction-Level Parallelism (ILP) is a technique where a single thread maintains multiple independent instruction chains in flight simultaneously, allowing the hardware to overlap the execution of one chain's instructions with the pipeline latency of another. On NVIDIA GPUs, the FP32 arithmetic pipeline has a latency of 4 cycles (measured on H200, consistent with the best-practices guide statement for compute capability 7.0+). A single dependent chain of FMA instructions therefore achieves only 1/4 of the peak throughput. By restructuring code to have 4 independent chains, a single thread can fully saturate the pipeline and reach peak throughput.

ILP is one of two complementary mechanisms for hiding arithmetic latency:

1. **Occupancy (TLP)**: the warp scheduler switches between warps on every cycle, so if enough warps are resident, a stalled warp is replaced by a ready one. This requires many active threads.
2. **ILP**: within a single warp, the scheduler can issue independent instructions from the same warp on consecutive cycles. This requires independent instruction chains within each thread, but does not require additional warps.

The best-practices guide explicitly notes (L1124): "with a high degree of exposed instruction-level parallelism (ILP) it is, in some cases, possible to fully cover latency with a low occupancy."

## Why

On H200 (sm_90a), the FP32 pipeline has a throughput of 128 operations per clock per SM (Table 5 of the best-practices guide). With a warp size of 32 threads, this means 128 / 32 = 4 warp-instructions per clock per SM (one per sub-partition). For a single warp running on one sub-partition, the throughput ceiling is 1 FMA per clock per warp.

However, with a single dependent FMA chain, the measured throughput is only 0.25 FMA per clock per warp (4.01 cycles per FMA), because each instruction must wait for the previous one to complete. With 4 independent accumulator chains, the measured throughput is 0.99 FMA per clock per warp (1.01 cycles per FMA) -- a **3.97x improvement** from ILP alone.

This matters in practice because:

- **Reduction kernels** with a single accumulator leave 75% of the FP32 pipeline idle during the accumulation phase.
- **Compute-bound kernels** at low occupancy (e.g., due to high register or shared-memory usage) cannot rely on TLP to hide latency; ILP is the only alternative.
- **Grid-stride loops** that process multiple elements per thread naturally expose ILP when each element is accumulated into a separate register.

## When to use

- **Reduction accumulation loops**: replace a single accumulator with N independent accumulators (N >= 4 for FP32 on H200). After the loop, combine them with a small tree reduction.

- **Grid-stride loops processing multiple elements per thread**: load and process K elements per iteration, accumulating into K independent registers. This is ILP by design.

- **Compute-bound tight loops**: any loop body where the critical path is a chain of dependent arithmetic operations. Unrolling + splitting into independent sub-chains exposes ILP.

- **Low-occupancy kernels**: when register pressure or shared-memory usage limits occupancy, ILP compensates for the reduced TLP by keeping the pipeline fed from within each warp.

- **`#pragma unroll`**: unrolling a loop exposes multiple iterations to the compiler, which can then schedule independent operations from different iterations in parallel. This is the simplest way to add ILP: if iterations are independent, `#pragma unroll` (or `#pragma unroll N`) lets the compiler see and schedule them concurrently.

## When NOT to use

- **Memory-bound kernels (elementwise, simple copy)**: if the kernel is bottlenecked on memory bandwidth, adding ILP to the arithmetic does not help -- the arithmetic pipeline is already idle waiting for data. In this case, ILP on memory access patterns (e.g., issuing multiple loads in flight) is what matters, not arithmetic ILP.

- **Already at full occupancy with independent warps**: if the SM has enough resident warps to keep the pipeline busy via TLP, additional ILP within each thread provides no benefit and wastes registers.

- **When register pressure is already critical**: each additional accumulator chain requires additional registers. On H200, the register file is 65536 32-bit registers per SM. If the kernel is already register-limited (the occupancy calculator shows register as the limiting factor), adding more accumulators can reduce occupancy, potentially hurting total throughput even as per-thread throughput improves. The sweet spot is typically 4 accumulators for FP32 (matching the pipeline depth).

- **Short loops with few iterations**: if the total iteration count is small (e.g., < 8), unrolling and splitting into chains adds code size with negligible benefit. The pipeline stall from a 4-iteration loop is only ~16 cycles total.

## Classical example: 4-accumulator reduction

This pattern replaces a single-accumulator grid-stride reduction with 4 independent chains, achieving near-peak FMA throughput during the accumulation phase.

```cuda
__global__ void reduce_sum_ilp4(const float* __restrict__ in,
                                float*       __restrict__ out,
                                int n) {
    // 4 independent accumulators per thread
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Grid-stride loop: each thread processes 4 elements per iteration
    int i = tid * 4;
    for (; i + 3 < n; i += stride * 4) {
        acc0 += in[i];
        acc1 += in[i + 1];
        acc2 += in[i + 2];
        acc3 += in[i + 3];
    }
    // Handle remainder
    for (; i < n; i += stride) {
        acc0 += in[i];
    }

    // Combine accumulators (serial, but only 3 adds)
    float val = acc0 + acc1 + acc2 + acc3;

    // Warp-level reduction (see warp-primitives skill)
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);

    if (threadIdx.x % 32 == 0)
        atomicAdd(out, val);
}
```

In this example, the 4 `+=` operations in the inner loop body are independent of each other, so the hardware can issue them on 4 consecutive cycles without pipeline stalls. Compared to a single-accumulator version, the accumulation phase runs at approximately 4x the throughput.

## ILP via `#pragma unroll`

When loop iterations are independent, `#pragma unroll` is the simplest way to expose ILP without manually managing multiple accumulators. The compiler unrolls the specified number of iterations and schedules independent operations across them.

```cuda
// Without unroll: compiler may not unroll, limiting ILP
for (int i = 0; i < N; ++i)
    out[tid + i * stride] = __fmaf_rn(in[tid + i * stride], scale, bias);

// With unroll: compiler sees 4 independent iterations, schedules ILP
#pragma unroll 4
for (int i = 0; i < N; ++i)
    out[tid + i * stride] = __fmaf_rn(in[tid + i * stride], scale, bias);
```

Key rules for `#pragma unroll` (from the programming guide L24765-L24805):
- No argument: fully unroll if trip count is compile-time constant.
- Argument N: unroll N iterations (N must be a positive integral constant expression).
- Argument 0 or 1: disable unrolling.
- Negative values: the pragma is ignored.

## Measured Characteristics

- [ILP FMA accumulator-chain probe](../../80-experience/hw-probes/ilp/2026-04-16-ilp.md): On H200 (sm_90a, CUDA 12.9), a dependent FMA chain (1 accumulator) costs **4.01 cycles/FMA**. With 2 independent chains: **2.00 cycles/FMA** (2.00x speedup). With 4 independent chains: **1.01 cycles/FMA** (3.97x speedup). With 8 independent chains: **1.01 cycles/FMA** (no further gain). The FP32 pipeline depth is confirmed to be 4 stages; 4 independent chains are necessary and sufficient to reach peak single-warp throughput.
