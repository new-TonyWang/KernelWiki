---
title: Pooling Pattern -- Decision Tree
pattern_class: cuda-core
op: pooling
covers:
- max-pool (1D/2D/3D)
- avg-pool (1D/2D/3D)
- adaptive-avg-pool
- adaptive-max-pool
status: draft
hardware:
  device: H200
  sm: 9.0a
source:
- path: spec
  anchor: Reference
id: routing-pooling-INDEX
type: operator-routing
vendor: nvidia
operator: pooling
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L538-L542
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L719-L726
architectures:
- sm90
- sm90a
languages:
- cuda-cpp
techniques:
- pipeline-stages
- vectorized-loads
- kernel-fusion
- loop-unrolling
- shared-memory-optimization
- tile-scheduling
- communication-overlap
kernel_types:
- fused-kernel
- attention
confidence: inferred
tags:
- pipeline-stages
- vectorized-loads
- kernel-fusion
- loop-unrolling
- shared-memory-optimization
- tile-scheduling
- communication-overlap
- fused-kernel
- attention
- cuda-cpp
---
# Pooling Pattern -- Decision Tree

This document guides the kernel-writing agent through a pooling task from initial problem statement to a working, optimized kernel. The decision tree enforces a **library-first** policy: only proceed to a custom kernel when the library path has been proven insufficient.

Pooling operators compute a windowed reduction over spatial dimensions of a tensor. Common variants: max-pool, avg-pool, and adaptive-pool in 1D, 2D, and 3D. The input is typically in NCHW (or NHWC) layout, and the output spatial dimensions are determined by kernel size, stride, and padding.

## Step 0 -- Try the library first

Before writing any custom CUDA code, check whether a production-quality library already handles the pooling operation.

```
Q0. Is the caller's environment PyTorch-based?
    YES --> Can torch.nn.functional.max_pool2d / avg_pool2d /
            adaptive_avg_pool2d / adaptive_max_pool2d handle the
            shape + dtype + kernel_size + stride + padding?
            YES --> Use the PyTorch op (cuDNN backend). DONE.
            NO  --> Continue to Q1.
    NO  --> Continue to Q1.

Q1. Is the caller using CUDA C++ and can invoke cuDNN?
    YES --> Use cudnnPoolingForward with the appropriate pooling
            mode (CUDNN_POOLING_MAX, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
            CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING).
            cuDNN pooling is heavily optimized for standard window sizes
            and NCHW/NHWC layouts.
            See library-fallback.md for API details. DONE.
    NO  --> Continue to Q2.

Q2. Does the library path fail to meet performance requirements after
    benchmarking, or does the use case require a feature the library
    does not support?
    YES --> Proceed to Step 1 (custom kernel).
    NO  --> Re-examine the library path. cuDNN pooling covers the
            vast majority of standard use cases.
```

**When to skip the library**: the library path is insufficient when:
- The pooling must be fused with a preceding or following operation (e.g., pooling + activation, pooling + batch-norm) to avoid an extra global-memory round-trip.
- A non-standard window shape or reduction rule is needed (e.g., weighted pooling, Lp-norm pooling, stochastic pooling).
- An unusual padding mode is required that the library does not support (e.g., reflection padding combined with pooling).
- The measured library latency exceeds the theoretical bandwidth-bound limit by more than 10% for the given shape.
- The operation is a small component of a larger fused kernel that must remain in a single launch (e.g., pooling inside a custom attention block).

## Step 1 -- Choose the custom pooling strategy

A pooling kernel is structurally a **2D windowed reduction**: for each output element, iterate over the pooling window in the input and accumulate a max or sum, then (for avg-pool) divide by the window size.

```
Q3. What is the pool type?
    max-pool
        --> Accumulate with max(acc, input_val) over the window.
            Initialize acc to -infinity (or the smallest representable
            value for the dtype).

    avg-pool
        --> Accumulate with sum(input_val) over the window.
            Divide by window_area at the end.
            For count_include_padding=true, window_area = kH * kW.
            For count_include_padding=false, window_area = number of
            valid (non-padded) input positions in the window.

    adaptive-pool
        --> The output size is fixed; kernel_size and stride are
            computed per output position:
              kH_start = floor(oh * iH / oH)
              kH_end   = floor((oh + 1) * iH / oH)
            Then reduce over input[kH_start:kH_end, kW_start:kW_end].
            This is equivalent to a variable-size window pooling.

Q4. Thread-to-output mapping?
    One thread per output element (the standard approach):
        tid maps to (n, c, oh, ow) in the output tensor.
        The thread iterates over the kernel_size x kernel_size window
        in the input, reading input[n, c, oh*stride+kh, ow*stride+kw]
        for kh in [0, kH), kw in [0, kW).

    Alternative -- one thread per input element with atomics:
        Rarely beneficial. The standard one-thread-per-output approach
        avoids atomics entirely and is simpler.
```

## Step 2 -- Optimization via ROUTING.md skills

After the basic custom kernel is working and correct, apply optimization skills from ROUTING.md in priority order:

1. **Coalescing** (wiki/nvidia/foundations/memory/coalescing/) -- ensure the output store is coalesced (adjacent threads write to adjacent memory locations). For NCHW layout, this means the innermost loop dimension (W) should map to consecutive threads. The input loads within the pooling window will have spatial locality but may not be perfectly coalesced depending on the stride and window size.

2. **Bank-conflict avoidance** (wiki/nvidia/foundations/memory/bank-conflict/) -- if the kernel stages input tiles into shared memory to reduce redundant global reads (the input windows of neighboring output elements overlap), ensure the shared memory layout avoids bank conflicts. Padding the shared memory tile width by 1 element (`__shared__ float tile[TILE_H][TILE_W + 1]`) is the standard fix.

3. **Vectorized access** (wiki/nvidia/foundations/memory/vectorized-access/) -- when the channel dimension is the innermost dimension (NHWC layout) and multiple channels can be loaded together, use float4 or float2 loads. For NCHW with a single channel per thread, vectorization applies to the width dimension if stride == 1 and kW is small.

4. **Instruction-level parallelism** (wiki/nvidia/foundations/compute/ilp/) -- within the pooling window loop, unrolling with `#pragma unroll` exposes independent loads and comparisons to the instruction scheduler. Most beneficial for larger window sizes (5x5, 7x7).

After each skill application, re-benchmark against the baseline (torch.nn.functional.max_pool2d / avg_pool2d or cuDNN) and follow the bottleneck-triage procedure in reasoning/bottleneck-triage.md.

## Step 3 -- Shared memory tiling for large windows

When the pooling window is large (kernel_size >= 5) or the stride is small (stride < kernel_size, i.e., overlapping windows), neighboring output elements read overlapping regions of the input. Loading the input tile into shared memory amortizes the redundant global reads:

```cuda
// Example: shared memory tiling for 2D pooling
// Each block computes a tile of output elements.
// The shared memory tile covers the corresponding input region
// (output_tile_H * stride + kH - stride) x (output_tile_W * stride + kW - stride).

__shared__ float smem[INPUT_TILE_H][INPUT_TILE_W + 1]; // +1 for bank-conflict padding

// Phase 1: cooperatively load the input tile into smem
// Phase 2: each thread reduces over its kH x kW window in smem
// Phase 3: write the output element to global memory
```

This approach trades shared memory capacity for reduced global memory traffic. The break-even point depends on the overlap ratio:
- stride == kernel_size (no overlap): no benefit from shared memory.
- stride == 1 (maximum overlap): significant benefit, especially for large windows. Each input element is read by up to kH * kW output elements; shared memory reduces this to one global read per element.

## Cross-references

- **Library fallback details**: `library-fallback.md`
- **Skill whitelist for this pattern**: `ROUTING.md`
- **Task packet template**: `TASK-PACKET.md`
- **Bottleneck triage after benchmarking**: `reasoning/bottleneck-triage.md`
