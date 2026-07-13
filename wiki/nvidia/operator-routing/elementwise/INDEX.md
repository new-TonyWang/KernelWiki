---
title: Elementwise Operator Pattern
pattern_class: cuda-core
op: elementwise
covers:
- add
- mul
- activation (relu, gelu, silu, swish, sigmoid)
- cast (fp32 to fp16, bf16 to fp32, etc.)
- fused chains (bias+activation, scale+add)
hardware:
  device: H200
  sm: 9.0a
source:
- path: spec
  anchor: Reference
id: routing-elementwise-INDEX
type: operator-routing
vendor: nvidia
operator: elementwise
source_refs:
- source_id: source-code/cuda-samples
  path: Samples/0_Introduction/vectorAdd/vectorAdd.cu
  anchor: L47-L54
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L539-L542
architectures:
- sm90
- sm90a
languages:
- cuda-cpp
- python
- triton
techniques:
- pipeline-stages
- vectorized-loads
- data-reuse
- kernel-fusion
- shared-memory-optimization
- software-exp
kernel_types:
- fused-kernel
- attention
confidence: inferred
tags:
- pipeline-stages
- vectorized-loads
- data-reuse
- kernel-fusion
- shared-memory-optimization
- software-exp
- fused-kernel
- attention
- cuda-cpp
- python
- triton
---
# Elementwise Operator Pattern -- Decision Tree

Elementwise operations apply a scalar function independently to each element (or corresponding pair/tuple of elements) of one or more tensors. Examples: add, mul, relu, gelu, silu, sigmoid, cast, scale+add, bias+activation.

The kernel shape is always *one logical thread per element* (or per vector of elements when using vectorized loads). Because the per-element computation is trivial (a few FLOPs at most), elementwise kernels are **purely memory-bound**: the optimization target is maximizing effective global-memory bandwidth, not arithmetic throughput.

---

## Decision tree

```
Step 0  Confirm this is a custom elementwise-kernel task.
 |
 |  - Requirements include a custom layout, fusion pattern, or measured gap.
 |  - The remaining tree selects dtype, thread mapping, and optimization skills.
 |

Step 1  Write a custom elementwise kernel.
 |
 The canonical structure is:
 |
 |   __global__ void elementwise_kernel(const T* in, T* out, int N) {
 |       int tid = blockIdx.x * blockDim.x + threadIdx.x;
 |       if (tid < N)
 |           out[tid] = f(in[tid]);
 |   }
 |
 |  - One thread per element (or per vector of elements).
 |  - Thread index == array index ensures coalesced access.
 |  - The math in f() is negligible; this is a fancy memcpy.
 |
 +---> 1a. CHOOSE DTYPE FIRST (precision + range decision).
 |       BEFORE selecting any optimization skill, pin down T.
 |       This is a CORRECTNESS decision, not a perf decision.
 |       Reference: wiki/nvidia/foundations/compute/half-precision-math/ §Precision.
 |
 |       Decision questions (answer in order):
 |
 |       Q1a.  Do values in `in[]` stay within [6.1e-5, 6.55e4]
 |             AND is the per-element computation a straight map
 |             (no sum over many terms, no softmax denominator,
 |             no running average)?
 |             YES --> fp16 is safe. Best pick on H200: 2x memory
 |                     bandwidth, 1.58x compute vs fp32 scalar.
 |             NO  --> Q1b.
 |
 |       Q1b.  Do values potentially exceed fp16's 6.55e4 ceiling
 |             but stay within fp32's range (3.4e38)?
 |             YES --> bf16 for storage. Dynamic range is fp32-
 |                     level but precision is COARSER than fp16
 |                     (eps ~7.8e-3 vs 4.9e-4). Acceptable for
 |                     training-friendly activations; risky for
 |                     thresholded decisions. On H200, bf16 packed
 |                     is 12% slower than fp16 packed but still
 |                     ~1.6x fp32 scalar.
 |             NO  --> fp32 (or fp64 if precision-critical). Do
 |                     not force half precision when the range or
 |                     mantissa resolution is insufficient.
 |
 |       Q1c.  If ANY of the below applies, USE fp32 regardless of
 |             the answers to Q1a/Q1b:
 |               - The op accumulates > 100 terms (sum, dot, soft-
 |                 max denominator, variance, running mean) without
 |                 a fp32 promotion in the body.
 |               - A threshold comparison is within 1% of a
 |                 decision boundary (e.g., ReLU with tiny eps,
 |                 conditional write with near-zero predicate).
 |               - The output feeds into a numerically sensitive
 |                 downstream consumer (matrix inverse, log-sum-
 |                 exp over wide dynamic range, argmax over near-
 |                 ties).
 |
 |       Q1d.  Mixed-precision escape hatch: if inputs are fp16/
 |             bf16 but the op internally accumulates, use the
 |             canonical pattern:
 |               float acc = 0.0f;
 |               for each i:
 |                 acc = fmaf(__half2float(in[i]), ..., acc);
 |               out[i] = __float2half_rn(acc);
 |             This preserves the memory-bandwidth win (2x) while
 |             protecting correctness. See half-precision-math §S3.
 |
 +---> Go to Step 2.

Step 2  Optimize via skills from ROUTING.md.
 |
 Apply skills iteratively following bottleneck-triage.md:
 |
 +---> 2a. Coalescing (critical for every elementwise kernel).
 |       Ensure stride-1 access: tid = blockIdx.x * blockDim.x + threadIdx.x
 |       and data[tid] indexing.
 |       See: wiki/nvidia/foundations/memory/coalescing/
 |
 +---> 2b. Vectorized loads (float4 / half8).
 |       When dtype size * 4 fits a vector register, cast pointers
 |       to float4* and load one vector per thread.  This quadruples
 |       the bytes per load instruction and reduces instruction count.
 |       NOTE: vectorized-access skill is not yet built in this KB.
 |       The principle is: use reinterpret_cast<float4*> and
 |       process 4 elements per thread.
 |
 +---> 2c. Grid-stride loop for large N.
 |       When N >> gridDim.x * blockDim.x, use:
 |         for (int i = tid; i < N; i += gridDim.x * blockDim.x)
 |             out[i] = f(in[i]);
 |       This allows a fixed grid size and amortizes launch overhead.
 |
 +---> 2d. Bank conflict avoidance (rarely needed).
 |       Only relevant if the kernel stages data through __shared__
 |       memory (uncommon for pure elementwise).
 |       See: wiki/nvidia/foundations/memory/bank-conflict/
 |
 +---> DONE.  Measure effective bandwidth vs. device peak.
        For a well-optimized elementwise kernel on H200, expect
        to reach 80-95% of peak memory bandwidth (~3.35 TB/s HBM3e).
```

---

## Elementwise kernel characteristics

| Property | Value |
|---|---|
| Bottleneck | Memory bandwidth (always) |
| Compute intensity | Near zero (1-5 FLOPs per element loaded) |
| Shared memory | Not needed (no data reuse across threads) |
| Thread mapping | 1:1 or 1:vector (tid == element index) |
| Key optimization | Coalesced global loads/stores |
| Secondary optimization | Vectorized loads (float4 / int4) |
| Target bandwidth | 80-95% of peak HBM bandwidth |
