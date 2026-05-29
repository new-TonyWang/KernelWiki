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
  path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L539-L542
---
# Elementwise Operator Pattern -- Decision Tree

Elementwise operations apply a scalar function independently to each element (or corresponding pair/tuple of elements) of one or more tensors. Examples: add, mul, relu, gelu, silu, sigmoid, cast, scale+add, bias+activation.

The kernel shape is always *one logical thread per element* (or per vector of elements when using vectorized loads). Because the per-element computation is trivial (a few FLOPs at most), elementwise kernels are **purely memory-bound**: the optimization target is maximizing effective global-memory bandwidth, not arithmetic throughput.

---

## Decision tree

```
Step 0  Can a library / framework handle this?
 |
 +---> Q0a. Is the operation a single built-in torch op?
 |       (torch.add, torch.mul, torch.relu, torch.sigmoid, ...)
 |       YES --> Use it.  PyTorch's ATen kernels are already fused
 |               and vectorized for contiguous tensors.  STOP.
 |       NO  --> Q0b.
 |
 +---> Q0b. Is this a short chain of ops that torch.compile can fuse?
 |       (e.g., x * scale + bias, followed by gelu)
 |       YES --> Wrap in a function, call torch.compile().
 |               The Triton codegen in torch.compile fuses
 |               elementwise chains into a single kernel
 |               automatically.  STOP.
 |       NO  --> Q0c.
 |
 +---> Q0c. Can thrust::transform express the operation?
 |       (simple unary or binary functor on device_vector)
 |       YES --> Use thrust::transform for a quick C++ prototype.
 |               Thrust generates a reasonably efficient kernel
 |               under the hood.  STOP.
 |       NO  --> Go to Step 1 (custom kernel).

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

## Why library-first?

Writing a custom CUDA kernel for elementwise operations is almost never necessary when working from Python:

1. **PyTorch built-in ops** (torch.add, torch.relu, etc.) use ATen kernels that are already vectorized and coalesced for contiguous tensors. They match or approach peak bandwidth.

2. **torch.compile** fuses short chains of pointwise ops into a single Triton kernel, eliminating intermediate materializations. This is the recommended path for fused elementwise chains like `gelu(x * W + b)`.

3. **thrust::transform** covers the C++ prototyping case -- the user provides a functor, and Thrust handles grid sizing and memory management.

A custom kernel is justified only when:
- The fusion pattern is not expressible by torch.compile (rare for pure elementwise).
- The operation requires a custom data layout or in-place update that the library does not support.
- Profiling shows the library version is measurably slower (e.g., non-contiguous tensors causing uncoalesced access in the ATen path).

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
