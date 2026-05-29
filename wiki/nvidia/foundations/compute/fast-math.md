---
title: Fast Math Intrinsics
status: draft
evidence_level: measured
applies_to_pattern_class:
- cuda-core
applies_to_ops:
- elementwise
- activation
- normalization
- softmax
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
  code: sources/experience/hw-probes/fast-math/artifacts/expf_probe.cu
  build: nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo -o expf_probe expf_probe.cu
  introspection: ''
  profile: ''
related_apis:
- __expf
- __logf
- __sinf
- __cosf
- __powf
- __fdividef
- __fmaf_rn
related_skills:
- warp-primitives
- coalescing
id: skill-fast-math
type: skill
vendor: nvidia
tags:
- cuda-cpp
applies_to:
- general
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L26909-L26916
- source_id: cuda-official/toolkit-docs-13.2
  path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L26969-L27038
- source_id: cuda-official/toolkit-docs-13.2
  path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L27041-L27094
- source_id: cuda-official/toolkit-docs-13.2
  path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1243-L1244
- source_id: cuda-official/toolkit-docs-13.2
  path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1556-L1581
- source_id: cuda-official/toolkit-docs-13.2
  path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L26917-L26967
- source_id: cuda-official/toolkit-docs-13.2
  path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L25993-L25996
- source_id: cuda-official/toolkit-docs-13.2
  path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1370-L1374
---
## What

CUDA fast math intrinsics are hardware-accelerated approximate versions of standard C math functions, available only in device code. They trade precision for throughput by mapping more directly to the GPU's Special Function Unit (SFU) hardware.

There are two families:

**Approximate transcendentals** (single-precision only): `__sinf`, `__cosf`, `__expf`, `__logf`, `__log2f`, `__log10f`, `__exp10f`, `__powf`, `__sincosf`, `__tanf`, `__tanhf`, `__fdividef`. These have reduced precision (documented ULP error bounds in the programming guide Table 58) but are faster because they use fewer native instructions. The SFU processes approximate operations at 16 ops/clock/SM on all compute capabilities (best-practices guide Table 5).

**Rounded arithmetic intrinsics** (IEEE-754 exact, 0 ULP): `__fadd_rn`, `__fmul_rn`, `__fmaf_rn`, `__fsub_rn`, `__fdiv_rn`, `__frcp_rn`, `__fsqrt_rn`, `__frsqrt_rn` (and `_rz`, `_ru`, `_rd` variants). These have zero ULP error but prevent the compiler from fusing operations into FMA instructions, which is useful when strict IEEE-754 rounding semantics are needed. `__frsqrt_rn` gives 0 ULP reciprocal square root.

The nvcc compiler flag `--use_fast_math` globally replaces standard math functions with their `__` intrinsic counterparts for all single-precision calls. It also implies `-ftz=true` (flush denormals to zero), `-prec-div=false` (imprecise division), and `-fmad=true` (enable FMA contraction). The programming guide (Section 5.5.9.3, Table 59) lists the complete set of affected functions.

## Why

Standard math library functions like `expf` and `sinf` are implemented as multi-instruction software sequences. They perform range reduction, polynomial approximation, and IEEE-754 rounding correction to achieve 1-2 ULP accuracy. This costs many instructions per call.

Fast math intrinsics like `__expf` bypass most of these correction steps and map more directly to the SFU `ex2.approx.f32` (or `lg2.approx.f32`, `sin.approx.f32`, etc.) instructions. The SFU has a dedicated pipeline with throughput of 16 ops/clock/SM on sm_9.0, compared to 128 ops/clock/SM for basic fp32 add/multiply. Standard `expf` issues multiple instructions including SFU calls plus correction arithmetic, so its effective throughput is lower than the SFU's raw 16 ops/clock rate. `__expf` reduces that instruction overhead.

Measured on H200 (sm_90a), `__expf` achieves 1.27x higher throughput than `expf` in a compute-bound loop (3318 Gop/s vs 2604 Gop/s). See Measured Characteristics below for the full probe record.

## When to use

- **Activation functions** (GeLU, SiLU/Swish, sigmoid, softmax-exp): these apply expf per element and tolerate the ~8 ULP error of `__expf` because the downstream operations (softmax normalization, gradient computation at reduced precision) absorb the small inaccuracy.

- **Normalization inverse-root** (LayerNorm, RMSNorm): `rsqrtf` (fast reciprocal square root, 2 ULP from Table 58) is the canonical single-instruction path. The compiler will use `rsqrtf` automatically only when `-prec-div=false` and `-prec-sqrt=false` are both active; the best-practices guide (L1372) recommends invoking `rsqrtf()` explicitly.

- **Softmax denominator**: the exp inside softmax is typically followed by a reduction sum, so per-element errors of 2-8 ULP are negligible relative to the sum magnitude.

- **Fast division**: `__fdividef(x, y)` gives 2 ULP error for `|y| in [2^-126, 2^126]` (Table 58, L26985-L26988). Faster than the default `/` operator when IEEE-754 precision is not required.

- **Trigonometric functions in signal processing kernels**: `__sinf` / `__cosf` are accurate to `2^-21.41` absolute error within `[-pi, pi]` (Table 58, L27017-L27023). Outside this range, error grows because the intrinsic skips the multi-precision argument reduction that the standard `sinf` performs.

## When NOT to use

- **Loss computation**: Cross-entropy and similar losses involve `log` of softmax outputs. Precision errors in `__logf` (3 ULP outside `[0.5, 2]`) can produce incorrect gradients and destabilize training.

- **Gradient accumulation**: FP32 gradient accumulators are used precisely to maintain precision. Substituting `__expf` or `__logf` in the backward pass undermines this goal.

- **Numerically sensitive reductions**: any code path where small per-element errors compound (e.g., prefix scan of log-probabilities, Kahan summation) should use standard library functions.

- **Large-argument trigonometry**: `__sinf(x)` for `|x| >> pi` gives unbounded error because it skips argument reduction. Standard `sinf` handles this correctly (at the cost of local memory for argument reduction when `|x|` is large, per best-practices guide L1561).

- **Mixed-precision training master weights**: The fp32 master copy of weights is updated in full precision. Fast math intrinsics in the update step corrupt the precision advantage of maintaining fp32 weights.

- **`--use_fast_math` on the entire compilation unit**: This flag globally downgrades every single-precision function. A more robust approach is to selectively call `__expf` / `__sinf` / etc. only where the precision loss is acceptable (programming guide L27045).

## Classical example: fast sigmoid

```cuda
// Standard sigmoid: uses expf (multi-instruction)
__device__ __forceinline__ float sigmoid_std(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Fast sigmoid: uses __expf (SFU-direct)
__device__ __forceinline__ float sigmoid_fast(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

// Even faster: use __fdividef to also speed up the division
__device__ __forceinline__ float sigmoid_fastest(float x) {
    return __fdividef(1.0f, 1.0f + __expf(-x));
}

__global__ void apply_sigmoid(const float* __restrict__ in,
                              float*       __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = sigmoid_fast(in[i]);
}
```

For softmax:

```cuda
// Fast softmax numerator: __expf for the exponential
__device__ float fast_softmax_exp(float x, float row_max) {
    return __expf(x - row_max);  // subtract max for numerical stability
}

// Fast reciprocal for normalization
__device__ float fast_softmax_norm(float x, float sum) {
    return __fdividef(x, sum);
}
```

For LayerNorm inverse root:

```cuda
// Direct rsqrtf call (2 ULP, recommended over 1.0f/sqrtf(x))
__device__ float fast_inv_sqrt(float variance_plus_eps) {
    return rsqrtf(variance_plus_eps);
}
```

## Measured Characteristics

- expf-vs-fast-expf probe: On H200 (sm_90a, CUDA 12.9), a compute-bound loop of 128 chained expf calls per thread (1M threads) measured `__expf` at **0.0404 ms** (3318 Gop/s) vs standard `expf` at **0.0516 ms** (2604 Gop/s), a **1.27x speedup**. Single-call precision: `expf` max ULP error = 2, `__expf` max ULP error = 8, both measured against double-precision reference over 1M random inputs in [-10, 10]. The speedup vanishes in memory-bound kernels where both versions are bottlenecked by global memory bandwidth.
