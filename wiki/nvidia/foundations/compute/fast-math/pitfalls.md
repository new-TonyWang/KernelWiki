---
title: Fast Math Intrinsics - Pitfalls
status: draft
evidence_level: spec
applies_to_pattern_class:
- cuda-core
applies_to_ops:
- elementwise
- activation
- normalization
- softmax
requires_sm: '>=3.0'
single_kernel_useful: true
source:
- path: spec
  anchor: Reference
id: pitfall-fast-math
type: pitfall
vendor: nvidia
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L26993-L26995
- source_id: cuda-official/toolkit-docs-13.2
  path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L27041-L27045
- source_id: cuda-official/toolkit-docs-13.2
  path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L25507
- source_id: cuda-official/toolkit-docs-13.2
  path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1561
- source_id: cuda-official/toolkit-docs-13.2
  path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1370-L1374
- source_id: cuda-official/toolkit-docs-13.2
  path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L27017-L27023
---
## P1: Input-dependent ULP error for __expf

**Symptom**: Precision degrades progressively as `|x|` increases, eventually producing results with tens or hundreds of ULP error.

**Cause**: The documented max ULP error for `__expf(x)` is `2 + floor(abs(1.173 * x))` (programming guide Table 58, L26993-L26995). This is not a fixed bound like standard `expf` (which has 2 ULP max). For `|x| = 10`, the theoretical worst case is 13 ULP; for `|x| = 80` (near overflow), it is 95 ULP.

**Fix**: Restrict `__expf` to inputs with bounded magnitude. For activation functions like sigmoid (`expf(-x)` where `x` is a pre-activation value), inputs are typically in `[-10, 10]` where the error stays below ~13 ULP. For larger ranges, use standard `expf`.

**Detection**: Compare kernel output against a double-precision reference on representative inputs covering the full expected range.

## P2: --use_fast_math is global and includes hidden side effects

**Symptom**: Enabling `--use_fast_math` changes not only transcendental function precision but also division precision, subnormal handling, and FMA contraction behavior throughout the entire compilation unit.

**Cause**: `--use_fast_math` implies three additional flags (programming guide L25507, L25993-L25996):
- `-ftz=true`: flushes all subnormal (denormalized) fp32 values to zero.
- `-prec-div=false`: replaces `/` with `__fdividef` (2 ULP).
- `-fmad=true`: allows contraction of `a*b + c` into FFMA (changes rounding behavior).

These side effects affect every floating-point operation in the file, not just transcendentals.

**Fix**: Do not use `--use_fast_math` globally. Instead, call fast intrinsics explicitly (`__expf`, `__sinf`, etc.) only in the specific code paths where reduced precision is acceptable (programming guide L27045: "A more robust approach is to selectively replace mathematical function calls with intrinsic versions only where the performance gains justify it").

## P3: __sinf / __cosf lose accuracy outside [-pi, pi]

**Symptom**: Large absolute errors for trigonometric functions called with large arguments.

**Cause**: `__sinf(x)` and `__cosf(x)` have `2^-21.41` absolute error for `x in [-pi, pi]`, but the error is documented as "larger otherwise" (programming guide Table 58, L27017-L27023). Standard `sinf` performs multi-precision argument reduction to handle large arguments correctly, but `__sinf` skips this reduction entirely. For large `|x|`, the argument- reduction error dominates, and the result can be arbitrarily wrong.

**Fix**: Ensure inputs to `__sinf` / `__cosf` are reduced to `[-pi, pi]` before calling the intrinsic. If the application naturally produces arguments in this range (e.g., phase angles), the intrinsic is safe. For unbounded inputs, use standard `sinf` / `cosf`.

## P4: __fdividef produces 0 for large denominators

**Symptom**: Division returns 0 instead of a small but finite result.

**Cause**: `__fdividef(x, y)` has 2 ULP accuracy only for `|y| in [2^-126, 2^126]` (programming guide Table 58, L26985-L26988). For `|y| > 2^126`, the function returns 0 regardless of `x`. Standard division correctly handles these cases.

**Fix**: Guard `__fdividef` calls against extreme denominator magnitudes, or use standard division when the denominator range is not known.

## P5: rsqrtf not generated automatically

**Symptom**: `1.0f / sqrtf(x)` does not compile to a single `rsqrt` instruction, resulting in slower code than expected.

**Cause**: The compiler can only optimize `1.0f / sqrtf(x)` into `rsqrtf(x)` when both `-prec-div=false` and `-prec-sqrt=false` are active (best-practices guide L1372). With default compiler flags, the reciprocal and square root remain separate operations.

**Fix**: Call `rsqrtf(x)` directly instead of writing `1.0f / sqrtf(x)`. This gives 2 ULP accuracy in a single SFU operation regardless of compiler flags.

## P6: -ftz=true silently changes results near zero

**Symptom**: Computations involving very small (subnormal) floating-point values produce zero instead of a tiny nonzero result.

**Cause**: When `-ftz=true` is active (either explicitly or via `--use_fast_math`), all subnormal fp32 values are flushed to zero. This affects any computation whose intermediate or final result falls in the subnormal range `(0, 1.175e-38)`.

**Fix**: If the application requires correct subnormal behavior (e.g., computing log-probabilities of rare events, where `expf(x)` for very negative `x` produces a subnormal), do not use `-ftz=true` or `--use_fast_math`. Use explicit `__expf` calls instead, which respect the `-ftz` flag but can be controlled separately.

## P7: No speedup in memory-bound kernels

**Symptom**: Switching from `expf` to `__expf` produces no measurable throughput improvement.

**Cause**: When a kernel is bottlenecked by global memory bandwidth (e.g., a single `expf` call per element loaded from HBM), the compute pipeline is underutilized and both `expf` and `__expf` complete within the memory latency window. The probe confirmed this: with 1 call per element on a 4M-element array, both versions ran at ~0.015 ms with no speedup.

**Fix**: The fast-math speedup materializes only in compute-bound regions where transcendental operations dominate the instruction mix. Confirm that the kernel is compute-bound (e.g., via NCU's `sm__throughput` metric) before applying this optimization.

## P8: __powf accumulates errors from __log2f and exp2f

**Symptom**: `__powf(x, y)` produces larger errors than expected for a single intrinsic call.

**Cause**: `__powf(x, y)` is implemented as `exp2f(y * __log2f(x))` (programming guide Table 58, L27001-L27003). It compounds errors from both `__log2f` (2 ULP) and `exp2f`, so the resulting error can exceed what users expect from a "fast" single function call.

**Fix**: For small integer exponents, use explicit multiplication (`x*x`, `x*x*x`) which is faster and exact. For base-2 or base-10 exponentiation, use `exp2f` / `exp10f` directly (best-practices guide L1572). Only use `__powf` when the exponent is a non-integer runtime value and precision is not critical.
