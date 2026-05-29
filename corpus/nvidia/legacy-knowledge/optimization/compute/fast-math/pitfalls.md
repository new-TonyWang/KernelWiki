# Fast Math -- Pitfalls

## P1: --use_fast_math Silently Reduces All Single-Precision Accuracy
**Symptom**: Numerical results diverge from reference CPU implementation beyond expected tolerance. Loss of special-value handling (NaN, Inf).
**Detection**: Validation tests fail on edge cases (e.g., very large arguments to sinf, division by values near zero). Denormalized numbers produce zero.
**Fix**: Remove `-use_fast_math` and selectively replace only the specific function calls where reduced accuracy is acceptable. Use `-ftz=true`, `-prec-div=false`, `-prec-sqrt=false` individually for finer control.
**Source**: Best Practices Guide, Section 12.1.9 (Math Libraries); Programming Guide, Section 5.5.9.3 (--use_fast_math Effect)

## P2: Large-Argument Error in __sinf / __cosf
**Symptom**: `__sinf(x)` and `__cosf(x)` produce large errors when `|x|` is much larger than pi.
**Detection**: Compare output against `sinf(x)` for values outside `[-pi, pi]`. Error grows as `|x|` increases. Standard `sinf` uses argument reduction that `__sinf` omits.
**Fix**: Reduce the argument to `[-pi, pi]` manually before calling `__sinf`, or use `sinpif(x / M_PI)` if the argument is in degrees/radians multiples of pi. For large arguments, stick with `sinf`.
**Source**: Programming Guide, Section 5.5.9.2 (Single-Precision-Only Intrinsic Functions) -- __sinf error grows outside `[-pi, pi]`

## P3: Double-Precision Constants Causing Implicit Promotion
**Symptom**: Single-precision kernel runs slower than expected. Ncu shows FP64 pipe utilization in a kernel that should be pure FP32.
**Detection**: Check SASS for `DADD`, `DMUL`, or `CVT` double-to-float instructions. Look for floating-point literals without the `f` suffix.
**Fix**: Add `f` suffix to all single-precision constants: `3.14f`, `0.5f`, `1.0f`. Use `-Xptxas -warn-double-usage` to detect double usage at compile time.
**Source**: Best Practices Guide, Section 12.1.7 (Other Arithmetic Instructions)

## P4: Using pow() for Simple Exponentiation
**Symptom**: `powf(x, 2.0f)` or `powf(x, 0.5f)` generates many instructions and high register pressure instead of a simple multiply or sqrt.
**Detection**: Ncu shows unexpectedly high instruction count for a simple kernel. Inspect SASS for heavy `pow` instruction sequences.
**Fix**: Replace `powf(x, 2.0f)` with `x * x`, `powf(x, 0.5f)` with `sqrtf(x)`, `powf(x, -0.5f)` with `rsqrtf(x)`, `powf(x, 1.0f/3.0f)` with `cbrtf(x)`. For base-2/10, use `exp2f`/`exp10f`.
**Source**: Best Practices Guide, Section 12.1.8 (Exponentiation With Small Fractional Arguments); Section 12.1.9 (Math Libraries)

## P5: Denormal Flush-to-Zero Breaks Algorithms
**Symptom**: Algorithms that rely on gradual underflow (e.g., certain compensated summation schemes, comparisons near zero) produce incorrect results when `-ftz=true` is enabled.
**Detection**: Results differ from CPU reference only for very small values near the denormal range.
**Fix**: Disable FTZ for the affected kernels. Compile sensitive kernels in a separate translation unit without `-ftz=true` or `-use_fast_math`.
**Source**: Best Practices Guide, Section 12.1.10 (Precision-related Compiler Flags)

## P6: Integer Division in Index Computation
**Symptom**: Index calculations using integer division are much slower than expected (20+ cycles per division on recent architectures).
**Detection**: Ncu shows high arithmetic instruction count in address computation code.
**Fix**: Use power-of-two dimensions where possible and replace division/modulo with shifts and masks. For non-power-of-two divisors, consider strength reduction using the multiply-high trick: `(uint64_t(x) * magic) >> shift`.
**Source**: Best Practices Guide, Section 12.1.4 (Division Modulo Operations)

## P7: Intrinsic replacement yields no benefit when the kernel is memory-latency-bound; the instruction count (3,407,872) and register usage (16) are identical between rounds, suggesting the compiler may already be emitting the fast-path instruction (e (discovered in verification)

**Symptom**: Intrinsic replacement yields no benefit when the kernel is memory-latency-bound; the instruction count (3,407,872) and register usage (16) are identical between rounds, suggesting the compiler may already be emitting the fast-path instruction (e.g. tanh.approx.f32) for this simple elementwise pattern.
**Source**: Level 3 sandbox verification (2026-04-04)

## P8: --use_fast_math is a no-op on kernels dominated by memory ops or simple arithmetic (add/mul/fma) that are already IEEE-compliant fast paths; the skill description should emphasize it only helps kernels with transcendental or precision-sensitive math calls (discovered in verification)

**Symptom**: --use_fast_math is a no-op on kernels dominated by memory ops or simple arithmetic (add/mul/fma) that are already IEEE-compliant fast paths; the skill description should emphasize it only helps kernels with transcendental or precision-sensitive math calls.
**Source**: Level 3 sandbox verification (2026-04-04)

## P9: The skill's claim that 'the compiler may not always optimize 1 (discovered in verification)

**Symptom**: The skill's claim that 'the compiler may not always optimize 1.0f/sqrtf(x) to rsqrtf due to IEEE-754 compliance' does not hold for modern nvcc at default settings with float32; this transformation appears to happen automatically.
**Source**: Level 3 sandbox verification (2026-04-04)

## P10: Profiling timeout on both baseline and optimized means the verification harness or kernel workload is misconfigured for this skill — the problem is infrastructure, not the optimization technique itself (discovered in verification)

**Symptom**: Profiling timeout on both baseline and optimized means the verification harness or kernel workload is misconfigured for this skill — the problem is infrastructure, not the optimization technique itself.
**Source**: Level 3 sandbox verification (2026-04-04)

## P11: Replacing powf with specialized function chains can increase instruction count if the compiler already performs these substitutions internally; always check SASS output before assuming source-level math substitutions help (discovered in verification)

**Symptom**: Replacing powf with specialized function chains can increase instruction count if the compiler already performs these substitutions internally; always check SASS output before assuming source-level math substitutions help.
**Source**: Level 3 sandbox verification (2026-04-04)

## P12: sincosf is a source-level readability improvement but not a performance optimization on modern CUDA toolchains — the compiler already performs this fusion automatically (discovered in verification)

**Symptom**: sincosf is a source-level readability improvement but not a performance optimization on modern CUDA toolchains — the compiler already performs this fusion automatically.
**Source**: Level 3 sandbox verification (2026-04-04)

## P13: The dramatic instruction reduction (58 (discovered in verification)

**Symptom**: The dramatic instruction reduction (58.5%) translates to only ~10% wall-clock speedup because removing compute work exposes memory latency as the new bottleneck — on memory-bound kernels, float suffix optimization alone will show negligible timing improvement despite fewer instructions.
**Source**: Level 3 sandbox verification (2026-04-04)
