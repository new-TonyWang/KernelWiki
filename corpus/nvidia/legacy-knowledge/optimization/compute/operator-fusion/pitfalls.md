# Operator Fusion -- Pitfalls

## P1: Accidental FMA Prevention with Explicit Intrinsics
**Symptom**: Using `__fmul_rn(a, b)` followed by `__fadd_rn(result, c)` generates two separate instructions instead of one FMA, halving arithmetic throughput for that pattern.
**Detection**: Inspect SASS output; look for separate `FMUL` + `FADD` instead of `FFMA`.
**Fix**: Write `a * b + c` (compiler generates FMA by default) or use `fmaf(a, b, c)` / `__fmaf_rn(a, b, c)`. Only use explicit `__fmul_rn`/`__fadd_rn` when FMA prevention is intentionally required for numerical analysis.
**Source**: Programming Guide, Section 5.5.9.1 (Basic Intrinsic Functions) -- "functions map to addition and multiplication operations that the compiler never merges into FFMA"

## P2: Kernel Fusion Increasing Register Pressure Beyond Occupancy Threshold
**Symptom**: Fused kernel runs slower than two separate kernels because occupancy dropped significantly due to combined register pressure.
**Detection**: Compare register usage of fused vs unfused kernels using `--ptxas-options=-v`. Ncu shows reduced occupancy.
**Fix**: Use `__launch_bounds__` to cap register usage. If the fused kernel still requires too many registers, consider partial fusion (fuse only the cheaper operations) or use shared memory as a staging area between the two phases.
**Source**: Best Practices Guide, Section 10.2.7.1 (Register Pressure)

## P3: FMA Producing Different Results Than Separate Mul+Add
**Symptom**: Numerical results differ between CPU (which may not use FMA) and GPU (which uses FMA by default). Tests pass on CPU but fail on GPU.
**Detection**: Differences are typically small (1-2 ULP) but can be significant in iterative algorithms or comparisons with exact values.
**Fix**: This is expected behavior, not a bug. FMA is more accurate. If exact CPU-GPU matching is required, compile with `-fmad=false` to disable FMA fusion, or use matching FMA on the CPU side (`-mfma` compiler flag).
**Source**: Programming Guide, Section 5.5.1.5 (Fused Multiply-Add FMA)

## P4: Over-Fusing Makes Kernels Harder to Debug and Profile
**Symptom**: A complex fused kernel is difficult to isolate performance bottlenecks within, since all operations share the same kernel launch.
**Detection**: Ncu reports a single kernel with high complexity; cannot determine which operation is the bottleneck.
**Fix**: During development, keep operations separate for profiling. Fuse only after identifying the memory-bound bottleneck between operations. Consider conditional compilation to enable/disable fusion.
**Source**: General optimization principle

## P5: Mixed-Precision FMA Rounding Differences
**Symptom**: Mixed-precision `fma.rn.f32.f16` gives different rounding behavior than converting f16 to f32 first and then doing f32 FMA.
**Detection**: Subtle numerical differences in mixed-precision training/inference pipelines.
**Fix**: Understand that `fma.rn.f32.f16` computes the product of the f16 inputs at higher internal precision before adding the f32 accumulator, with a single final rounding to f32. This is actually more accurate. Accept the difference or use separate convert+FMA if exact behavior matching is needed.
**Source**: PTX ISA, fma.rnd.f32.{f16/bf16} instruction

## P6: The compiler already generates FMA by default with -fmad=true, so explicitly using __fmaf_rn on an already-FMA'd kernel produces no additional benefit — the baseline likely already had FMA instructions (discovered in verification)

**Symptom**: The compiler already generates FMA by default with -fmad=true, so explicitly using __fmaf_rn on an already-FMA'd kernel produces no additional benefit — the baseline likely already had FMA instructions.
**Source**: Level 3 sandbox verification (2026-04-06)

## P7: The optimized version actually executes more instructions (9 (discovered in verification)

**Symptom**: The optimized version actually executes more instructions (9.02M vs 8.45M, +6.7%) suggesting the epilogue fusion loop over fragment elements may generate additional overhead that partially offsets any register-residency benefit at this problem size.
**Source**: Level 3 sandbox verification (2026-04-06)

## P8: NCU per-kernel metrics (DRAM bytes, compute throughput, cycles) remain nearly unchanged after fusion, which can mislead into thinking fusion had no effect — the gain is in fewer kernel launches and eliminated inter-kernel global memory round-trips that NCU aggregates differently (discovered in verification)

**Symptom**: NCU per-kernel metrics (DRAM bytes, compute throughput, cycles) remain nearly unchanged after fusion, which can mislead into thinking fusion had no effect — the gain is in fewer kernel launches and eliminated inter-kernel global memory round-trips that NCU aggregates differently.
**Source**: Level 3 sandbox verification (2026-04-06)

## P9: Despite 46% faster wall time, the optimized kernel shows lower compute throughput (59 (discovered in verification)

**Symptom**: Despite 46% faster wall time, the optimized kernel shows lower compute throughput (59.8% vs 77.4% SM throughput) and higher long-scoreboard stalls (23.4% vs 11.7%), indicating the kernel became memory-latency-bound after instruction reduction — the compute pipe now finishes so fast that it spends more relative time waiting on memory.
**Source**: Level 3 sandbox verification (2026-04-06)
