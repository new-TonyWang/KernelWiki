# Compiler Hints -- Pitfalls

## P1: __restrict__ Increasing Register Pressure
**Symptom**: Adding `__restrict__` to all pointers causes occupancy to drop because the compiler caches more loads in registers.
**Detection**: `--ptxas-options=-v` shows increased register count after adding `__restrict__`. Ncu shows reduced occupancy.
**Fix**: Use `__restrict__` in combination with `__launch_bounds__` to cap the register budget. The trade-off is between fewer memory operations (from alias elimination) and higher register pressure.
**Source**: Programming Guide, Section 5.4.1.4 (__restrict__ Pointers) -- "the use of restricted pointers can negatively impact performance by reducing occupancy"

## P2: __launch_bounds__ maxThreadsPerBlock Not Usable in Host Code
**Symptom**: Kernel launched with fewer threads than intended because `MY_KERNEL_MAX_THREADS` depends on `__CUDA_ARCH__`, which is undefined in host code.
**Detection**: Kernel launches with the default (non-arch-dependent) value instead of the intended value.
**Fix**: Determine the number of threads at runtime using `cudaGetDeviceProperties` to check the compute capability, or use a compile-time constant that does not depend on `__CUDA_ARCH__`.
**Source**: Programming Guide, Section 5.4.3.2 (Launch Bounds)

## P3: Over-Aggressive Unrolling Bloating Code Size
**Symptom**: `#pragma unroll` on a loop with large or unknown trip count causes enormous code generation, filling the instruction cache and potentially harming performance.
**Detection**: Compilation is very slow; resulting binary is much larger than expected. Ncu shows high instruction cache miss rate (`sm__sass_inst_cache_miss_rate`).
**Fix**: Use `#pragma unroll N` with a specific factor (e.g., 4 or 8) instead of full unrolling. Use `#pragma unroll 1` to disable unrolling when the compiler over-unrolls.
**Source**: Programming Guide, Section 5.4.9.1 (#pragma unroll)

## P4: __builtin_assume with False Predicate Causes Undefined Behavior
**Symptom**: Silent incorrect results or crashes when the assumed condition is not actually true at runtime.
**Detection**: Extremely difficult to detect. The compiler optimizes based on the assumption, potentially removing safety checks.
**Fix**: Only use `__builtin_assume` when the condition is provably always true. Do not use it as an optimization hint for "usually true" conditions -- use `__builtin_expect` for that purpose instead.
**Source**: Programming Guide, Section 5.4.9.3 (__builtin_assume)

## P5: Unsigned Loop Counters Preventing Strength Reduction
**Symptom**: Loops with unsigned counters and non-trivial stride expressions generate more instructions than expected.
**Detection**: Compare SASS output between signed and unsigned loop counter versions.
**Fix**: Declare loop counters as `int` (signed) rather than `unsigned int`. The compiler can apply more aggressive optimizations (strength reduction, induction variable elimination) when overflow is undefined.
**Source**: Best Practices Guide, Section 12.1.5 (Loop Counters Signed vs. Unsigned)

## P6: Modifying __grid_constant__ Parameters
**Symptom**: Undefined behavior when writing to or modifying a `__grid_constant__` parameter or its sub-objects, including `mutable` members.
**Detection**: May manifest as silent data corruption or intermittent incorrect results.
**Fix**: Never modify a `__grid_constant__` parameter. Copy to a local variable first if modification is needed. All declarations of the function must be consistent with the annotation.
**Source**: Programming Guide, Section 5.4.1.5 (__grid_constant__ Parameters)

## P7: Long scoreboard stalls remain dominant (88 (discovered in verification)

**Symptom**: Long scoreboard stalls remain dominant (88.35%), meaning the kernel is still memory-latency-limited even after __restrict__; further gains would require latency hiding (e.g., higher occupancy or instruction-level parallelism), not just alias elimination.
**Source**: Level 3 sandbox verification (2026-04-04)

## P8: On already-simple kernels with low register pressure, compiler flags can slightly degrade performance (SM throughput dropped ~1%) or produce no measurable change, giving a false sense of optimization effort (discovered in verification)

**Symptom**: On already-simple kernels with low register pressure, compiler flags can slightly degrade performance (SM throughput dropped ~1%) or produce no measurable change, giving a false sense of optimization effort.
**Source**: Level 3 sandbox verification (2026-04-04)

## P9: On simple kernels with low register usage, __launch_bounds__ has zero effect because the compiler already satisfies the occupancy constraint naturally; the skill only matters for register-heavy kernels where spilling trades registers for occupancy (discovered in verification)

**Symptom**: On simple kernels with low register usage, __launch_bounds__ has zero effect because the compiler already satisfies the occupancy constraint naturally; the skill only matters for register-heavy kernels where spilling trades registers for occupancy.
**Source**: Level 3 sandbox verification (2026-04-05)

## P10: Register usage nearly doubled (16→30 per thread), cutting register-limited occupancy from 16 to 8 warps and active warps from 89% to 73%; for compute-bound kernels this occupancy loss could negate the ILP gains (discovered in verification)

**Symptom**: Register usage nearly doubled (16→30 per thread), cutting register-limited occupancy from 16 to 8 warps and active warps from 89% to 73%; for compute-bound kernels this occupancy loss could negate the ILP gains.
**Source**: Level 3 sandbox verification (2026-04-05)

## P11: nvcc's PTX backend largely ignores __builtin_assume_aligned; alignment hints matter more for host compilers (GCC/Clang) than for the CUDA device compiler, which infers alignment from cast patterns like ((float4*)ptr)[idx] (discovered in verification)

**Symptom**: nvcc's PTX backend largely ignores __builtin_assume_aligned; alignment hints matter more for host compilers (GCC/Clang) than for the CUDA device compiler, which infers alignment from cast patterns like ((float4*)ptr)[idx].
**Source**: Level 3 sandbox verification (2026-04-05)

## P12: Signed counters increased register usage from 23 to 32 per thread, reducing register-limited occupancy from 10 to 8 blocks; on register-starved kernels this trade-off could backfire (discovered in verification)

**Symptom**: Signed counters increased register usage from 23 to 32 per thread, reducing register-limited occupancy from 10 to 8 blocks; on register-starved kernels this trade-off could backfire.
**Source**: Level 3 sandbox verification (2026-04-05)

## P13: These compiler hints are essentially no-ops for memory-bound kernels where execution time is dominated by DRAM latency, not branch misprediction or instruction scheduling (discovered in verification)

**Symptom**: These compiler hints are essentially no-ops for memory-bound kernels where execution time is dominated by DRAM latency, not branch misprediction or instruction scheduling.
**Source**: Level 3 sandbox verification (2026-04-05)

## P14: __grid_constant__ is a no-op optimization for small scalar parameter structs; the compiler already keeps them in constant memory/registers without the annotation (discovered in verification)

**Symptom**: __grid_constant__ is a no-op optimization for small scalar parameter structs; the compiler already keeps them in constant memory/registers without the annotation.
**Source**: Level 3 sandbox verification (2026-04-05)
