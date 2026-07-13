# Register Pressure -- Pitfalls

## P1: Excessive Register Usage Causing Low Occupancy
**Symptom**: Only 1-2 blocks per SM; kernel is latency-bound. Nsight Compute shows register count as the occupancy limiter.
**Detection**: Compile with `--ptxas-options=-v`. If registers > 64 per thread, occupancy may be severely limited.
**Fix**: Apply `__launch_bounds__` or `__maxnreg__` to cap register usage. Reduce live variables. Use shared memory for large arrays.
**Source**: Best Practices Guide, Section 10.2.7.1 (Register Pressure)

## P2: Register Spilling to Local Memory Without Awareness
**Symptom**: Kernel runs slower than expected. Profile shows local memory (lmem) traffic.
**Detection**: Compile with `--ptxas-options=-v` and check lmem bytes. Non-zero lmem means spilling. Check for `ld.local`/`st.local` in PTX.
**Fix**: Reduce register pressure by simplifying the kernel, breaking into smaller functions, or accepting higher register count with lower occupancy.
**Source**: Best Practices Guide, Section 10.2.4 (Local Memory)

## P3: __launch_bounds__ and __maxnreg__ Used Together
**Symptom**: Compilation error or unexpected behavior when both qualifiers are applied to the same kernel.
**Detection**: Compiler may issue error or warning.
**Fix**: Use only one of the two. `__launch_bounds__` is preferred for occupancy-targeted tuning; `__maxnreg__` for exact register limits.
**Source**: Programming Guide, Section 5.4.3.3 (Maximum Number of Registers per Thread)

## P4: Aggressive Register Reduction Causing Excessive Spilling
**Symptom**: Setting -maxrregcount too low causes massive local memory usage. Kernel slows down despite higher occupancy.
**Detection**: lmem usage increases dramatically with lower register limit. Net performance decreases.
**Fix**: Balance register count against spilling. Reduce gradually and profile at each step. A moderate register count (32-64) is usually optimal.
**Source**: Best Practices Guide, Section 10.2.7.1 (Register Pressure)

## P5: Dynamic Arrays Forcing Compiler to Use Local Memory
**Symptom**: Declaring `float arr[N]` where N is a variable causes compiler to allocate in local memory.
**Detection**: PTX shows `.local` allocation for the array. lmem usage is high.
**Fix**: Use constant-size arrays (known at compile time). If dynamic size is needed, use shared memory instead.
**Source**: Programming Guide, Section 2.2.3.4 (Local Memory)

## P6: Architecture-Dependent __launch_bounds__ Not Updated
**Symptom**: Kernel compiled with __launch_bounds__ tuned for older architecture; occupancy is suboptimal on newer GPU.
**Detection**: Profile on target GPU shows occupancy below expected. Register limit from launch bounds is too conservative or too aggressive.
**Fix**: Use `__CUDA_ARCH__` to set architecture-specific launch bounds, or use the occupancy API to determine optimal config at runtime.
**Source**: Programming Guide, Section 5.4.3.2 (Launch Bounds)

## P7: Applying __launch_bounds__ to a kernel that is not register-limited has zero effect; the compiler already meets the occupancy target naturally, so the annotation is a no-op (discovered in verification)

**Symptom**: Applying __launch_bounds__ to a kernel that is not register-limited has zero effect; the compiler already meets the occupancy target naturally, so the annotation is a no-op.
**Source**: Level 3 sandbox verification (2026-04-05)

## P8: Reducing register count only helps when register pressure is the actual occupancy bottleneck; on simple kernels where occupancy is already limited by other factors (block limits, shared memory), __maxnreg__ adds compiler constraints without benefit and may cause minor spilling overhead (discovered in verification)

**Symptom**: Reducing register count only helps when register pressure is the actual occupancy bottleneck; on simple kernels where occupancy is already limited by other factors (block limits, shared memory), __maxnreg__ adds compiler constraints without benefit and may cause minor spilling overhead.
**Source**: Level 3 sandbox verification (2026-04-05)

## P9: When the baseline kernel already has moderate register usage (40 regs) and high L1 cache hit rates, -maxrregcount can be counterproductive because spill/fill traffic destroys cache performance and introduces memory latency stalls that dwarf any occupancy benefit (discovered in verification)

**Symptom**: When the baseline kernel already has moderate register usage (40 regs) and high L1 cache hit rates, -maxrregcount can be counterproductive because spill/fill traffic destroys cache performance and introduces memory latency stalls that dwarf any occupancy benefit.
**Source**: Level 3 sandbox verification (2026-04-05)

## P10: Shared memory occupancy limit dropped from 32 to 13 blocks/SM due to the 17 (discovered in verification)

**Symptom**: Shared memory occupancy limit dropped from 32 to 13 blocks/SM due to the 17.9KB shared memory allocation; L1 hit rate also fell from 89.1% to 50.7% — trading local memory spills for shared memory can cap occupancy on kernels with larger working sets.
**Source**: Level 3 sandbox verification (2026-04-05)

## P11: Long scoreboard stalls remained ~45% despite doubled occupancy, indicating that for this memory-bound kernel the latency hiding gains plateau quickly — further occupancy increases alone won't help much (discovered in verification)

**Symptom**: Long scoreboard stalls remained ~45% despite doubled occupancy, indicating that for this memory-bound kernel the latency hiding gains plateau quickly — further occupancy increases alone won't help much.
**Source**: Level 3 sandbox verification (2026-04-05)
