# Tensor Core -- Pitfalls

## P1: Fragment Architecture Mismatch Across Compilation Units
**Symptom**: Incorrect MMA results or memory corruption when linking separately compiled object files targeting different compute capabilities (e.g., sm_70 and sm_75).
**Detection**: Mismatched results that only appear in multi-TU builds; difficult to detect at compile time.
**Fix**: Never pass `wmma::fragment` objects across translation unit boundaries compiled for different architectures. Store fragments to memory (`store_matrix_sync`) and reload from the other TU.
**Source**: Programming Guide, Section 5.4.11.5 (Restrictions)

## P2: Divergent Control Flow Around WMMA Calls
**Symptom**: Kernel hangs or produces garbage output.
**Detection**: Deadlock during execution; no error returned since the warp never completes.
**Fix**: WMMA operations require all 32 threads in the warp to participate. Never place `mma_sync`, `load_matrix_sync`, or `store_matrix_sync` inside a branch where some lanes are inactive. Ensure the controlling condition evaluates identically across the entire warp.
**Source**: Programming Guide, Section 5.4.11 (Warp Matrix Functions)

## P3: Misaligned Fragment Loads
**Symptom**: Undefined behavior or incorrect results from `load_matrix_sync`.
**Detection**: Ncu memory access errors; silent corruption.
**Fix**: The `mptr` pointer must be 256-bit (32-byte) aligned, and `ldm` must be a multiple of 8 for `__half` (16 bytes) or 4 for `float` (16 bytes). Pad shared memory allocations to satisfy alignment.
**Source**: Programming Guide, Section 5.4.11.1 (Description)

## P4: Register Pressure from Multiple Fragments
**Symptom**: Reduced occupancy, increased register spills to local memory, unexpected performance degradation.
**Detection**: Check register usage with `--ptxas-options=-v`. Ncu occupancy analysis shows low achieved occupancy.
**Fix**: Reduce the number of live fragments by reusing accumulators. Use `__launch_bounds__` to guide the compiler. Consider smaller tile sizes (e.g., 16x16x16 instead of 32x8x16) if register pressure is the bottleneck.
**Source**: Best Practices Guide, Section 10.2.7.1 (Register Pressure); Programming Guide, Section 5.4.3.2 (Launch Bounds)

## P5: Ignoring TF32 Rounding for FP32 Inputs
**Symptom**: Larger-than-expected numerical error when using TF32 Tensor Cores, because inputs were not explicitly rounded to tf32 before the MMA.
**Detection**: Numerical comparison against fp32 reference shows ~10-bit mantissa precision loss instead of expected behavior.
**Fix**: Explicitly call `__float_to_tf32()` on each element of the A and B fragments before `mma_sync`. Without this, the hardware truncates to tf32 in an implementation-defined manner.
**Source**: Programming Guide, Section 5.4.11.2 (Alternate Floating Point)

## P6: Using WMMA on Unsupported Tile Size / Type Combination
**Symptom**: Compilation error or runtime failure.
**Detection**: Compiler error mentioning unsupported WMMA configuration.
**Fix**: Consult the supported combinations table in the Programming Guide. For example, bf16 requires sm_80+, tf32 only supports 16x16x8, and double only supports 8x8x4. Sub-byte types (u4, s4, b1) are deprecated on sm_90.
**Source**: Programming Guide, Section 5.4.11.6 (Element Types and Matrix Sizes)

## P7: Roofline analysis reports uses_tensor_cores=false in both cases even though ncu clearly shows tensor core activity jumped from 0 (discovered in verification)

**Symptom**: Roofline analysis reports uses_tensor_cores=false in both cases even though ncu clearly shows tensor core activity jumped from 0.03% to 2.72% — the detection threshold is too conservative. Also, long scoreboard stalls rose from 47% to 81%, indicating the kernel is now heavily memory-latency bound since compute finishes so fast.
**Source**: Level 3 sandbox verification (2026-04-06)
