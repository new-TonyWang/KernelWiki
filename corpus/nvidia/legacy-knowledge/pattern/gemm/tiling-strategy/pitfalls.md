# Tiling Strategy -- Pitfalls

## P1: Tile Dimensions Not Aligned to MMA Atom
- **Symptom**: Compilation error or runtime assertion failure
- **Root cause**: bM not multiple of M_atom, bN not multiple of N_atom, or bK not multiple of K_atom
- **WGMMA rules**: M_atom = 64, N_atom = chosen N (8..256 in steps of 8), K_atom = 32/sizeof(element)
- **Fix**: Always verify divisibility: `static_assert(bM % 64 == 0 && bN % N_atom == 0 && bK % K_atom == 0)`

## P2: Oversubscribing Shared Memory
- **Symptom**: Kernel launch failure with `cudaErrorLaunchOutOfResources`
- **Root cause**: num_stages * per_stage_bytes + pipeline_barriers > max SMEM per SM
- **Common mistake**: Forgetting pipeline barrier storage (mbarrier objects) in SMEM budget
- **Fix**: Compute total SMEM including barriers; check against `cudaDeviceGetAttribute(cudaDevAttrMaxSharedMemoryPerBlockOptin)`
- **Reminder**: Must call `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)` to opt in beyond 48KB

## P3: Wrong Swizzle Mode for Data Type
- **Symptom**: Incorrect computation results with WGMMA
- **Root cause**: Non-16-bit operand types (FP8, INT8) require K-major layout, not MN-major
- **Fix**: For FP8/INT8, use `GMMA::Layout_K_SW128_Atom<T>` instead of `GMMA::Layout_MN_SW128_Atom<T>`

## P4: cuBLAS Heuristic Returns Suboptimal Tile
- **Symptom**: cuBLAS GEMM much slower than expected for a specific problem shape
- **Root cause**: `cublasLtMatmulAlgoGetHeuristic` returns a reasonable but not optimal config
- **Fix**: Iterate over all algorithm IDs from `cublasLtMatmulAlgoGetIds` and benchmark each
- **Note**: The top heuristic result is usually within 10% of optimal, but for non-square shapes the gap can be larger

## P5: Ignoring the Epilogue SMEM Cost
- **Symptom**: Kernel crashes or needs to reduce stages when epilogue is added
- **Root cause**: Epilogue (TMA store, EVT reductions) may require additional SMEM beyond mainloop tiles
- **Fix**: Account for epilogue SMEM in the total budget before selecting num_stages

## P6: Tile Too Large for Occupancy-1
- **Symptom**: Performance lower than expected despite high arithmetic intensity
- **Root cause**: If SMEM usage allows only 1 CTA per SM, the SM is underutilized during barriers/stalls
- **Mitigation**: For Hopper warp-specialized GEMM, 1 CTA per SM is standard and acceptable
- **Mitigation**: For Ampere multistage GEMM, consider smaller tiles to allow 2 CTAs per SM

## P7: Split-K with Too Many Splits
- **Symptom**: Split-K slower than no splitting
- **Root cause**: Each split adds synchronization overhead (barriers + GMEM workspace read/write)
- **Also**: Very small per-CTA K reduces arithmetic intensity and instruction-level parallelism
- **Fix**: Start with 2-4 splits and benchmark; rarely beneficial beyond 8 splits

## P8: Short scoreboard stalls jumped from 7 (discovered in verification)

**Symptom**: Short scoreboard stalls jumped from 7.3% to 33.6%, indicating shared memory latency becomes the dominant bottleneck with larger, register-heavy tiles — occupancy-limited kernels have fewer warps to hide this latency.
**Source**: Level 3 sandbox verification (2026-04-06)

## P9: Adding more SMEM pipeline stages to a kernel that doesn't use tensor cores and is already L2-cache-bound increases instruction count and register pressure without any throughput gain — the SMEM staging skill presupposes a compute-bound or DRAM-bound tensor-core kernel (discovered in verification)

**Symptom**: Adding more SMEM pipeline stages to a kernel that doesn't use tensor cores and is already L2-cache-bound increases instruction count and register pressure without any throughput gain — the SMEM staging skill presupposes a compute-bound or DRAM-bound tensor-core kernel.
**Source**: Level 3 sandbox verification (2026-04-06)

## P10: Occupancy collapsed to 3 warps (register-limited) and short scoreboard stalls doubled to 37 (discovered in verification)

**Symptom**: Occupancy collapsed to 3 warps (register-limited) and short scoreboard stalls doubled to 37.7%, indicating the larger tile is near the register-pressure cliff; one more register per thread could spill to local memory and erase the gains.
**Source**: Level 3 sandbox verification (2026-04-06)

## P11: Applying WGMMA-style swizzled shared memory layouts to non-tensor-core kernels increases register usage and short-scoreboard stalls, causing a net regression rather than eliminating bank conflicts (discovered in verification)

**Symptom**: Applying WGMMA-style swizzled shared memory layouts to non-tensor-core kernels increases register usage and short-scoreboard stalls, causing a net regression rather than eliminating bank conflicts.
**Source**: Level 3 sandbox verification (2026-04-06)

## P12: For well-tuned library defaults (like cuBLAS on standard GEMM sizes), exhaustive algo search adds compilation/setup overhead with negligible runtime benefit; the L2 hit rate actually dropped slightly (92 (discovered in verification)

**Symptom**: For well-tuned library defaults (like cuBLAS on standard GEMM sizes), exhaustive algo search adds compilation/setup overhead with negligible runtime benefit; the L2 hit rate actually dropped slightly (92.28% → 90.85%).
**Source**: Level 3 sandbox verification (2026-04-06)

## P13: Even with Split-K the kernel is still 2 (discovered in verification)

**Symptom**: Even with Split-K the kernel is still 2.5x slower than cuBLAS (27.4us vs 69us), likely because cuBLAS uses tensor cores (sm__pipe_tensor_cycles_active only reached 0.74%) and has better L2 locality — Split-K alone is not sufficient to match vendor libraries on small M/N problems.
**Source**: Level 3 sandbox verification (2026-04-06)
