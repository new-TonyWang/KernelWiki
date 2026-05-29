# GEMM Pattern -- Skills (Overview)

## When to Apply
- Matrix multiplication C = alpha * op(A) * op(B) + beta * C
- Batched GEMM, grouped GEMM, strided-batched GEMM
- Any operation that decomposes into a series of MMA (matrix multiply-accumulate) calls

## Decision Framework: Custom Kernel vs cuBLAS

### Use cuBLAS when
- Standard GEMM shapes (M, N, K all >= 256 and well-aligned)
- No need for custom epilogues beyond what `cublasLtMatmul` epilogue fusion offers
- Need deterministic results (set `cublasSetAtomicsMode(CUBLAS_ATOMICS_NOT_ALLOWED)`)
- Rapid prototyping: `cublasGemmEx` covers most mixed-precision cases

### Write custom kernel when
- Non-standard epilogue (e.g., custom activation, attention softmax, loss computation)
- Fused operators (GEMM + LayerNorm, GEMM + residual + activation)
- Unusual shapes (very small M or N, very large K) where cuBLAS heuristic underperforms
- Need architectural-specific tuning (TMA, WGMMA, TMEM) beyond cuBLAS reach

## Core Optimization Axes

### 1. Tiling (see tiling-strategy/)
- Choose CTA tile (bM x bN x bK) to balance arithmetic intensity vs shared memory usage
- Arithmetic intensity = 2*bM*bN*bK / ((bM*bK + bN*bK + bM*bN) * sizeof(dtype))
- Larger tiles increase intensity but consume more SMEM and registers
- Tile sizes must be multiples of the MMA atom dimensions

### 2. Pipelining (see pipeline-overlap/)
- Overlap global-to-shared memory copy with MMA computation
- Multi-stage buffering: allocate N copies of operand tiles in SMEM
- More stages = more overlap opportunity but more SMEM consumed
- On Hopper: TMA + mbarrier pipeline; on Ampere: cp.async + commit groups

### 3. Warp Specialization (see warp-specialization/)
- Dedicate producer warps to data movement, consumer warps to computation
- Re-partition registers dynamically: producer needs ~24-40 registers, consumer needs ~160-256
- Particularly effective on Hopper+ where TMA is register-light
- Alternative: multistage design where all warps do both roles (works well on Ampere)

### 4. Grid-Level Scheduling
- Persistent kernels: launch exactly num_SMs CTAs, each processes multiple tiles
- Threadblock swizzling/rasterization: reorder tile assignment for L2 cache locality
- Stream-K: fractional tile assignment eliminates wave quantization
- Hybrid Stream-K: use Stream-K only for partial waves, data-parallel for full waves

### 5. cuBLAS Integration (see cublas-integration/)
- `cublasLtMatmul`: epilogue fusion (bias, ReLU, GELU, quantization)
- Algorithm selection via `cublasLtMatmulAlgoGetHeuristic` or exhaustive search
- Workspace allocation affects which algorithms are available

## Key Performance Metrics
- Utilization = achieved TFLOPS / peak TFLOPS
- State-of-art Hopper FP16 GEMM: ~84% utilization (630 TFLOPS on H100 PCIe)
- 65% utilization achievable with basic pipelining alone
- Gap between 65% and 84% closed by persistent kernels, threadblock rasterization, epilogue overlap

## Architecture-Specific Notes

### Ampere (SM80)
- Use `cp.async` for GMEM-to-SMEM transfer
- MMA operands must be in registers (requires SMEM-to-RMEM load pipeline)
- Multistage design preferred (no register reallocation support)
- Best atom: `mma.sync.aligned` with 16x8x16 shapes

### Hopper (SM90)
- Use TMA for GMEM-to-SMEM (register-light, handles index calculation)
- WGMMA sources operand B from SMEM directly (no SMEM-to-RMEM needed)
- `setmaxnreg` enables producer/consumer register rebalancing
- Best atom: `wgmma.mma_async.sync.aligned` with M=64, N up to 256
- Cluster support: TMA multicast across cluster for operand sharing

### Blackwell (SM100)
- TMEM (Tensor Memory): 256KB per SM for accumulator storage
- UMMA (`tcgen05.mma`): fully async, accumulates in TMEM, largest atom 128x256x16
- 2-CTA MMA: pairs of CTAs cooperate for 256x256 tiles
- TMA multicast + DSMEM for cross-CTA operand sharing within clusters

## Cross-References
- optimization/memory/shared-memory-cache -- SMEM allocation for tile buffers
- optimization/memory/data-prefetch -- TMA and cp.async pipeline mechanics
- optimization/compute/tensor-core -- MMA instruction details
- optimization/compute/warp-specialization -- producer/consumer pattern
- optimization/memory/register-pressure -- setmaxnreg and spill avoidance
- optimization/latency/occupancy-tuning -- register/SMEM vs occupancy tradeoff
