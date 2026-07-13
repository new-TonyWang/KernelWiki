# Tiling Strategy -- Skills

## When to Apply
- Choosing CTA tile dimensions (bM, bN, bK) for a GEMM kernel
- Selecting MMA atom size and relating it to tile dimensions
- Balancing arithmetic intensity, SMEM usage, and occupancy

## Skill 1: Compute Arithmetic Intensity for Tile Selection
- For a bM x bN x bK tile: FLOPs = 2 * bM * bN * bK
- Memory transfers (bytes) = (bM * bK + bN * bK + bM * bN) * sizeof(element)
- Arithmetic intensity = FLOPs / bytes
- Target: intensity >> memory bandwidth / compute throughput (operational intensity threshold)
- Example: 128x128x128 with FP16 = 2*128^3 / ((128*128 + 128*128 + 128*128) * 2) = ~85 ops/byte
- Example: 128x64x128 with FP16 = 2*128*64*128 / ((128*128 + 64*128 + 128*64) * 2) = ~64 ops/byte

## Skill 2: MMA Atom Constraints on Tile Dimensions
- WMMA (Volta/Turing): atoms like 16x16x16 -- tile dims must be multiples of 16
- mma.sync (Ampere): atoms like 16x8x16 -- tile must be multiple of these
- WGMMA (Hopper): M always 64, N = 8 to 256 (multiples of 8), K = 16 for FP16
  - bM must be multiple of 64, bN multiple of chosen N, bK multiple of 16
- UMMA (Blackwell): largest single-CTA atom 128x256x16
  - 2-CTA mode: 256x256x16 effective atom

## Skill 3: SMEM Budget Determines Maximum Stages
- Per-stage SMEM = (bM * bK + bN * bK) * sizeof(element) + pipeline barrier overhead
- Total SMEM = per_stage * num_stages + accumulator-related SMEM (if any)
- H100 max SMEM per SM: 228 KB (with opt-in)
- Typical good configs for Hopper FP16:
  - bM=256, bN=256, bK=96, 2 stages (FP16 accumulation)
  - bM=256, bN=192, bK=128, 2 stages (FP32 accumulation)
  - bM=128, bN=128, bK=64, 3 stages

## Skill 4: Tile Size vs Wave Quantization Tradeoff
- Larger tiles: higher arithmetic intensity, fewer total tiles, more wave quantization risk
- Smaller tiles: lower intensity but more tiles, less wave quantization
- Rule: choose the largest tile that keeps intensity well above roofline threshold
- Use Stream-K to handle wave quantization rather than shrinking tiles

## Skill 5: SMEM Layout with Swizzling for Bank-Conflict-Free Access
- WGMMA requires swizzled SMEM layouts to avoid bank conflicts
- Use CUTLASS layout atoms: `GMMA::Layout_{MN|K}_SW{128|64|32|INTER}_Atom<T>`
- `tile_to_shape` replicates the layout atom over the full tile shape
- Swizzle mode determines contiguous dimension length:
  - SW128: 64 elements of FP16 = 128 bytes contiguous
  - SW64: 32 elements of FP16 = 64 bytes contiguous
- For non-16-bit types (e.g., FP8, INT8): layout must be K-major

## Skill 6: cuBLAS Tile Selection via Heuristic API
- `cublasLtMatmulAlgoGetHeuristic`: returns ranked algorithm list with tile/stage configs
- `cublasLtMatmulAlgoCapGetAttribute` with `CUBLASLT_ALGO_CAP_TILE_IDS`: query supported tiles
- `cublasLtMatmulAlgoConfigSetAttribute` with `CUBLASLT_ALGO_CONFIG_TILE_ID`: force specific tile
- Exhaustive search over all algo IDs sometimes finds better configs than heuristic top-1

## Skill 7: Split-K Tiling for K-Dominated Problems
- When M and N are small but K is large, standard tiling underutilizes the GPU
- Split-K divides K dimension across multiple CTAs, each producing partial results
- Requires workspace for partial results and turnstile barrier for reduction
- Tradeoff: more parallelism vs reduction overhead and lower per-CTA arithmetic intensity

## Key Principle
The fundamental tension in tiling is: larger tiles give better compute-to-memory ratio but fewer tiles means worse GPU occupancy and more wave quantization. Resolve this tension through scheduling (Stream-K, persistent kernels) rather than compromising tile size.

## Cross-References
- pattern/gemm/pipeline-overlap -- stage count directly tied to tile SMEM budget
- optimization/memory/bank-conflict-avoidance -- swizzle layout details
- optimization/memory/shared-memory-cache -- SMEM capacity constraints
- optimization/latency/occupancy-tuning -- tile size affects occupancy
