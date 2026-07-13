# GEMM Pattern -- Pitfalls

## P1: Wave Quantization Destroys Utilization on Small Problems
- **Symptom**: Sharp performance drop when output tiles barely exceed a wave boundary
- **Root cause**: If num_tiles = num_SMs + 1, the GPU needs 2 full waves, halving utilization
- **Example**: On H100 (114 SMs), 115 tiles runs at ~50% utilization vs 114 tiles at ~100%
- **Fix**: Use persistent kernels with Stream-K scheduling to fractionally split tiles
- **Detection**: Plot TFLOPS vs problem size -- sawtooth pattern indicates wave quantization

## P2: Reducing Tile Size to Fix Wave Quantization Backfires
- **Symptom**: Smaller tiles reduce wave quantization but overall performance is worse
- **Root cause**: Halving bN halves arithmetic intensity (FLOPs/byte drops significantly)
- **Example**: 128x128x128 tile = 85.3 ops/byte; 128x64x128 tile = only 64 ops/byte
- **Also**: Fewer instructions per CTA reduces latency-hiding opportunity for warp scheduler
- **Also**: MMA atom constraints may prevent arbitrary tile size reduction
- **Fix**: Use Stream-K or Split-K instead of smaller tiles; only reduce tile size as last resort

## P3: SMEM Layout Incompatibility with WGMMA
- **Symptom**: Incorrect results or crash when calling wgmma with custom SMEM layouts
- **Root cause**: WGMMA requires specific SMEM layouts based on swizzle mode and core matrices
- **Fix**: Always use `GMMA::Layout_{MN|K}_SW{128|64|32}_Atom<T>` with `tile_to_shape`
- **Note**: The contiguous dimension of the layout atom must equal the swizzle byte count

## P4: Catastrophic Register Spilling in Warp-Specialized Kernels
- **Symptom**: Performance drops from ~480 TFLOPS to ~21 TFLOPS
- **Root cause**: Without `setmaxnreg`, compiler assigns equal registers to all warps; consumer warpgroups spill to local memory
- **Example**: FP32 accumulation GEMM without register reallocation: 2784 bytes stack frame, 4764 bytes spill stores
- **Fix**: Use `cutlass::arch::warpgroup_reg_dealloc<LowerCount>()` for producers and `warpgroup_reg_alloc<HigherCount>()` for consumers
- **Constraint**: Total registers across all warpgroups must not exceed 64K per SM (e.g., 24/240/240 for 3 warpgroups)

## P5: Incorrect Performance Measurement with Special Matrix Values
- **Symptom**: Reported TFLOPS much higher than realistic workloads
- **Root cause**: Matrices initialized with +/-1 allow hardware shortcuts that inflate throughput
- **Example**: +/-1 matrix initialization inflated from ~530 to ~630 TFLOPS on one kernel
- **Fix**: Always benchmark with random floating-point values, not integer-valued floats

## P6: Split-K Synchronization Overhead
- **Symptom**: Split-K doesn't improve performance despite reducing wave quantization
- **Root cause**: Turnstile reduction requires GMEM workspace writes, barrier waits, and extra memory traffic
- **Impact**: Each split adds arrive/wait/reduce overhead; too many splits negate the utilization gain
- **Fix**: Use hybrid Stream-K which minimizes the number of tiles needing reduction

## P7: L2 Cache Skew in Pure Stream-K Scheduling
- **Symptom**: Stream-K performs well at wave boundaries but worse than data-parallel mid-wave
- **Root cause**: Stream-K eliminates waves, causing different SMs to process different K-offsets simultaneously, destroying L2 cache reuse for shared operand tiles
- **Fix**: Use hybrid Stream-K: data-parallel for full waves (good cache behavior), Stream-K only for the partial tail wave
- **Heuristic**: Switch from Stream-K to data-parallel when the tail wave is at least half full

## P8: Forgetting to Handle the Epilogue in the Consumer Warpgroup
- **Symptom**: Output values are all zeros or garbage
- **Root cause**: In warp-specialized design, the accumulator lives in consumer warp registers; epilogue must run in consumer warps, not producer warps
- **Fix**: Place the epilogue code path inside the consumer warp branch

## P9: TMA Transaction Bytes Mismatch
- **Symptom**: Hang or incorrect synchronization in pipelined GEMM
- **Root cause**: `mbarrier.arrive.expect_tx` count does not match actual bytes transferred by TMA
- **Fix**: Transaction bytes must equal the total size of all TMA copies arriving at the same mbarrier phase (both A and B tiles)

## P10: Cluster Multicast Bitmask Errors
- **Symptom**: Data corruption when using TMA multicast with thread block clusters
- **Root cause**: Incorrect ctaMask in `cp.async.bulk.tensor` multicast -- wrong CTAs receive the data
- **Fix**: For operand A, mask includes all CTAs in the same row; for operand B, mask includes all CTAs in the same column. Use `create_tma_multicast_mask` utility.
- **Note**: Post-MMA synchronization bitmask should be bitwise OR of A and B multicast masks
