# Elementwise Pattern -- Pitfalls

## P1: Launching a Separate Kernel for Fuse-able Elementwise
- **Symptom**: Elementwise op after GEMM takes disproportionate wall-clock time
- **Root cause**: Standalone elementwise kernel is memory-bandwidth bound; the data round-trip to HBM is unnecessary if the data was just produced by GEMM
- **Impact**: For a bias+ReLU after GEMM, the unfused version adds an entire HBM read+write of the full output matrix
- **Fix**: Fuse into GEMM epilogue via cuBLAS epilogue fusion or CUTLASS EVT

## P2: Scalar Loads Instead of Vectorized
- **Symptom**: Elementwise kernel achieves <50% of peak memory bandwidth
- **Root cause**: Loading one float (4 bytes) per thread instead of float4 (16 bytes)
- **Fix**: Use vector types (float4, int4, __half2) and process multiple elements per thread
- **Requirement**: Input/output pointers must be aligned to vector width (16 bytes for float4)

## P3: Unaligned Vector Access
- **Symptom**: Memory access violation or silently incorrect results
- **Root cause**: `reinterpret_cast<float4*>(ptr)` on a pointer not aligned to 16 bytes
- **Fix**: Ensure allocations are 128-byte aligned (cudaMalloc provides this by default)
- **Fix**: For sub-array offsets, check alignment; fall back to scalar loads for unaligned tails

## P4: Using Standard Math Functions When Intrinsics Suffice
- **Symptom**: Activation function (exp, tanh, sqrt) is a performance bottleneck
- **Root cause**: Standard `expf()` is accurate to 1 ULP but much slower than `__expf()` (~2 ULP error)
- **Impact**: For most ML workloads, the extra precision is irrelevant
- **Fix**: Use `__expf`, `__tanhf`, `__sinf`, `__cosf`, `__frsqrt_rn` intrinsics
- **Blanket fix**: Compile with `--use_fast_math` (applies to entire compilation unit)

## P5: Packed Half Operations Not Used for FP16
- **Symptom**: Half-precision elementwise kernel achieves only 1x throughput improvement over FP32
- **Root cause**: Operating on individual `__half` values instead of `__half2` packed pairs
- **Fix**: Use `__hadd2`, `__hmul2`, `__hfma2`, `__hfma2_relu` to process 2 half values simultaneously
- **Benefit**: 2x throughput on the same hardware

## P6: Grid Size Larger Than Necessary
- **Symptom**: Many blocks launched with few threads doing useful work
- **Root cause**: Grid size calculated as `ceil(N / blockDim)` even for small N
- **Impact**: Each block has launch overhead; minimal-work blocks waste GPU resources
- **Fix**: Use grid-stride loop with a capped grid size (e.g., min(ceil(N/blockDim), num_SMs * blocks_per_SM))

## P7: Division and Modulo Operations in Hot Loop
- **Symptom**: Unexpected compute bottleneck in an "elementwise" kernel
- **Root cause**: Integer division and modulo are expensive (20-30 cycles each)
- **Common case**: Multi-dimensional index calculation inside the inner loop
- **Fix**: Replace division with multiplication by reciprocal; replace modulo with bitwise AND (for powers of 2)
- **Fix**: Precompute dimensional offsets outside the loop; use strength reduction

## P8: EVT Argument Tree Order Mismatch
- **Symptom**: Compilation error or incorrect epilogue values when using custom EVTs
- **Root cause**: In CUTLASS EVT, the template parameter order is (NodeOp, Child1, Child2, ...) but the argument struct order is (Child1_args, Child2_args, ..., NodeOp_args) -- reversed for the node operation
- **Fix**: Carefully match argument tree to the EVT definition; draw the tree to verify correspondence

## P9: Type Mismatch in Mixed-Precision Elementwise
- **Symptom**: Silent precision loss or wrong results
- **Root cause**: Accumulating FP16 values in FP16 instead of promoting to FP32
- **Example**: Summing thousands of FP16 values overflows or loses precision
- **Fix**: Cast to FP32 for intermediate computation; cast back to FP16 only for final storage

## P10: Ignoring Memory Bandwidth Roofline
- **Symptom**: Tuning compute aspects (occupancy, ILP) with no performance improvement
- **Root cause**: Elementwise ops have O(1) compute per byte; performance is bounded by memory bandwidth regardless of compute optimizations
- **Fix**: Focus optimization on memory throughput: vectorized loads, coalescing, minimizing total memory traffic (fusion)

## P11: Long scoreboard stalls increased from 75 (discovered in verification)

**Symptom**: Long scoreboard stalls increased from 75.9% to 91.2% — vectorized access makes the kernel more memory-latency sensitive, so occupancy or latency-hiding becomes critical for further gains.
**Source**: Level 3 sandbox verification (2026-04-05)

## P12: The fused kernel executes 12% more instructions (622K vs 557K) — fusion can increase per-kernel instruction count, so watch register pressure on more complex epilogue chains (discovered in verification)

**Symptom**: The fused kernel executes 12% more instructions (622K vs 557K) — fusion can increase per-kernel instruction count, so watch register pressure on more complex epilogue chains.
**Source**: Level 3 sandbox verification (2026-04-05)

## P13: CUTLASS EVT carries significant dispatch and abstraction overhead that can make it slower than simple PyTorch fused ops for small/moderate problem sizes; the fusion benefit only pays off when GEMM is large enough to amortize the overhead (discovered in verification)

**Symptom**: CUTLASS EVT carries significant dispatch and abstraction overhead that can make it slower than simple PyTorch fused ops for small/moderate problem sizes; the fusion benefit only pays off when GEMM is large enough to amortize the overhead.
**Source**: Level 3 sandbox verification (2026-04-05)

## P14: Grid-stride loops require the fixed grid size to still be large enough to saturate the GPU; a naive small grid (e (discovered in verification)

**Symptom**: Grid-stride loops require the fixed grid size to still be large enough to saturate the GPU; a naive small grid (e.g. num_SMs * few_blocks) on a memory-bound elementwise kernel destroys occupancy and throughput.
**Source**: Level 3 sandbox verification (2026-04-05)

## P15: L1 cache hit rate dropped from 64 (discovered in verification)

**Symptom**: L1 cache hit rate dropped from 64.6% to 0.0% with coalesced access — uncoalesced patterns artificially inflate L1 hits via replayed partial loads, so high L1 hit rate can be a symptom of poor coalescing rather than good caching.
**Source**: Level 3 sandbox verification (2026-04-05)

## P16: cuBLAS shifts the bottleneck from memory to compute for elementwise operations: registers per thread doubled (16→32), barrier stalls appeared (0%→9 (discovered in verification)

**Symptom**: cuBLAS shifts the bottleneck from memory to compute for elementwise operations: registers per thread doubled (16→32), barrier stalls appeared (0%→9.67%), and 166M instructions were executed vs 11.5M in the custom kernel — the library's generality is pure overhead at this scale.
**Source**: Level 3 sandbox verification (2026-04-05)
