---
title: Instruction-Level Parallelism (ILP) - Pitfalls
status: draft
evidence_level: measured
applies_to_pattern_class:
- cuda-core
applies_to_ops:
- reduction
- scan
- elementwise
- normalization
requires_sm: '>=3.0'
single_kernel_useful: true
source:
- path: spec
  anchor: Reference
id: pitfall-ilp
type: pitfall
vendor: nvidia
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1064-L1065
- source_id: cuda-official/toolkit-docs-13.2
  path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1124
- source_id: cuda-official/toolkit-docs-13.2
  path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L24765-L24805
- source_id: cuda-official/toolkit-docs-13.2
  path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1229-L1232
---
## P1: Register pressure from excessive accumulators

**Symptom**: Kernel occupancy drops significantly after adding ILP accumulators, and overall throughput decreases despite higher per-thread throughput.

**Cause**: Each additional accumulator chain adds registers. On H200, the register file is 65536 32-bit registers per SM. If a kernel already uses, say, 128 registers/thread at 256 threads/SM (2 blocks of 128), adding 4 extra FP32 accumulators increases register usage to 132 registers/thread, potentially preventing the launch of a second block. The loss of TLP (thread-level parallelism) from reduced occupancy can outweigh the gain from ILP.

**Detection**: Compile with `nvcc -Xptxas=-v` to see register usage per kernel. Compare occupancy before and after adding accumulators using the CUDA Occupancy Calculator.

**Fix**: Limit the number of accumulators to the pipeline depth (4 for FP32 on H200). Going beyond 4 gives no ILP benefit (measured: 8-acc = 4-acc = 1.01 cycles/FMA) but increases register pressure. If occupancy is already critical, consider using `__launch_bounds__` or `-maxrregcount` to cap registers, and accept a smaller ILP factor (2 accumulators give 2x, which may be the best tradeoff).

## P2: Compiler defeats manual ILP via optimization

**Symptom**: Hand-written multi-accumulator code runs no faster than single accumulator.

**Cause**: The compiler may recognize that the accumulators are independent and re-merge them, or it may already be applying ILP-like scheduling internally. More commonly, `-O3` with `#pragma unroll` may produce code where the unrolled iterations are *not* truly independent (e.g., because of address computation dependencies or aliasing concerns).

**Detection**: Inspect the generated SASS with `cuobjdump -sass <binary>`. Look for back-to-back dependent instructions with no independent instructions interleaved between them.

**Fix**: Use `__fmaf_rn` (intrinsic FMA) instead of plain `a * b + c` if the compiler is fusing/reordering in unexpected ways. Mark accumulators as `volatile` during debugging to confirm they are not being merged (but remove `volatile` for production, as it prevents register allocation). Alternatively, use inline PTX assembly (`asm volatile`) to guarantee instruction ordering.

## P3: Applying ILP to memory-bound kernels

**Symptom**: Adding multiple accumulators to a simple elementwise or copy-like kernel shows no speedup.

**Cause**: The kernel is bottlenecked on memory bandwidth, not arithmetic throughput. The FP32 pipeline is already idle most of the time, waiting for data from L2 or HBM. Adding ILP to arithmetic instructions cannot fix a memory bottleneck.

**Detection**: Use `ncu` (Nsight Compute) to check the `sm__pipe_fma_cycles_active` metric. If FMA pipe utilization is already low (e.g., < 20%), the kernel is memory-bound and arithmetic ILP will not help.

**Fix**: For memory-bound kernels, ILP on the *memory access* side is what matters -- issuing multiple independent loads so the memory subsystem can overlap requests. This is a different technique: use vectorized loads (`float4`), or load multiple elements before processing any of them (software pipelining). See the `vectorized-access` and `coalescing` skills.

## P4: Excessive unrolling increases instruction cache pressure

**Symptom**: Large unroll factors (e.g., `#pragma unroll 32` or full unroll of a long loop) cause unexpected slowdown.

**Cause**: Unrolling a loop N times multiplies the instruction footprint by ~N. If the unrolled code exceeds the L0/L1 instruction cache capacity, the SM stalls on instruction cache misses. On H200, the instruction cache is limited (exact size not publicly documented), and very large unrolled loops can trigger this.

**Detection**: In Nsight Compute, check `smsp__inst_executed` vs `smsp__warps_launched` -- if the ratio is unexpectedly high, the loop may be over-unrolled. Also look for `stall_imc` (instruction miss count) stalls.

**Fix**: Limit unrolling to the ILP-useful depth. For FP32, `#pragma unroll 4` is sufficient to saturate the pipeline. Unrolling beyond 4 increases code size without throughput benefit.

## P5: Non-independent loop iterations mistaken for ILP

**Symptom**: `#pragma unroll` applied to a loop where iterations have dependencies does not improve throughput.

**Cause**: Unrolling only exposes ILP if iterations are truly independent. In a scan (prefix sum), for example, iteration `i` depends on iteration `i-1`, so unrolling merely inlines the loop without creating independent chains.

**Example**:
```cuda
// WRONG: these iterations are dependent -- unrolling does not add ILP
float prefix = 0.0f;
#pragma unroll 4
for (int i = 0; i < N; ++i) {
    prefix += in[i];       // depends on previous prefix
    out[i] = prefix;
}
```

**Fix**: Restructure the algorithm to have independent sub-computations. For reductions, split input into K independent partial sums (one per accumulator), then combine at the end. For scans, use the Blelloch or Hillis-Steele parallel algorithms where multiple elements can be processed independently within each step.

## P6: FP associativity changes with multiple accumulators

**Symptom**: Multi-accumulator reduction produces slightly different numerical results compared to single-accumulator version.

**Cause**: Floating-point addition is not associative. Changing the order of summation (by splitting into K independent partial sums and then combining) changes the rounding pattern. For fp32, the difference is typically within a few ULP for well-conditioned sums, but for ill-conditioned sums (e.g., alternating large positive and negative values), the difference can be significant.

**Detection**: Compare max absolute error between single-accumulator and multi-accumulator results on representative inputs.

**Fix**: This is inherent to the technique and usually acceptable. If exact bitwise reproducibility with the serial order is required, ILP via multiple accumulators cannot be used for reduction. Consider compensated summation (Kahan) if both ILP and high numerical accuracy are needed -- Kahan summation can be applied independently to each accumulator chain.

## P7: Forgetting to combine accumulators after the loop

**Symptom**: Incorrect final result (only 1/K of the true sum).

**Cause**: After a K-accumulator reduction loop, the programmer forgets to add the K partial sums together before the warp-level or block-level reduction step.

**Example**:
```cuda
float acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
for (int i = tid; i < N; i += stride * 4) {
    acc0 += in[i];
    acc1 += in[i + stride];
    acc2 += in[i + stride * 2];
    acc3 += in[i + stride * 3];
}
// BUG: only acc0 is used in the warp reduction
float val = acc0;  // Should be: acc0 + acc1 + acc2 + acc3
```

**Fix**: Always combine all accumulators before proceeding to the collective reduction step: `float val = acc0 + acc1 + acc2 + acc3;`.
