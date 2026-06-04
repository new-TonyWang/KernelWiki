---
id: pitfall-gemm-ptx
type: pitfall
vendor: nvidia
title: Pitfalls
architectures:
- sm90
- sm90a
languages:
- ptx
- cuda-cpp
- cute-dsl
hardware_features:
- wgmma
- tma
techniques:
- shared-memory-optimization
kernel_types:
- gemm
confidence: inferred
tags:
- wgmma
- tma
- shared-memory-optimization
- gemm
- ptx
- cuda-cpp
- cute-dsl
---
# Cutlass-free Hopper GEMM — pitfalls

## 0. B's storage layout for .SS_TN

After reading cutlass's `cute/atom/mma_traits_sm90_gmma.hpp` `make_gmma_desc<MajorMode>` (lines 220–296), one root cause of mismatches at M=64 N=8 K=16 is **B's physical storage layout** (independent of the per-thread fragment-store mapping in pitfall #1):

- `wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {acc}, descA, descB, ..., transA=0, transB=0` with `.ss_TN` means A is **K-major (row-major M×K)** and B is **N-major (col-major K×N)**.
- "B in col-major K×N" means physical storage = each col of N is laid out contiguously in K, with N cols separated by K bytes. Equivalently, B is stored as if it were row-major N×K (transpose).
- Allocating B as **row-major K×N** (each row has N elements, rows separated by N bytes) is **not** what `.SS_TN` expects; it's what `.SS_NT` would expect.
- The fix: either (a) transpose B on the host before TMA so smem holds it in col-major K×N (= row-major N×K), OR (b) change the wgmma op variant (transA, transB) to match the row-major K×N layout.

A descriptor-encoding fix attempt (LBO=128, SBO=16 for B) addresses a secondary symptom but not the underlying B-layout mismatch. This hypothesis is consistent with the observed "magnitudes are similar but values are permuted" failure mode (a transpose of B permutes the col-sums computed at each output row).

Pitfall #1 below describes the per-thread fragment layout, which is independent and may also need to be re-derived from the PTX ISA spec; if the B-layout fix alone produces zero mismatches, the fragment-store mapping was probably correct all along.

## 1. wgmma m64nNk16 fragment layout is NOT the m16n8k16 layout

The Ampere `mma.sync.aligned.m16n8k16.f32.bf16.bf16` instruction has a well-known per-thread fragment layout: each thread holds 4 floats at row=`(lane_id/4)`, col=`(lane_id%4)*2..(lane_id%4)*2+1`, and a sister pair at row+8. The Hopper `wgmma.mma_async.sync.aligned.m64nNk16.f32.bf16.bf16` is **not** the same: it spans 4 warps in the M direction (M=64 = 4 warps × 16 rows) and the per-thread fragment layout is described in PTX ISA §"Hopper Tensor Core Operations / Matrix Fragments" — it is NOT `warp_id*16 + lane_id/4` for the M-row offset. The correct mapping requires careful reading of that section or extracting it from cutlass's `cute::SM90::GMMA::MMA_64x*x16_F32BF16BF16_SS_TN` atom traits. The current cutlass-free harness uses the (incorrect) Ampere-style mapping and therefore produces a permuted output, which is why outputs differ from cuBLAS at M=64 N=8 K=16.

## 2. wgmma SS-variant descriptors: ld_bytes vs sd_bytes are atom-shape-dependent

The smem matrix descriptor encodes two strides — leading-dim and stride-dim — measured between **8x8 fragment boundaries**, not between elements. For an A matrix in row-major bf16 (M×K):
- leading-dim stride = byte distance between consecutive 8-row tile groups in M = 8 × K × sizeof(elem).
- stride-dim stride = byte distance between consecutive 8-col tile groups in K = 8 × sizeof(elem).

Using `ld=256, sd=256` is correct for M=64 K=16 only by coincidence (8*16*2 = 256 AND 8*32 = 256 because of the row-vs-col-major interpretation); for other atom shapes the values must be recomputed.

## 3. cuBLAS row-major-vs-col-major requires careful op_T handling

cuBLAS internally is column-major. To compute D[M×N] = A[M×K] · B[K×N] in row-major using cuBLAS, the canonical idiom is:

```cpp
cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
             N, M, K,         // sizes swapped to compute (B^T)·(A^T) col-major
             &alpha,
             B, ..., N,       // B is K×N row-major == N×K col-major (lda=N)
             A, ..., K,       // A is M×K row-major == K×M col-major (lda=K)
             &beta, D, ..., N);
// Result D is M×N in row-major.
```

Get the lda/ldb wrong and the cuBLAS comparison fails identically to a layout-mapped GEMM bug — making it hard to isolate the wgmma layout bug from a cuBLAS-call bug. Sanity-check by setting all-1.0 inputs and verifying both ours and cuBLAS produce all-K outputs first.

## 4. CUtensorMap dimension order: fastest-moving FIRST

For both A and B, `cuTensorMapEncodeTiled`'s `global_dim` array has the fastest-moving dimension first. For row-major A[M, K] (K is fastest-moving), pass `global_dim = {K, M}`. Get this wrong and the TMA loads transposed data — the kernel still runs but the math is over the wrong matrix.

## 5. The cutlass-free gate (zero symbols) is necessary but NOT sufficient

Verification has two components: (a) zero `cutlass::` / `cute::` symbols, (b) numeric correctness against cuBLAS within tolerance. The current code passes (a) but not (b). Don't claim closure on (a) alone — the cutlass-abandonment guarantee is structural, but the "matches cuBLAS within tolerance" is the correctness guarantee. Both must hold.

## 6. Fragment-layout debugging is faster with a single-warp kernel first

Before debugging a 4-warp m64n8k16 fragment-store mapping, build a 1-warp m16n8k16 (Ampere-style) variant and verify it produces correct output. The fragment-layout pattern within a single warp is identical between Ampere and Hopper (one is just the building block of the other); confirming the within-warp mapping eliminates one variable when debugging the cross-warp arrangement.
