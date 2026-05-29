---
api: cutlass-free GEMM via TMA-PTX + wgmma-PTX composition
namespace: ptx
probe_slug: gemm-ptx-hello
status: partial
kind: api-end-to-end
trigger: cutlass-free GEMM hello-world (TMA + wgmma composed)
evidence_level: measured
clock_policy: as-launched
measured_on: H200-SXM | sm_90a | cuda 12.9.86 | driver 570.124.06
verdict: partial
source:
- path: blogs/colfax/cutlass-tutorial-fast-matrix-multiplication-with-wgmma-on-nvidia-hopper-gpus
  anchor: Hopper wgmma GEMM walkthrough
artifacts:
  code: sources/experience/api-probes/gemm-ptx/artifacts/gemm_ptx.cu
  build: sources/experience/api-probes/gemm-ptx/artifacts/build.sh
  run: sources/experience/api-probes/gemm-ptx/artifacts/run.sh
  introspection: sources/experience/api-probes/gemm-ptx/artifacts/device.json
  profile: sources/experience/api-probes/gemm-ptx/artifacts/profiles/2026-04-28-gemm-ptx-hello.csv
  distilled_view: wiki/nvidia/code-walkthroughs/ptx-gemm/gemm_ptx.cu
  distilled_build: wiki/nvidia/code-walkthroughs/ptx-gemm/build.sh
upstream_repo: none (hand-rolled cutlass-free implementation)
id: exp-gemm-ptx
type: experience
vendor: nvidia
title: 2026 04 28 Gemm Ptx Hello
---
# Probe — Cutlass-free GEMM hello-world

**Goal**: compose the cutlass-free TMA-PTX + wgmma-PTX primitives into a single GEMM kernel; verify zero `cutlass::` / `cute::` symbols; compare numerically against cuBLAS at the same shape.

## Method

Single-kernel GEMM at M=64 N=8 K=16 (one wgmma m64n8k16 bf16 instance + two TMA loads):

1. Host: random small bf16 inputs A (64×16) and B (16×8); build CUtensorMaps via `cuTensorMapEncodeTiled`.
2. Device: kernel issues TMA load A + B into smem via `cp.async.bulk.tensor.2d.shared::cluster.global` + mbarrier sync.
3. Device: kernel issues `wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16` from inline PTX, accumulates into per-thread fragment.
4. Device: kernel stores fragment to global D.
5. Host: cuBLAS computes the reference D via `cublasGemmEx` at the same shape with bf16 inputs / f32 accum.
6. Host: element-by-element compare ours vs cuBLAS.

## Setup

```bash
ssh h200_ncu '
  export PATH=/usr/local/cuda-12.9/bin:$PATH
  mkdir -p /root/gemm_ptx
  cd /root/gemm_ptx
  bash build.sh        # build + cutlass-free verification
  ./gemm_ptx           # numeric comparison vs cuBLAS
'
```

## Run-log excerpt (verbatim)

```
$ bash build.sh
=== nvcc -E grep for cutlass::/cute:: in preprocessed source ===
PASS: preprocessor produces 0 cutlass:: / cute:: matches.
=== cuobjdump --dump-elf-symbols grep for cutlass::/cute:: in linked binary ===
PASS: linked binary has 0 cutlass:: / cute:: symbols.

$ ./gemm_ptx
=== gemm_ptx (cutlass-free GEMM at M=64 N=8 K=16, bf16 inputs / f32 accum) ===
  ptx     hD[0..3]: 2.6250 0.0000 -0.7500 0.3750
  cuBLAS  hD[0..3]: 1.5000 0.2500 -1.6250 -0.3750
  max_abs_diff = 5.875000
  max_rel_diff = 33.000000
  mismatches (both abs > 1e-2 AND rel > 1e-2): 500 / 512
  verdict: FAIL
```

## Verdict

**partial.** The cutlass-abandonment gates **PASS** (zero `cutlass::` / `cute::` matches at both preprocessor and linked-binary levels — the structural-correctness gate). The numeric-correctness gate still **FAILS**: 497/512 mismatches after two-of-three diagnosed fixes (B host-side transpose to col-major K×N for `.SS_TN`, descriptor SBO 256→16 per cutlass `make_gmma_desc`). The residual error is in the per-thread fragment-store mapping in `gemm_ptx_kernel`, which uses an Ampere-style `(warp_id*16 + lane_id/4, lane_id%4*2)` layout incorrect for Hopper's 4-warp m64 atom. See `wiki/nvidia/foundations/compute/gemm-ptx/pitfalls.md` Pitfall #0 (B-layout, fixed) and Pitfall #1 (fragment-store, still open).

## What works

- Build pipeline: `nvcc -gencode=arch=compute_90a,code=sm_90a` produces a working binary.
- TMA load: the cutlass-free `tma_hello` (loaded separately) verifies the TMA primitive against ground-truth host source bytes — exact match at every element.
- wgmma instruction issue: the cutlass-free `wgmma_hello` (loaded separately, all-1.0 inputs) verifies the wgmma instruction produces all-K outputs as expected.
- Cutlass-free verification (preprocessor + linked binary): both gates PASS.
- cuBLAS reference comparator: works (cuBLAS computes a non-zero reference; the diff is well-defined).

## What is broken

- The per-thread fragment-store mapping in the wgmma m64n8k16 kernel is still permuted relative to the actual Hopper-spec layout. After the B-layout + SBO fixes, the residual 497/512 mismatches isolate this as the only remaining bug; extracting the fragment layout from `cute::SM90::GMMA::MMA_64x8x16_F32BF16BF16_SS::DRegisters` is the next step.

## Follow-up fix plan

1. Read PTX ISA §"Asynchronous Warpgroup-Level Matrix Instructions / Matrix Fragments / m64nNk16" to extract the exact per-thread (row, col) mapping.
2. Cross-check by extracting the same layout from cutlass's `cute::SM90::GMMA::MMA_64x8x16_F32BF16BF16_SS_TN` atom (the kernel never includes this; cross-checking happens in a separate audit binary).
3. Update `gemm_ptx.cu`'s store stage with the corrected mapping; rerun.
4. Promote `wiki/nvidia/foundations/compute/gemm-ptx/skill.md` from `partial` to `verified` once 0 mismatches against cuBLAS.

## Known caveats

- This is the smallest meaningful shape (M=64 N=8 K=16, single wgmma instance, single TMA load each). Running aligned shapes (e.g. 2048³) and non-aligned shapes is follow-up work; those scale up by adding K-iteration loops + multi-tile in M/N. None of those are landed yet.
- bf16 only; tf32 follows the same pattern with an instruction-mnemonic substitution.
- No persistent / warp-specialized schedule; this is a one-shot single-CTA kernel.
- No fused prologue or epilogue; this is the bare aligned-shape GEMM.
