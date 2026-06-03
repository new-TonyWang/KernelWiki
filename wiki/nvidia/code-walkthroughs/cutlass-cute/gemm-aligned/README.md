---
id: code-cutlass-cute-gemm-aligned-README
type: code-walkthrough
vendor: nvidia
title: Readme
upstream_repo: NVIDIA/cutlass-cute
---
# wiki/nvidia/code-walkthroughs/cutlass-cute/gemm-aligned — aligned GEMM library usage

Library-usage knowledge for the cutlass / cute aligned-GEMM API path on Hopper sm_90a. This directory does **not** carry buildable code — the canonical reproducible artifact is

- `artifacts/experience/api-probes/gemm/gemm_aligned.cu` (a vendored copy of cutlass example 48 at commit `f74fea9c`) plus its `build.sh` / `run.sh` / `device.json` / `profiles/*.csv` siblings.

Use that path to actually build and run. This directory documents *how* to use the library and *which knobs to turn*.

## When to use

Aligned GEMM is the cutlass-API canonical kernel for Hopper shapes where (M, N, K) are exact integer multiples of the chosen wgmma atom (`MMA_64x128x8_F32TF32TF32_SS_TN` for tf32; `MMA_64xNx16_F32BF16BF16_SS_TN` family for bf16/fp16). No tail / no predicate / no padding — the fast-path warp-specialized cooperative mainloop runs every iteration.

## Library API at a glance

The collective-builder entry point composes four template axes:

| Axis | Knob | Typical Hopper choice |
|---|---|---|
| `MMA atom` | `cute::SM90::GMMA::MMA_64xNxK_*_SS_TN` | tf32 → `64x128x8`; bf16/fp16 → `64x128x16`; fp8 → `64x128x32` |
| `TileShape` | `cute::Shape<_M, _N, _K>` | `<_128, _128, _32>` (tf32) / `<_128, _256, _64>` (bf16 large) |
| `ClusterShape` | `cute::Shape<_X, _Y, _1>` | `<_1,_1,_1>` plain WS / `<_2,_1,_1>` pingpong / `<_4,_2,_1>` cooperative |
| `KernelSchedule` | `cutlass::gemm::KernelTmaWarpSpecialized*` | `*Cooperative` matches Hopper aligned baseline |
| `EpilogueSchedule` | `cutlass::epilogue::*WarpSpecialized*` | `TmaWarpSpecializedCooperative` for cooperative; `NoSmemWarpSpecialized` for plain WS |
| `StageCount` | `StageCountAutoCarveout<sizeof(epilogue smem)>` | auto-carveout from epilogue smem footprint |

The `(ClusterShape, KernelSchedule, EpilogueSchedule)` tuple is **co-constrained**; mismatched tuples either fail `gemm.can_implement(args)` or run at degraded throughput. See `pitfalls.md` of the parent skill for the catalogue.

## Cross-references

- Skill prose + measured numbers: `wiki/nvidia/foundations/compute/gemm.md`
- Failure modes and template-tuple constraints: `wiki/nvidia/foundations/compute/gemm/aligned/pitfalls.md`
- Canonical reproducible artifact: `artifacts/experience/api-probes/gemm/gemm_aligned.cu` + `build.sh`
- Measurement record (512³ / 2048³ / 8192³): `sources/experience/api-probes/gemm.md`
