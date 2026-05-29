---
id: code-cutlass-cute-gemm-aligned-tuning
type: code-walkthrough
vendor: nvidia
title: Tuning
upstream_repo: NVIDIA/cutlass-cute
---
# gemm-aligned — tuning log (skeleton)

Template-parameter search and configuration-strategy notes for the cutlass aligned GEMM. Populate from the canonical artifacts under `sources/experience/api-probes/gemm/`; do not duplicate raw csv numbers here — link to them.

## Template axes

| Axis | Values to sweep | Notes |
|---|---|---|
| `TileShape` | `<128,128,32>` / `<128,256,32>` / `<128,128,64>` | tf32 K=8 must be padded to 32 in tile shape |
| `ClusterShape` | `<1,1,1>` / `<2,1,1>` / `<4,2,1>` | pairs with KernelSchedule |
| `KernelSchedule` | `KernelTmaWarpSpecialized` / `*Pingpong` / `*Cooperative` / `KernelScheduleAuto` | KernelScheduleAuto is shape-dependent — pin explicitly for reproducibility |
| `EpilogueSchedule` | `NoSmemWarpSpecialized` / `TmaWarpSpecialized` / `*Cooperative` | must match KernelSchedule's persistence model |
| `Stages` | `StageCountAuto` / `StageCountAutoCarveout<N>` | N = `sizeof(typename CollectiveEpilogue::SharedStorage)` |

## Configuration strategy

(Fill in as data lands.)

- Small problems (M*N ≤ a few SM-fulls of CTAs): plain WS `<1,1,1>` typically wins. Reference: `sources/experience/api-probes/gemm/2026-04-28-warp-specialization-ablation.csv`.
- Medium / large with uniform K: pingpong `<2,1,1>`.
- Large + cluster-multicast feasible: cooperative `<4,2,1>`.

## can_implement failures

(Catalogue of `(ClusterShape, KernelSchedule, EpilogueSchedule)` tuples that get rejected. Fill in as encountered — most are documented in `wiki/nvidia/foundations/compute/gemm/aligned/pitfalls.md` already.)

## Open questions

- Stages autocarveout actual values across (TileShape, dtype) — needs `cuobjdump --dump-sass` inspection per build.
- Whether `KernelScheduleAuto` consistently picks the same schedule as the manually-pinned `*Cooperative` at 2048³ / 8192³.

## References

- Skill: `wiki/nvidia/foundations/compute/gemm/aligned/skill.md`
- Pitfalls: `wiki/nvidia/foundations/compute/gemm/aligned/pitfalls.md`
- Measured: `sources/experience/api-probes/gemm/2026-04-28-gemm-aligned.md`
