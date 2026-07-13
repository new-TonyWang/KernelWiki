---
id: code-cutlass-cute-gemm-tail-README
type: code-walkthrough
vendor: nvidia
title: Readme
upstream_repo: NVIDIA/cutlass-cute
architectures:
- sm90
- sm90a
languages:
- cuda-cpp
- cute-dsl
hardware_features:
- wgmma
- cluster
techniques:
- warp-specialization
kernel_types:
- gemm
confidence: inferred
tags:
- wgmma
- cluster
- warp-specialization
- gemm
- cuda-cpp
- cute-dsl
---
# wiki/nvidia/code-walkthroughs/cutlass-cute/gemm-tail — non-aligned tail-handling library usage

Library-usage knowledge for cutlass GEMM at non-CtaTile-multiple problem shapes. The kernel template is the same warp-specialized cooperative kernel as `gemm-aligned`; what differs is the predicate / mask path that activates on the boundary CTA tile when M / N / K are not CtaTile multiples. Canonical reproducible artifacts:

- Cooperative `<128, 128, 32>` + Cluster `<4, 2, 1>` for tile-misaligned and kilo-scale cases: `artifacts/experience/api-probes/gemm/gemm_tail.cu`
- Smaller-tile non-cooperative `<64, 64, 32>` + Cluster `<1, 1, 1>` for "far smaller than tile" (M < 128): `artifacts/experience/api-probes/gemm/gemm_tail_small.cu`

Use those paths to build and run.

## When to use

Whenever the caller cannot pre-pad to the CtaTile boundary upstream (memory budget, layout shared with another op, etc.). For `M < wgmma_atom_M = 64`, no wgmma kernel template applies — fall back to a non-wgmma path.

## Library knob: kernel template selection by shape

The cooperative kernel's `can_implement` rejects shapes below ~128 in M because the cooperative cluster + tile combination has no room for the smaller boundary tile. Pick the template by shape category:

| Category | Concrete shape | Kernel template | Cluster | Throughput regime |
|---|---|---|---|---|
| Far smaller than CtaTile | 80³ | small-tile `<64,64,32>` non-cooperative | `<1,1,1>` | launch-bound (single-digit μs at 4-CTA grids) |
| Between 1 and 2 cooperative CtaTiles | 200³ | cooperative `<128,128,32>` | `<4,2,1>` | tail-handling fully active; ~6× lower throughput than the aligned path at the same problem scale |
| Far larger with non-aligned tail | 1440³ | cooperative `<128,128,32>` | `<4,2,1>` | most of grid is aligned fast path; tail is ~32 % overhead |
| Sub-wgmma-M-floor (M < 64) | 30³ | refused — both templates return `kErrorInvalidProblem` | — | use non-wgmma fallback (mma.sync / SIMT FMA) |

Reference numbers in `sources/experience/api-probes/gemm.md`.

## Cross-references

- Skill: `wiki/nvidia/foundations/compute/gemm/non-aligned-tail/skill.md`
- Pitfalls: `wiki/nvidia/foundations/compute/gemm/non-aligned-tail/pitfalls.md`
- Canonical artifacts: `artifacts/experience/api-probes/gemm/gemm_tail.cu` + `gemm_tail_small.cu`
- Measurement record: `sources/experience/api-probes/gemm.md`
