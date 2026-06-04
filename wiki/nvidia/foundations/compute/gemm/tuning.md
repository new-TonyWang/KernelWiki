---
id: skill-gemm-tuning
type: skill
vendor: nvidia
title: Tuning
tags:
- cuda-cpp
- wgmma
- warp-specialization
- gemm
applies_to:
- general
source:
- path: spec
  anchor: CUDA Programming Guide
evidence_level: spec
architectures:
- sm90
- sm90a
languages:
- cuda-cpp
hardware_features:
- wgmma
techniques:
- warp-specialization
kernel_types:
- gemm
confidence: source-reported
---
# Non-aligned GEMM Tuning Parameters

## Compile-time parameters

All aligned-GEMM tuning knobs apply (see `wiki/nvidia/foundations/compute/gemm/aligned/tuning.md`). Additional knobs specific to non-aligned shapes:

| Knob | cutlass template param | Default | Range | Effect |
|---|---|---|---|---|
| Tail strategy | `MainloopSm90TmaGmmaWarpSpecialized*` | predicate-based | predicate, split-k tail | How boundary CTAs handle partial tiles. Predicate: wgmma atoms mask out-of-bounds elements. Split-K: decompose the K dimension into aligned chunks + remainder. |
| Fallback tile shape | `CtaTile` (auto-selected by CollectiveBuilder) | varies | (64,64,32), (128,128,32) | For shapes far smaller than the primary tile, cutlass may select a smaller tile. The tail strategy applies to the executing kernel's tile, not the canonical tile. |

## Key interaction: shape vs executing kernel

For non-aligned shapes, the kernel that actually executes may differ from the canonical kernel (cutlass's `can_implement` check selects the first compatible kernel from a priority list). The tail cost is measured against the executing kernel's tile, not the canonical tile.

See BitLesson `BL-20260428-shape-non-aligned-on-executing-kernel` for the diagnostic that established this distinction.

## Measured sweep reference

Non-aligned shapes: `sources/experience/api-probes/gemm.md`
Adjacent-shape proxy measurement (1440 vs 1536): `artifacts/experience/api-probes/gemm/2026-04-28-gemm-tail.csv`
