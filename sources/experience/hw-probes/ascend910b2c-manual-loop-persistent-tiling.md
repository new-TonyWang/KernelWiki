---
id: exp-ascend910b2c-manual-loop-persistent-tiling
title: "Ascend 910B2C Manual Multi-Tile Loop and Persistent Blocks"
type: experience
vendor: ascend
probe_slug: ascend910b2c-manual-loop-persistent-tiling
status: verified
kind: local-benchmark
trigger: manual loop/persistent tiling ablation for high-block-count elementwise workload
evidence_level: measured
clock_policy: unknown
measured_on:
  device: Ascend 910B2C
  host: ssh 910b
  date: '2026-07-10'
architectures:
- ascend910b
- ascend910b2
- ascend910b2c
languages:
- triton-ascend
- python
techniques:
- tiling-optimization
- vector-core-partition
- persistent-kernel
- loop-unrolling
- avoid-scalar-lowering
kernel_types:
- elementwise
tags:
- triton-ascend
- tiling-optimization
- vector-core-partition
- persistent-kernel
- loop-unrolling
- avoid-scalar-lowering
- elementwise
confidence: experimental
artifact_dir: artifacts/experience/hw-probes/ascend910b2c-manual-loop-persistent-tiling
artifacts:
  manual: artifacts/experience/hw-probes/ascend910b2c-manual-loop-persistent-tiling/docs/manual_loop_persistent_tiling.md
  manual_loop_script: artifacts/experience/hw-probes/ascend910b2c-manual-loop-persistent-tiling/profile/round11_manual_loop/scripts/manual_loop_case024.py
  manual_loop_summary: artifacts/experience/hw-probes/ascend910b2c-manual-loop-persistent-tiling/profile/round11_manual_loop/analysis/summary.csv
  manual_loop_report: artifacts/experience/hw-probes/ascend910b2c-manual-loop-persistent-tiling/profile/round11_manual_loop/REPORT.md
  case024_ablation_script: artifacts/experience/hw-probes/ascend910b2c-manual-loop-persistent-tiling/profile/round11_case024_ablation/scripts/ablate_case024.py
  case024_ablation_summary: artifacts/experience/hw-probes/ascend910b2c-manual-loop-persistent-tiling/profile/round11_case024_ablation/analysis/summary.csv
conclusions:
  workload: "Large bf16 elementwise workload, shape 15x255x1x1x256x8, 7,833,600 elements, TILE_SIZE=4096, 1913 logical tiles."
  baseline_variant: loop1
  baseline_launch_blocks: 1913
  baseline_event_us: 195.120
  best_loop_variant: persistent_48
  best_launch_blocks: 48
  best_event_us: 119.970
  best_prof_duration_us: 78.603
  scalar_us_loop1: 59.787
  scalar_us_persistent_48: 7.068
  exact_equal_to_loop1: true
  warning: "loop32 regressed to 166.690 us despite fewer launch blocks, so loop factor must be measured rather than minimized blindly."
open_questions:
- Clock policy and full production coverage were not recorded in this artifact; treat absolute timings as local benchmark evidence.
- Evidence is centered on one large elementwise workload; use it as a tuning pattern and re-measure for other shapes/dtypes.
---
# Ascend 910B2C Manual Multi-Tile Loop and Persistent Blocks

This source record localizes a manual-loop/persistent-block tiling note and related profiling code artifacts for a high-block-count elementwise workload.

The captured manual (`artifacts/experience/hw-probes/ascend910b2c-manual-loop-persistent-tiling/docs/manual_loop_persistent_tiling.md`) documents a Triton-Ascend pattern for reducing scalar/control overhead when an elementwise or streaming kernel launches far more logical programs than the physical vector-core count.

Key measured result for the large bf16 elementwise workload on Ascend 910B2C:

| Variant | Launch blocks | Event us | Prof duration us | Scalar us | Scalar ratio | Exact equal |
|---|---:|---:|---:|---:|---:|---|
| `loop1` | 1913 | 195.120 | 166.186 | 59.787 | 0.364 | yes |
| `loop8` | 240 | 122.100 | 81.404 | 9.326 | 0.119 | yes |
| `loop40` | 48 | 120.120 | 77.403 | 6.710 | 0.090 | yes |
| `persistent_48` | 48 | 119.970 | 78.603 | 7.068 | 0.094 | yes |

The artifact bundle includes the production `solution/kernel.py`, baseline code, profiling scripts, CSV summaries, and reports used by the manual note.
