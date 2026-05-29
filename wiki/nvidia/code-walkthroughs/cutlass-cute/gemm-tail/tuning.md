---
id: code-cutlass-cute-gemm-tail-tuning
type: code-walkthrough
vendor: nvidia
title: Tuning
upstream_repo: NVIDIA/cutlass-cute
---
# gemm-tail — tuning log (skeleton)

Template-selection rules and tail-cost characterization for non-aligned GEMM. Populate from `80-experience/api-probes/gemm/2026-04-28-gemm-tail.md`.

## Decision rule (current best understanding)

```
if M < 64:           use non-wgmma path (out of cutlass aligned/tail catalogue)
elif M < 128:        small-tile <64,64,32> + Cluster <1,1,1>     (gemm_tail_small.cu)
elif M, N tile-mis:  cooperative <128,128,32> + Cluster <4,2,1>  (gemm_tail.cu)
else:                aligned                                       (gemm_aligned.cu)
```

Boundary thresholds 64 / 128 are wgmma-atom and cooperative-cluster constraints respectively; not heuristics.

## Tail-cost characterization

(Fill in from probe ablation csv as it lands.)

| Shape | Aligned baseline (same problem-size) | Tail variant | Tail overhead |
|---|---|---|---|
| 200³ | (would-be 256³ aligned) | 8.6 μs / 1.85 TFLOPS | ~6× lower than aligned path at same scale |
| 1440³ | 2048³ ≈ 188 K GFLOPS | 46.1 μs / 129 K GFLOPS | ~32 % |

## Open questions

- At what M (between 64 and 128) does the small-tile kernel cross the cooperative break-even? Current data has only 80³ and 200³.
- Does padding to the next CtaTile-multiple ever beat the predicated tail path? (Measure `mem_bw_overhead_from_padding` vs `tail_predicate_overhead`.)

## References

- Skill: `30-skill/compute/gemm/non-aligned-tail/skill.md`
- Pitfalls: `30-skill/compute/gemm/non-aligned-tail/pitfalls.md`
- Measured: `80-experience/api-probes/gemm/2026-04-28-gemm-tail.md`
