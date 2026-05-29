---
id: code-cutlass-cute-persistent-kernel-tuning
type: code-walkthrough
vendor: nvidia
title: Tuning
upstream_repo: NVIDIA/cutlass-cute
---
# persistent-kernel — tuning log (skeleton)

Pingpong vs cooperative selection + persistent-scheduler knobs. Numbers are link-only — pull from `80-experience/api-probes/gemm/2026-04-28-persistent-kernel-ablation.md`.

## Pingpong vs cooperative

| Axis | Pingpong | Cooperative |
|---|---|---|
| Cluster | `<2,1,1>` | `<4,2,1>` (or larger) |
| Consumers | 2 alternating warpgroups (different tiles) | 2 cooperating warpgroups (same tile, different rows) |
| TMA | per-CTA load | cluster-multicast required |
| K-uniformity | required (otherwise one consumer stalls) | not required |
| Best at | uniform-K problems where producer overlap is the bottleneck | very large M*N where cluster-multicast amortizes per-tile setup |

## Tile-scheduler choice

| Scheduler | When |
|---|---|
| `PersistentScheduler` (default) | M*N tile grid is at least one wave + the per-tile work is uniform |
| `StreamKScheduler` | K is large enough that splitting it across CTAs improves SM utilization (split-K combine done in scheduler tail) |

## Stages count interaction

`StageCountAutoCarveout<sizeof(typename CollectiveEpilogue::SharedStorage)>` accounts for the persistent epilogue's smem footprint. Recompute when changing:
- `EpilogueSchedule` (NoSmem vs Tma vs TmaCooperative)
- accumulator dtype
- output dtype

## Open questions

- StreamK on/off vs Persistent at fixed (M,N,K) — current ablation only covers Persistent.
- Pingpong's actual K-uniformity threshold (how non-uniform can per-tile K be before pingpong loses to plain WS?).

## References

- Skill: `50-classical-algo/persistent-kernel/skill.md`
- Pitfalls: `50-classical-algo/persistent-kernel/pitfalls.md`
- Ablation: `80-experience/api-probes/gemm/2026-04-28-persistent-kernel-ablation.md`
- Example 48 cooperative mainloop notes: `60-code/cutlass-cute/example48-hopper-warp-specialized-gemm/`
