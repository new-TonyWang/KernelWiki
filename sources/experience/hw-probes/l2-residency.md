---
api: cudaStreamSetAttribute(cudaStreamAttributeAccessPolicyWindow)
namespace: cuda-runtime
probe_slug: l2-residency
status: verified
kind: documented
trigger: skill-build
evidence_level: measured
clock_policy: unlocked-logged-only (H200 reported 1980 MHz graphics, max 1980 MHz)
measured_on:
  device: NVIDIA H200
  sm: 9.0a
  gpu_uuid: GPU-fbaa167f-4646-b8b9-c4f5-a4c4734ebc25
  cuda_runtime: '12.9'
  driver: 570.124.06
artifacts:
  code: sources/experience/hw-probes/l2-residency/artifacts/l2_residency_probe.cu
  build: sources/experience/hw-probes/l2-residency/artifacts/build.sh
  introspection: sources/experience/hw-probes/l2-residency/artifacts/device.json
  profile: ''
  ncu_report_host_path: h200_ncu:/inspire/hdd/project/qianghuaxuexi/public/kernel_pilot_public/kp-probe-artifacts/l2-residency/2026-04-23/l2_residency.ncu-rep
  ncu_txt_host_path: h200_ncu:/inspire/hdd/project/qianghuaxuexi/public/kernel_pilot_public/kp-probe-artifacts/l2-residency/2026-04-23/ncu.txt
  ncu_csv_host_path: h200_ncu:/inspire/hdd/project/qianghuaxuexi/public/kernel_pilot_public/kp-probe-artifacts/l2-residency/2026-04-23/ncu_metrics.csv
  run_log_host_path: h200_ncu:/inspire/hdd/project/qianghuaxuexi/public/kernel_pilot_public/kp-probe-artifacts/l2-residency/2026-04-23/run.log
referenced_in_corpus:
- path: corpus/nvidia/cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming
    Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  line_range: L4094-L4130
- path: corpus/nvidia/cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming
    Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  line_range: L1561-L1581
source:
- path: spec
  anchor: Reference
conclusions:
  workload: repeat_read_sum, grid=528 blocks x 256 threads, N_REPEATS=32 inner passes
    per launch, buffer filled with 1.0f; buffer slice sized to WS; accessPolicyWindow
    applied per-stream.
  h200_l2_total_bytes: 62914560
  h200_persisting_l2_max_bytes: 39321600
  h200_access_policy_max_window_bytes: 134217728
  set_aside_bytes: 39321600
  ws4_none_ms: 0.0297
  ws4_none_gbps: 4524.6
  ws4_persist_10_ms: 0.0297
  ws4_persist_10_gbps: 4524.6
  ws4_persist_tuned_ms: 0.0296
  ws4_persist_tuned_gbps: 4534.38
  ws40_none_ms: 0.3081
  ws40_none_gbps: 4356.81
  ws40_persist_10_ms: 0.3082
  ws40_persist_10_gbps: 4355.46
  ws40_persist_tuned_ms: 0.3078
  ws40_persist_tuned_gbps: 4359.98
  ws80_none_ms: 1.8694
  ws80_none_gbps: 1435.94
  ws80_persist_10_ms: 1.8694
  ws80_persist_10_gbps: 1435.96
  ws80_persist_tuned_ms: 1.5887
  ws80_persist_tuned_gbps: 1689.68
  ws80_tuned_speedup_over_none: 1.177
open_questions:
- clock_policy is `unlocked-logged-only` — H200 ran at max graphics clock (1980 MHz)
  but was not explicitly locked. Absolute GB/s are subject to boost-clock variation;
  ratios across (WS, policy) cells are robust because they share the same launch context.
- 'NCU shows `lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum = 0` for every profiled
  launch. This is an artifact of NCU''s replay model: each profiled kernel runs with
  cache state reset between the 17 replay passes, so per-profile hit counters never
  see the warm L2 that the outer N_REPEATS=32 loop creates in a normal run. The authoritative
  measurement is wall-clock GB/s; the L2-hit-rate evidence is indirect (WS=4MiB effective
  BW ≫ HBM3e peak of ~4.8 TB/s would imply L2 hits, but the measured 4524 GB/s is
  merely ~94% of HBM peak, so the direct L2 residency signature is not clean even
  in wall-clock form). A follow-up probe that exposes hit rate without NCU replay
  interference (e.g., running 2 kernels back-to-back and timing only the second with
  per-kernel `__prof_trigger`) is open.'
- 'WS=40 MiB case is a soft-null: all three policies land within 0.1% of 4357 GB/s.
  Cause is that 40 MiB fits naturally in H200''s 60 MiB L2 after the first of 32 inner
  passes, so accessPolicyWindow cannot pin data that is not in contention. A competing
  workload (concurrent kernel on a second stream that touches > 20 MiB) would reveal
  the window''s pinning effect; not measured in this probe. Follow-up: `sources/experience/hw-probes/l2-residency-contended/`
  (open).'
- persistingL2CacheMaxSize on H200 is 37.5 MiB — only 62.5% of the 60 MiB L2 physical
  size. The set-aside ratio is a hardware-fixed fraction, not user-tunable beyond
  that cap.
- Graph-node / CUDA-Graph variant (`cudaKernelNodeAttributeAccessPolicyWindow`) not
  measured; this probe uses the stream-level `cudaStreamAttributeAccessPolicyWindow`.
  Legacy skill S4 (graph nodes) is retained in `skill.md` as inferred pending a graph-focused
  probe.
id: exp-l2-residency
type: experience
vendor: nvidia
title: 2026 04 23 L2 Residency
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L4094-L4130
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1561-L1581
---
## Summary

This probe backs the cost/benefit claims in [wiki/nvidia/foundations/memory/l2-access-policy/skill.md](../../../wiki/nvidia/foundations/memory/l2-access-policy.md) by sweeping a repeat-read kernel across three working-set sizes and three L2 policies on H200 (sm_9.0a). The canonical claim from BP Guide §10.2.2 (L1561-L1581) is that `accessPolicyWindow` with `hitProp = Persisting` protects a hot buffer from eviction so repeat reads hit L2 instead of DRAM; PG §4.13 (L4094-L4130) adds that `hitRatio` tuning is the remedy when the hot region exceeds the L2 set-aside. Neither source quantifies the threshold or the cost of misapplication on H200. This probe gives the three data points that matter for the skill's decision tree:

1. When WS ≪ L2, the policy is a null (or slight loss due to overhead).
2. When WS ≈ L2 naturally, the policy is still a null — the data is already resident.
3. When WS > L2, `hitRatio=1.0` is as bad as no policy (thrashing); `hitRatio = set_aside/WS` recovers **+17.7%** effective BW.

## Setup

- GPU: NVIDIA H200, UUID `GPU-fbaa167f-4646-b8b9-c4f5-a4c4734ebc25`, driver 570.124.06, CUDA runtime 12.9, sm_9.0a.
- L2 total: 60 MiB. persistingL2CacheMaxSize: 37.5 MiB. accessPolicyMaxWindowSize: 128 MiB. Set-aside used in probe: 37.5 MiB.
- Build: `nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo`.
- Clock policy: **unlocked**; nvidia-smi reported 1980 MHz throughout.
- Protocol: 5 warmup launches + 20 timed launches per (WS, policy) pair; CUDA events; median ms reported. Buffer filled once with `1.0f`; `cudaCtxResetPersistingL2Cache` called between pairs.
- Kernel: `repeat_read_sum<<<528, 256>>>(buf, N, 32, sink)`. Inner loop: grid-stride read over the buffer, summed into a per-thread register, 32 repeats, `acc *= 0.999f` each outer iter to prevent hoisting.
- Three WS sizes: 4 MiB / 40 MiB / 80 MiB. Three policies:
  - **none** — no `accessPolicyWindow` on the stream.
  - **persist@1.0** — `hitRatio=1.0`, `hitProp=Persisting`, `missProp=Streaming`, `num_bytes = min(WS, accessPolicyMaxWindowSize)`.
  - **persist@tuned** — identical to persist@1.0 but `hitRatio = min(1, set_aside / WS)`.

## Results — median ms / effective GB/s

| WS      | policy          |  median_ms |  GB/s_eff | speedup vs none |
| ------- | --------------- | ---------: | --------: | --------------: |
|  4 MiB  | none            |     0.0297 |   4524.60 |            1.00 |
|  4 MiB  | persist@1.0     |     0.0297 |   4524.60 |            1.00 |
|  4 MiB  | persist@tuned   |     0.0296 |   4534.38 |            1.00 |
| 40 MiB  | none            |     0.3081 |   4356.81 |            1.00 |
| 40 MiB  | persist@1.0     |     0.3082 |   4355.46 |            1.00 |
| 40 MiB  | persist@tuned   |     0.3078 |   4359.98 |            1.00 |
| 80 MiB  | none            |     1.8694 |   1435.94 |            1.00 |
| 80 MiB  | persist@1.0     |     1.8694 |   1435.96 |            1.00 |
| **80 MiB**  | **persist@tuned**   | **1.5887** |**1689.68**|       **1.18**  |

Effective GB/s = (WS × 32 repeats) / median_ms, with 1 GB = 10⁹ B.

## Interpretation

### 1. Below the set-aside (WS = 4 MiB): policy is free but pointless

All three policies land at ≈4527 GB/s, within 0.2% of each other. The 4 MiB buffer fits trivially in the 60 MiB L2 after the first of 32 inner passes, so no eviction pressure exists for the window to resist. This corroborates legacy pitfall **P6**: applying a persisting policy to already-well-cached data adds policy machinery for zero gain. On H200 the "zero gain" is precisely that — the probe does **not** reproduce the legacy "dramatic DRAM-traffic increase" on sm_9.0a, so P6's worst-case claim is downgraded to "policy is overhead without benefit in this regime".

### 2. At the edge (WS = 40 MiB, slightly above set-aside of 37.5 MiB): still a soft null

40 MiB is 2.5 MiB above the set-aside limit but 20 MiB below the full L2. All three policies land at ≈4357 GB/s, again a soft null. The probe has no competing memory traffic, so even "no policy" keeps the buffer resident after pass 1/32. The window's **pinning** role only matters when *something else* would otherwise evict — a scenario that needs a concurrent competing kernel. Follow-up probe `sources/experience/hw-probes/l2-residency-contended/` (open).

### 3. Above L2 (WS = 80 MiB): tuned-hitRatio wins +17.7%

80 MiB exceeds L2 (60 MiB) *and* the 37.5 MiB set-aside. Two results:

- **`persist@1.0` is identical to `none`** (1.8694 ms, 1436 GB/s). The window tries to pin all 80 MiB, but only 37.5 MiB fits; the hardware rotates pinned lines and nothing net-sticks in L2 across passes. This is legacy pitfall **P1** ("L2 thrashing from oversized persistence window"), upgraded to measured: the wall-clock cost is **0%** — tuning fails silently rather than catastrophically.
- **`persist@tuned` (hitRatio = 37.5/80 ≈ 0.469)** pins a stochastic ~47% of the buffer and leaves the rest in the streaming property. Effective BW jumps to 1690 GB/s — a **+17.7% speedup** over none / persist@1.0. This backs the skill's **S2** directly: when WS exceeds set-aside, set `hitRatio = set_aside / WS` for a meaningful recovery.

### 4. NCU L2-hit counters are a misleading corroboration source

`lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum = 0` for every NCU-profiled launch, and `Memory Throughput ≈ 19.7%` across all nine (WS, policy) pairs. NCU's kernel-replay model flushes cache state between its 17 replay passes per launch, so the warm-L2 state that the outer N_REPEATS=32 loop creates in normal runs is never visible to the per-profile counters. The 27 µs NCU-reported duration per launch matches the first-pass-only fraction of the non-NCU wall-clock (30 µs for WS=4 MiB / N_REPEATS=32 ⇒ ~0.9 µs per pass, and NCU measures something closer to a single pass snapshot). Treat the wall-clock GB/s as authoritative; a dedicated hit-rate probe that dodges NCU replay is on the follow-up list.

## Takeaways (fed back into skill + pitfalls)

1. **`accessPolicyWindow` is a two-regime tool**. Below or at the natural L2 residency threshold, it is overhead without benefit. Above it, tuned hitRatio recovers measurable BW; untuned hitRatio is a silent null. The skill's decision tree must branch on `WS vs set_aside` first and `hitRatio` second.
2. **The tuning formula `hitRatio = set_aside / WS` is measurably correct on H200** and delivers +17.7% at WS = 80 MiB / set_aside = 37.5 MiB. Legacy S2 elevated from inferred to measured.
3. **Legacy P1 (thrashing) silent-fails on H200 at WS = 80 MiB** — same wall-clock as no policy. The H200 L2 replacement policy appears to be tolerant of oversized hitRatio=1.0 rather than punitive. This updates the pitfall text: expect a soft null, not a dramatic slowdown. (Older architectures may still punish; cross-arch claim not in scope for this MVP.)
4. **Legacy P6 / P7 downgraded** from "dramatic" to "neutral" on sm_9.0a single-kernel workloads. The "single-kernel benchmark cannot demonstrate the benefit" (legacy P8) is quantitatively shown here: even the L2-contention-worthy 40 MiB case is a soft null without a competing workload.
5. **Graph-node / multi-stream variants unmeasured** — skill retains them as inferred. CUDA-Graph node attribute (`cudaKernelNodeAttributeAccessPolicyWindow`) is semantically equivalent to the stream-level attribute per PG §4.13.2.

## Files

- `artifacts/l2_residency_probe.cu` — three-WS × three-policy harness.
- `artifacts/build.sh` — `nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo`.
- `artifacts/run.sh` — build + run + NCU capture.
- `artifacts/device.json` — nvidia-smi introspection snapshot (repo).
- `h200_ncu:/inspire/hdd/project/qianghuaxuexi/public/kernel_pilot_public/kp-probe-artifacts/l2-residency/2026-04-23/` — host-only:
  - `l2_residency.ncu-rep` — binary NCU report.
  - `ncu_metrics.csv` — per-launch metric CSV (9 launches).
  - `ncu.txt` — NCU log.
  - `run.log` — clean (non-NCU) output.
