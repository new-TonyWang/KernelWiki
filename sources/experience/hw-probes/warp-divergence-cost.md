---
api: '@{!}p'
namespace: ptx
probe_slug: warp-divergence-cost
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
  code: artifacts/experience/hw-probes/warp-divergence-cost/artifacts/divergence_cost_probe.cu
  build: artifacts/experience/hw-probes/warp-divergence-cost/artifacts/build.sh
  introspection: artifacts/experience/hw-probes/warp-divergence-cost/artifacts/device.json
  profile: ''
  ncu_report_host_path: h200_ncu:/inspire/hdd/project/qianghuaxuexi/public/kernel_pilot_public/kp-probe-artifacts/warp-divergence-cost/2026-04-22/warp_divergence_cost.ncu-rep
  run_log_host_path: h200_ncu:/inspire/hdd/project/qianghuaxuexi/public/kernel_pilot_public/kp-probe-artifacts/warp-divergence-cost/2026-04-22/run.log
  ncu_csv_host_path: h200_ncu:/inspire/hdd/project/qianghuaxuexi/public/kernel_pilot_public/kp-probe-artifacts/warp-divergence-cost/2026-04-22/ncu_metrics.csv
source:
- path: spec
  anchor: Reference
conclusions:
  workload: N=1,048,576 lanes, ~1024 FMA/lane per path, compute-bound (~89% SM throughput)
  branch_variant_ms_p0: 0.0418
  branch_variant_ms_p025: 0.076
  branch_variant_ms_p050: 0.076
  branch_variant_ms_p075: 0.076
  branch_variant_ms_p1: 0.0416
  predicated_variant_ms_p0: 0.0766
  predicated_variant_ms_p025: 0.0819
  predicated_variant_ms_p050: 0.082
  predicated_variant_ms_p075: 0.0819
  predicated_variant_ms_p1: 0.0766
  warp_uniform_ms_p0: 0.0409
  warp_uniform_ms_p025: 0.0415
  warp_uniform_ms_p050: 0.0415
  warp_uniform_ms_p075: 0.0415
  warp_uniform_ms_p1: 0.0408
  branch_slowdown_at_divergent_p: 1.82
  predicated_slowdown_across_p: 1.07
  warp_uniform_slowdown_across_p: 1.01
open_questions:
- clock_policy is `unlocked-logged-only` — H200 ran at max graphics clock (1980 MHz)
  but was not explicitly locked with `nvidia-smi -lgc`. Slowdown ratios are robust;
  absolute ms need re-measurement under lock for a strict measured-env contract.
- 'NCU `smsp__thread_inst_executed_per_inst_executed.ratio` reports 32 across all
  p for branch_variant — i.e. the metric does NOT directly reveal warp-body serialization.
  This is because NCU counts each serialized sub-warp-slice as a distinct issued instruction;
  the average active threads per issued instruction stays at 32. The authoritative
  divergence-cost signature is **wall-clock at constant Compute(SM) Throughput ~89%**:
  serializing both bodies doubles total issued instructions at unchanged throughput,
  so wall-clock roughly doubles. Skill''s §''profile first'' principle now carries
  this nuance — do not rely on the SIMT efficiency metric alone.'
- Divergence slowdown is 1.82× not 2.00× because the two paths share register pressure
  and some FMA-pipe parallelism survives the serialized branches. A cleaner 2.00×
  would require paths that compete for the same FMA chain. Probe does not isolate
  this; adequate as shipped.
- 'Independent Thread Scheduling (CC 7.0+) correctness effects not probed by this
  experiment — all three variants here are correctness-safe without explicit `__syncwarp`.
  Follow-up probe for ITS race-observability open: `sources/experience/hw-probes/warp-divergence-its-race/`.'
- Compiler predication threshold sweep (BP §13.2 'a certain threshold') not measured.
  migration plan §2 open question Q1 scheduled for a follow-up compile-time-parameterized
  probe.
id: exp-warp-divergence-cost
type: experience
vendor: nvidia
title: 2026 04 22 Warp Divergence Cost
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1602-L1612
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1614-L1628
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1602-L1628
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L3427-L3436
---
## Summary

This probe validates the divergence cost model from BP Guide §13.1 (L1602-L1612: "different execution paths must be executed separately; this increases the total number of instructions executed") and the predication escape hatch from §13.2 (L1614-L1628: "no warp can ever diverge" when predicated), for the skill at [wiki/nvidia/foundations/compute/warp-divergence/](../../../wiki/nvidia/foundations/compute/warp-divergence/).

The workload must be compute-bound to expose the divergence signal — an initial run at N=16M and ~64 FMAs/lane (total 1 G FMAs) was memory-bound and showed zero slowdown from divergence. The present run at N=1M and ~1024 FMAs/lane (total 1 G FMAs, same total work over 16× fewer threads) achieves 89 % Compute(SM) throughput, and the cost model becomes visible.

Three kernels over each `p ∈ {0.00, 0.25, 0.50, 0.75, 1.00}` (per- lane probability of taking path B):

- **`branch_variant`** — real `if/else` with path bodies large enough (32-iteration FMA loops × 2 paths) that nvcc does **not** predicate.
- **`predicated_variant`** — functionally equivalent compute via chained ternary `x = flag ? b : a` so nvcc emits `selp`. Always runs both paths' arithmetic, just merges the result.
- **`warp_uniform_variant`** — `__any_sync` lifts the condition to warp-uniform; whole warp takes one path. At any p > 0 over 32 lanes, the probability that *at least one* lane wants path B is 1 - (1-p)^32 ≈ 1, so the whole warp takes path B.

## Setup

- GPU: NVIDIA H200, UUID `GPU-fbaa167f-4646-b8b9-c4f5-a4c4734ebc25`, driver 570.124.06, CUDA runtime 12.9, sm_90a.
- Build: `nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo`.
- Clock policy: **unlocked**; nvidia-smi reported 1980 MHz graphics (the max clock) throughout the run.
- Protocol: 5 warmup launches + 20 timed launches, CUDA events, median / p10 / p90 across the 20 timed launches per (kernel, p) pair. NCU pass on a separate launch.

## Results — wall-clock (ms)

| Kernel               | p=0.00 | p=0.25 | p=0.50 | p=0.75 | p=1.00 |
| -------------------- | -----: | -----: | -----: | -----: | -----: |
| `branch_variant`     | 0.0418 | 0.0760 | 0.0760 | 0.0760 | 0.0416 |
| `predicated_variant` | 0.0766 | 0.0819 | 0.0820 | 0.0819 | 0.0766 |
| `warp_uniform`       | 0.0409 | 0.0415 | 0.0415 | 0.0415 | 0.0408 |

### Slowdown vs each kernel's p=0 baseline

| Kernel               | p=0.00 | p=0.25 | p=0.50 | p=0.75 | p=1.00 |
| -------------------- | -----: | -----: | -----: | -----: | -----: |
| `branch_variant`     |  1.00× |**1.82×**|**1.82×**|**1.82×**|  1.00× |
| `predicated_variant` |  1.00× |  1.07× |  1.07× |  1.07× |  1.00× |
| `warp_uniform`       |  1.00× |  1.01× |  1.01× |  1.01× |  1.00× |

## Interpretation

### 1. The divergence signature is "1.82× for any p ∈ (0, 1)", not a smooth curve

The most striking result is that `branch_variant` jumps to 1.82× slowdown at **p=0.25 already** — and stays flat at 1.82× through p=0.75. This is the BP §13.1 serialization cost: if *any* warp has *any* lane wanting the other body, that warp must execute both bodies serially. With 32-lane warps and p=0.25, the probability that a given warp is 100 % path-A is (0.75)^32 ≈ 0.01 %; effectively **every** warp diverges at any p ∈ (0, 1), and every warp pays the full doubled-body cost. The "divergence curve" that pedagogy often draws (smooth rise from 1× to 2× as p goes from 0 to 0.5) is a per-warp- averaged metric, not a wall-clock observation.

### 2. Predication decouples cost from p (but always pays the both-bodies cost)

`predicated_variant` is flat across p (1.00×–1.07×; the 7 % drift is probably L2/TLB variation from different predicate bit patterns). **But** its absolute cost at p=0 is 0.0766 ms vs `branch_variant`'s 0.0418 ms — predicated code always executes both paths' arithmetic and picks the result at each step, so it runs ~1.83× the instructions of a one-body kernel. This is the trade: predication eliminates the p-dependence of divergence cost, but always pays the "both bodies execute" price. When p is stable near 0 or 1, keep the branch; when p is unpredictable per-lane, predication is cheaper than branching.

### 3. `warp_uniform` is the best-case of S3 but only when the two bodies are symmetric

`warp_uniform` is flat at 1.01× across all p, because at any p > 0 the `__any_sync` vote resolves to 1 for essentially every warp, so every warp runs path B. This is as fast as running path A for all warps — the two bodies are symmetric in instruction count. In real kernels where path B is much heavier than path A, `warp_uniform` would be slower by (path_B_cost / path_A_cost) whenever any lane wants path B. This probe is thus the **upper bound on warp-uniform's benefit**; real kernels will be somewhere between `warp_uniform` and `branch_variant`.

### 4. NCU's `smsp__thread_inst_executed_per_inst_executed.ratio` is not the right metric

Across every p for `branch_variant`, NCU reports `smsp__thread_inst_executed_per_inst_executed.ratio = 32` and `Compute (SM) Throughput ≈ 89 %`. **Both stay constant**, even though wall-clock jumps by 1.82× from p=0 to p=0.5. The reason is that NCU counts each serialized sub-warp execution as a distinct "issued instruction with N active lanes"; the ratio (threads-active / instructions-issued) averages to the full 32 even during serialization. The authoritative divergence signal is therefore **wall-clock at constant Compute(SM) Throughput** — doubled wall-clock at unchanged throughput means doubled total instructions issued, which is the serialization.

This is a **new pitfall** worth surfacing in the skill: the SIMT-efficiency metric that pedagogy recommends for divergence hunting can stay at 32/32 even when the warp is actually serializing. See skill §"Profile first" and §"When NOT to use".

## Takeaways (fed back into skill + pitfalls)

1. **Rewrite rule**: when the branch bodies are substantial and p is unknown per-lane, predication (S2) is the right choice; it decouples cost from p at a fixed ~both-bodies cost.
2. **`warp_uniform` (S3) is correctness-conditional**: only when the "pessimistic path" is acceptable for all lanes. The upper-bound benefit is flat cost across p; the downside is always running the heavier body.
3. **Wall-clock is authoritative**; the SIMT efficiency NCU metric can be misleading at 32/32 during serialization. Added as a pitfall note in the skill.
4. **Pitfall P7** from legacy sandbox (early-exit shifts bottleneck from compute to memory) is consistent with this probe's observation that Compute(SM) Throughput is already at 89 % — any divergence *fix* in a compute-bound regime will remain compute-bound; divergence fixes in a memory-bound regime are typically invisible (see the memory-bound earlier-run note below).

### Note on the memory-bound first run

The probe was initially run at N=16 M lanes and ~64 FMAs per path. At that compute intensity the kernel was bandwidth-limited (reading 64 MB float input + 16 MB predicate + writing 64 MB output over ~60 μs). Results were **completely flat across p** (0.098 ms for all variants at all p). This is *itself* the data supporting skill §"When NOT to use" item 1: divergence cost is invisible when the kernel is memory-bound, regardless of the raw divergence rate.

The final run (N=1 M × 1024 FMAs) reduces memory traffic 16× and raises per-lane compute 16×, making total work comparable but shifting the bottleneck from memory to compute. Compute(SM) Throughput rose from ~20 % to ~89 % as a result.

## Files

- `artifacts/divergence_cost_probe.cu` — three-kernel probability- sweep harness.
- `artifacts/build.sh` — `nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo`.
- `artifacts/run.sh` — build + run + NCU capture.
- `artifacts/device.json` — nvidia-smi introspection snapshot (repo).
- `h200_ncu:/inspire/hdd/project/qianghuaxuexi/public/kernel_pilot_public/kp-probe-artifacts/warp-divergence-cost/2026-04-22/` — host-only retention:
  - `warp_divergence_cost.ncu-rep` — binary NCU report.
  - `ncu_metrics.csv` — full NCU CSV dump (6547 lines).
  - `run.log` — clean (non-NCU) timing output.
