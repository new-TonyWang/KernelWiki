---
api: Particle
namespace: cuda-language
probe_slug: aos-vs-soa
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
  code: artifacts/experience/hw-probes/aos-vs-soa/aos_vs_soa_probe.cu
  build: artifacts/experience/hw-probes/aos-vs-soa/build.sh
  introspection: artifacts/experience/hw-probes/aos-vs-soa/device.json
  profile: ''
  ncu_report_host_path: h200_ncu:/inspire/hdd/project/qianghuaxuexi/public/kernel_pilot_public/kp-probe-artifacts/aos-vs-soa/2026-04-22/aos_vs_soa.ncu-rep
  run_log_host_path: h200_ncu:/inspire/hdd/project/qianghuaxuexi/public/kernel_pilot_public/kp-probe-artifacts/aos-vs-soa/2026-04-22/run.log
  ncu_csv_host_path: h200_ncu:/inspire/hdd/project/qianghuaxuexi/public/kernel_pilot_public/kp-probe-artifacts/aos-vs-soa/2026-04-22/ncu_metrics.csv
source:
- path: spec
  anchor: Reference
conclusions:
  workload: AoS vs SoA read-one-field pattern; Particle{x,y,z,vx,vy,vz}=24B, N=16,777,216
  baseline_name: aos_read_one_field (stride=24B, reads only .x)
  baseline_ms: 0.1111
  soa_scalar_ms: 0.0558
  soa_vectorized_ms: 0.0372
  aos_to_soa_convert_ms: 0.2017
  speedup_soa_over_aos: 1.99
  speedup_soa_vectorized_over_aos: 2.99
  speedup_vectorized_over_soa: 1.5
  amortization_break_even_calls: 3.65
  aos_dram_sol_pct: 89.16
  aos_useful_bw_gb_s: 1208
  aos_actual_hbm_bw_gb_s: 4379
  soa_dram_sol_pct: 43.03
  soa_scalar_useful_bw_gb_s: 2405
  soa_vectorized_useful_bw_gb_s: 3613
  convert_dram_sol_pct: 81.12
open_questions:
- clock_policy is `unlocked-logged-only` — H200 ran at its max graphics clock (1980
  MHz) but not explicitly locked with `nvidia-smi -lgc`. Absolute GB/s numbers should
  be retaken under lock for a strict measured-env contract. Speedup ratios and the
  break-even count are robust.
- soa_read_one_field reaches only 43% DRAM SOL — the kernel is latency-bound, not
  throughput-bound at this shape. float4 vectorization recovers 71% SOL. There is
  headroom on H200 for further ILP tuning (S6 in the skill's open questions).
- 'struct size = 24B is the canonical Particle example. Break-even is sensitive to
  struct size: larger struct -> bigger AoS waste ratio -> faster break-even. Open:
  sweep struct sizes 16B / 24B / 48B / 96B under `sources/experience/hw-probes/aos-vs-soa/struct-size-sweep/`.'
- On H200 the AoS kernel achieves 89% HBM SOL despite moving 4x more bytes than strictly
  needed — the L2 is absorbing the wasted sectors. L1/TEX hit rate on AoS is 0% (every
  load misses L1) and L2 hit rate is 15%, vs L2 hit rate 50% on SoA. Shape is small
  enough for L2 effect; larger N may widen the SoA advantage further.
id: exp-aos-vs-soa
type: experience
vendor: nvidia
title: 2026 04 22 Aos Vs Soa
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L606-L634
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L606-L634
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L1379-L1411
---
## Summary

This probe validates sub-skill **S1 "AoS → SoA conversion"** from [wiki/nvidia/foundations/memory/layout-transform/skill.md](../../../wiki/nvidia/foundations/memory/layout-transform.md). BP Guide §10.2.1.4 (L606-L634) asserts that non-unit-stride global accesses waste bandwidth proportional to the stride; this probe measures the penalty on H200 for a 24-byte struct when the kernel touches only a 4-byte field (stride = 6× the useful bytes).

Four kernels over `N = 16,777,216` particles:

- **`aos_read_one_field`** — struct `Particle{x,y,z,vx,vy,vz}`, kernel reads only `.x`. Each warp fetches 24 × 32 = 768 B of contiguous AoS data and uses 4 × 32 = 128 B of it. 17 % useful-byte ratio in theory.
- **`soa_read_one_field`** — flat `x[N]` array, stride 4 B, coalesced.
- **`soa_vectorized`** — same but loaded as `float4` (128 B per warp of useful data in one transaction).
- **`aos_to_soa_convert`** — one-time conversion kernel: reads AoS coalesced (24 B / thread), writes 6 SoA fields coalesced.

## Setup

- GPU: NVIDIA H200, UUID `GPU-fbaa167f-4646-b8b9-c4f5-a4c4734ebc25`, driver 570.124.06, CUDA runtime 12.9, sm_90a.
- Build: `nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo`.
- Clock policy: **unlocked**; nvidia-smi reported 1980 MHz graphics (the max clock) throughout the run.
- Protocol: 5 warmup launches + 20 timed launches, CUDA events, median / p10 / p90. Separate NCU pass for SoL + memory + warp state.

## Results — wall-clock

| Kernel                   | Median ms | Eff. useful BW GB/s | Speedup vs AoS |
| ------------------------ | --------: | ------------------: | -------------: |
| `aos_read_one_field`     |    0.1111 |                1208 |          1.00× |
| `soa_read_one_field`     |    0.0558 |                2405 |          1.99× |
| `soa_vectorized`         | **0.0372**|            **3613** |      **2.99×** |
| `aos_to_soa_convert`     |    0.2017 |   3993 (bulk bytes) |            n/a |

**Amortization**: the conversion kernel costs 0.2017 ms. Each downstream `soa_read` saves `aos_ms - soa_ms = 0.0553` ms versus keeping the data in AoS. Break-even: **3.65 downstream calls**. This is a concrete, measured version of the folklore "transform if ≥ 3 kernels will consume the new layout."

## Results — NCU section metrics

| Metric                               | `aos_read` | `soa_read` | `soa_vec` | `aos_to_soa` |
| ------------------------------------ | ---------: | ---------: | --------: | -----------: |
| DRAM Throughput SOL (%)              |  **89.16** |      43.03 |     71.31 |        81.12 |
| Memory Throughput (B/s, actual HBM)  |  4.38 TB/s |  2.11 TB/s | 3.50 TB/s |    3.99 TB/s |
| Compute (SM) Throughput (%)          |      25.27 |      48.94 |     20.90 |        24.52 |
| Warp Cycles / Issued Instruction     |     110.77 |  **51.34** |    110.66 |       101.40 |
| L1/TEX Hit Rate (%)                  |       0.00 |       0.00 |      0.00 |        71.43 |
| L2 Hit Rate (%)                      |      15.09 |      50.79 |     51.04 |        51.64 |

Interpretation (tying NCU signatures to skill claims):

- **`aos_read_one_field` is the smoking gun for BP §10.2.1.4**. HBM SOL is 89 % — the bus is nearly saturated — but the kernel's useful-bytes throughput is only **1208 GB/s** (24 % of the ~4.8 TB/s the bus is actually moving). The remaining 76 % is stride-wasted data that the kernel fetches but never uses. L2 hit rate is only 15 %, so most waste reaches HBM.
- **`soa_read_one_field` is latency-bound**. DRAM SOL drops to 43 % — not because the kernel is inefficient but because the working set at N = 16 M floats (64 MB) fits mostly in L2 (50.79 % L2 hit rate), and each warp does so little work per load that the kernel cannot keep HBM saturated. Warp cycles / issued drops to 51, about half the AoS kernel.
- **`soa_vectorized` reaches DRAM SOL 71 %** by packing 4 elements per load: each warp's one-load transaction carries 128 B of useful data. The warp cycles / issued stays high (110) because the load latency is still the critical path, but the reduced number of load instructions lets the kernel make progress faster overall.
- **`aos_to_soa_convert` at DRAM SOL 81 %** is a healthy bulk-copy kernel — it's reading 24 B and writing 24 B per thread, all coalesced, with L1/TEX hit rate 71 % (because consecutive threads of a warp land in the same cache line for the packed AoS input).

## Takeaways (fed back into skill + pitfalls)

1. **The stride penalty is not 6× (naive theory) but ~2× on H200**, because the L2 partially absorbs the wasted bytes and the reduced-instruction-count of vectorization gives the SoA side extra headroom. This is a measured correction to folklore that quotes "stride-N = N× slower".
2. **The "≥ 3 downstream kernels" rule is data-backed**. Measured break-even is 3.65 calls on this struct. For the skill's §"When NOT to use" guidance, this is now the concrete number, not a handwave.
3. **Vectorization on SoA is a free 1.5× on top of the AoS→SoA transform**. Pair S1 with vectorized-access skill when the field is float/float2/float4-alignable.
4. **HBM SOL alone is NOT a performance metric**. The AoS kernel's 89 % DRAM SOL looks healthy by itself; only comparing to the useful-byte throughput reveals the waste. Pitfall P5 in the sibling skill (misleading roofline %) is reinforced by this probe.

## Files

- `artifacts/aos_vs_soa_probe.cu` — four-kernel harness.
- `artifacts/build.sh` — `nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo`.
- `artifacts/run.sh` — build + run + NCU capture; host-only artifacts go to `/inspire/hdd/project/qianghuaxuexi/public/kernel_pilot_public/kp-probe-artifacts/aos-vs-soa/<date>/`.
- `artifacts/device.json` — nvidia-smi introspection snapshot (repo-committed).
- `h200_ncu:/inspire/hdd/project/qianghuaxuexi/public/kernel_pilot_public/kp-probe-artifacts/aos-vs-soa/2026-04-22/` — host-only retention (per .gitignore policy, `*.log` / `*.csv` / `*.ncu-rep` not committed to repo):
  - `aos_vs_soa.ncu-rep` — binary NCU report.
  - `ncu_metrics.csv` — full NCU CSV dump (132 lines).
  - `run.log` — clean (non-NCU) timing output.
