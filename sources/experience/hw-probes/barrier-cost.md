---
api: __syncthreads
namespace: cuda-runtime
probe_slug: barrier-cost
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
  code: sources/experience/hw-probes/barrier-cost/artifacts/barrier_cost_probe.cu
  build: sources/experience/hw-probes/barrier-cost/artifacts/build.sh
  introspection: sources/experience/hw-probes/barrier-cost/artifacts/device.json
  profile: ''
  ncu_report_host_path: h200_ncu:/inspire/hdd/project/qianghuaxuexi/public/kernel_pilot_public/kp-probe-artifacts/barrier-cost/2026-04-22/barrier_cost.ncu-rep
  run_log_host_path: h200_ncu:/inspire/hdd/project/qianghuaxuexi/public/kernel_pilot_public/kp-probe-artifacts/barrier-cost/2026-04-22/run.log
  ncu_csv_host_path: h200_ncu:/inspire/hdd/project/qianghuaxuexi/public/kernel_pilot_public/kp-probe-artifacts/barrier-cost/2026-04-22/ncu_metrics.csv
source:
- path: spec
  anchor: Reference
conclusions:
  workload: 10,000 barriers per kernel, single-block launch, block sizes {128, 256,
    512, 1024}
  syncwarp_ns_per_call_b128: 14.61
  syncwarp_ns_per_call_b1024: 22.13
  syncthreads_ns_per_call_b128: 31.81
  syncthreads_ns_per_call_b1024: 61.51
  mbarrier_ns_per_call_b128: 50.5
  mbarrier_ns_per_call_b1024: 137.17
  syncthreads_scaling_128_to_1024: 1.93
  mbarrier_scaling_128_to_1024: 2.72
  mbarrier_vs_syncthreads_b128: 1.59
  mbarrier_vs_syncthreads_b1024: 2.23
open_questions:
- clock_policy is `unlocked-logged-only` — H200 ran at max graphics clock (1980 MHz)
  but was not explicitly locked. Absolute ns/call numbers should be retaken under
  lock for a strict measured-env contract; ratios are robust.
- 'The probe measures BARE barrier cost only — no overlap work between `arrive` and
  `wait` for mbarrier. The skill''s S2 (arrive/wait split for overlap) requires a
  separate probe with real independent compute in the overlap window. Current result
  confirms pitfall P6: on bare cost, mbarrier is 1.6-2.2x slower than __syncthreads.
  Follow-up probe: `sources/experience/hw-probes/barrier-async-overlap/`.'
- 'NCU CSV captured only the first kernel''s launches (syncwarp_loop; --launch-count
  12 slots exhausted before reaching syncthreads/mbarrier). This is acceptable because
  wall-clock ns/call is the authoritative measurement; the NCU pass''s value was corroboration
  of Compute SOL ~ 0 % (expected: single-block kernels cannot occupy the 132-SM H200).
  The dependency on further NCU detail is not on the critical path for this skill.'
- sm_90a named-barrier throughput (PTX §9.7.13.1 / BP §12.1.3 claim of 16 ops/clock
  on sm_8.x) was not re-measured for sm_9.0a. The measured ns/call at B=1024 (61.5
  ns = ~122 cycles) is consistent with 16 ops/clock averaged across all 32 warps of
  a 1024-thread block, but a direct throughput probe (N warps all calling __syncthreads
  simultaneously) is open.
- Cluster-level barrier (`barrier.cluster.arrive/wait`, sm_90+) not measured — blocked
  on wiki/nvidia/hardware/thread-block-cluster/ bootstrap (bucket F).
id: exp-barrier-cost
type: experience
vendor: nvidia
title: 2026 04 22 Barrier Cost
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1402-L1404
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L3647-L3704
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1402-L1404
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L3647-L3704
---
## Summary

This probe measures the **bare per-call cost** of three synchronization primitives at four block sizes, to back the cost model in [wiki/nvidia/foundations/sync/barrier-optimization/skill.md](../../../wiki/nvidia/foundations/sync/barrier-optimization.md). BP Guide §12.1.3 (L1402-L1404) gives the per-SM throughput for `__syncthreads` (16 ops/clock on sm_7.x / sm_8.x) but does not compare against `__syncwarp` or `cuda::barrier` arrive+wait. PG §3.2.4.2 (L3647-L3704) documents that async barriers give benefit *via overlap*, but does not quantify the bare-cost gap when there is no overlap.

Three kernels over block sizes {128, 256, 512, 1024}:

- **`syncwarp_loop`** — inner loop of 10,000 `__syncwarp(0xFFFFFFFFu)` calls.
- **`syncthreads_loop`** — inner loop of 10,000 `__syncthreads()` calls plus one cheap `dummy += 1` payload to prevent the compiler from collapsing the loop.
- **`mbarrier_loop`** — inner loop of 10,000 `cuda::barrier::arrive()`
  + `cuda::barrier::wait(token)` round-trips, with one-time `init()`.

## Setup

- GPU: NVIDIA H200, UUID `GPU-fbaa167f-4646-b8b9-c4f5-a4c4734ebc25`, driver 570.124.06, CUDA runtime 12.9, sm_90a.
- Build: `nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo`.
- Clock policy: **unlocked**; nvidia-smi reported 1980 MHz (the max clock) throughout the run.
- Protocol: 5 warmup launches + 20 timed launches per (kernel, block size) pair; CUDA events; median ms / 10,000 = ns/call.
- Single-block launches — so per-call cost is measured under "SM fully occupied by one block" (no warps from other blocks to hide the barrier wait). This is the worst-case shape for block barriers.

## Results — ns per barrier call

| Block | `__syncwarp` | `__syncthreads` | `mbarrier arrive+wait` | mbarrier / syncthreads |
| ----- | -----------: | --------------: | ---------------------: | ---------------------: |
| 128   |        14.61 |           31.81 |                  50.50 |                  1.59× |
| 256   |        14.61 |           36.35 |                  52.53 |                  1.45× |
| 512   |        15.43 |           44.44 |                  74.37 |                  1.67× |
| 1024  |        22.13 |           61.51 |                 137.17 |              **2.23×** |

### Scaling factor (cost at B=1024 ÷ cost at B=128)

| Primitive              | Scaling factor |
| ---------------------- | -------------: |
| `__syncwarp`           |          1.51× |
| `__syncthreads`        |          1.93× |
| `mbarrier arrive+wait` |      **2.72×** |

## Cycles per call (at 1980 MHz)

| Block | `__syncwarp` | `__syncthreads` | `mbarrier arrive+wait` |
| ----- | -----------: | --------------: | ---------------------: |
| 128   |        ~29 c |          ~63 c |                 ~100 c |
| 256   |        ~29 c |          ~72 c |                 ~104 c |
| 512   |        ~31 c |          ~88 c |                 ~147 c |
| 1024  |        ~44 c |         ~122 c |                 ~272 c |

## Interpretation

### 1. `__syncwarp` is near-flat across block size (as expected)

`__syncwarp` operates on a single warp at a time regardless of block size. The small drift from 14.6 ns at B=128 to 22.1 ns at B=1024 (1.51×) reflects warp-scheduler contention: with 32 warps in flight at B=1024, each warp waits longer to issue its barrier when multiple warps try simultaneously. The instruction itself is ~7 cycles per the NCU-measured "Warp Cycles Per Issued Instruction" for `syncwarp_loop`; the remaining ~22 cycles at B=1024 are warp- scheduler queuing.

### 2. `__syncthreads` scales weakly with block size (1.93× for 8× more threads)

As block grows from 128 (4 warps) to 1024 (32 warps), the barrier must wait for all 32 warps to arrive and still only costs 1.93× as much as at 4 warps. This is consistent with BP §12.1.3's "16 ops/ clock" throughput at the SM level — adding warps costs ~linear extra cycles to drain, bounded above by the serial "last warp to arrive" stall. On sm_9.0a the measured 122-cycle cost at B=1024 is in the same order of magnitude as `ceil(32 warps / 16 ops-per-clock) * constant`.

### 3. mbarrier arrive+wait is 1.6–2.2× slower than `__syncthreads` (bare cost)

On H200 sm_9.0a, `cuda::barrier::arrive()` + `cuda::barrier::wait()` in a tight loop is consistently *slower* than `__syncthreads()`, and the gap widens with block size (1.59× at B=128 → 2.23× at B=1024). Two causes, both expected:

- **Instruction count**: mbarrier is a memory-backed object; the arrive+wait pair emits `mbarrier.arrive.shared::cta.b64` (returns a token) + `mbarrier.try_wait.shared::cta.b64` (blocks on token). That is ~2–3× the instruction count of a single `bar.sync`.
- **Phase-state memory traffic**: the 64-bit mbarrier state lives in smem; every arrive/wait has a read-modify-write cycle through the shared memory subsystem, whereas `bar.sync` uses the dedicated on-SM named-barrier hardware (16 slots) without going through smem.

The skill's S2 ("arrive/wait split for overlap") therefore **only pays off when there is real independent work in the overlap window** — typically ≥ 100 cycles of compute between `arrive` and `wait`. Pitfall P6 in the skill's pitfalls file documents this; the probe gives the bare-cost backing number.

### 4. NCU's Compute SOL is ~0.11 % for these kernels (expected)

With a single 128–1024 thread block on a 132-SM H200, 131 SMs sit idle for the duration of the probe. Compute SOL and Memory SOL are both near zero. This is by design — the probe measures *instruction cost per call*, not throughput. NCU captured only the first kernel (`syncwarp_loop`) before exhausting the 12-launch window; the captured fraction corroborates the low-SOL expectation but adds nothing beyond the wall-clock measurement.

## Takeaways (fed back into skill + pitfalls)

1. **The block-barrier cost hierarchy on H200 sm_9.0a is warp → block → mbarrier**, with ratios ~1 : ~2–3 : ~3–6 at typical block sizes. Skill §S1 "narrow the scope" is backed by these numbers: dropping from `__syncthreads` to `__syncwarp` where correct saves ~2× the per-call cost.
2. **mbarrier bare cost is a loss without overlap**. Pitfall P6 (legacy-anecdotal) is upgraded to measured: mbarrier arrive+wait alone costs 1.6–2.2× a `__syncthreads`; only S2 (real independent work) recovers the gap.
3. **`__syncthreads` cost scales sub-linearly with block size** on H200 (1.93× for 8× the threads). Increasing block size doesn't double the barrier cost — it's mostly additive waiting, bounded by the "slowest warp" cost.
4. **Cluster barriers not measured.** Follow-up probe blocked on `wiki/nvidia/hardware/thread-block-cluster/` bootstrap.

## Files

- `artifacts/barrier_cost_probe.cu` — three-kernel, block-size-swept harness.
- `artifacts/build.sh` — `nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo`.
- `artifacts/run.sh` — build + run + NCU capture.
- `artifacts/device.json` — nvidia-smi introspection snapshot (repo).
- `h200_ncu:/inspire/hdd/project/qianghuaxuexi/public/kernel_pilot_public/kp-probe-artifacts/barrier-cost/2026-04-22/` — host-only:
  - `barrier_cost.ncu-rep` — binary NCU report.
  - `ncu_metrics.csv` — 240-line CSV (covers first kernel only).
  - `run.log` — clean (non-NCU) output.
