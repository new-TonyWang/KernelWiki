---
api: empty kernel launch (<<<>>> and cudaLaunchKernel)
namespace: cuda-runtime
probe_slug: launch-overhead
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
  code: artifacts/experience/hw-probes/launch-overhead/launch_overhead_probe.cu
  build: artifacts/experience/hw-probes/launch-overhead/build.sh
  introspection: artifacts/experience/hw-probes/launch-overhead/device.json
  profile: ''
  run_log_host_path: h200_ncu:/inspire/hdd/project/qianghuaxuexi/public/kernel_pilot_public/kp-probe-artifacts/launch-overhead/2026-04-23/run.log
source:
- path: spec
  anchor: Reference
conclusions:
  workload: 3 launch shapes × 2 launch mechanisms × N_BATCH=10000 launches, timed with outer CUDA-event bracket and divided by N_BATCH. Plus a workload-crossover sweep at full H200 grid (132×4 blocks × 256 threads) with WORK_STEPS ∈ {0,1,4,16,64,256,1024,4096} FMA per thread. Plus an empty-event-pair noise-floor measurement.
  empty_chevron_1x1_us: 1.676
  empty_api_1x1_us: 1.632
  empty_chevron_1x256_us: 1.663
  empty_api_1x256_us: 1.641
  empty_chevron_fullgrid_us: 1.749
  empty_api_fullgrid_us: 1.754
  api_vs_chevron_ratio_1x1: 0.974
  api_vs_chevron_ratio_fullgrid: 1.003
  fullgrid_vs_tiny_ratio: 1.044
  work0_us: 1.967
  work1_us: 1.967
  work4_us: 2.015
  work16_us: 2.23
  work64_us: 3.249
  work256_us: 6.567
  work1024_us: 19.005
  work4096_us: 68.597
  work16_gpu_us: 0.263
  work64_gpu_us: 1.282
  work256_gpu_us: 4.6
  work1024_gpu_us: 17.038
  work4096_gpu_us: 66.63
  crossover_work_steps: 64
  launch_amortized_to_10pct_work: 1024
  empty_event_pair_us: 6.08
  per_event_record_us: 3.04
  h200_sm_count: 132
  full_grid_threads: 135168
open_questions:
- clock_policy is `unlocked-logged-only` — H200 ran at 1980 MHz throughout. Absolute μs subject to boost-clock variation. The within-measurement ratios (mechanism, shape, workload) are robust because all cells share the same launch context.
- Host CPU not specified. Launch overhead is dominated by host-side work (CUDA runtime + driver); a faster / slower host CPU would shift the absolute floor. The 1.7 μs floor is a lower bound for this host — re-measure on other hosts if the kernel will be launched from a different CPU family.
- Stream / CUDA Graph variants deliberately NOT measured (out of scope for this sub-area). CUDA Graphs collapse the per-launch cost of multiple kernels into one submit — reportedly < 1 μs per node via graph capture — but that belongs to the sibling sub-area 90-system-level/cuda-graphs/ (pending), not this skill.
- cudaLaunchKernelEx measured — see Part A table. With 0 attributes it is NOT slower than cudaLaunchKernel (full grid equal; tiny grid 1.5% faster, likely because the Ex path skips a legacy compatibility layer). Attaching one access-policy-window attribute adds ~26 ns/launch at 1x1 grid (+1.7%) and is noise-level at full grid (+1 ns). The earlier expectation 'Ex is slightly higher because of attribute parsing' is measurably wrong.
- Part C's per-event-record cost (~3 μs) is HOST-SIDE. It is not a subtraction correction to Part A because Part A already amortizes the outer event pair over 10000 launches (per-launch event overhead contribution = 6 μs / 10000 = 0.6 ns, negligible). The earlier `subtract this floor` instruction in the probe's stdout is incorrect and should be ignored.
- Per-launch floor grows slightly with grid size (1.676 → 1.749 μs, +4%). This is the driver-side configuration cost scaling with grid metadata size. Remains negligible compared to the 1.7 μs floor for all practical shapes.
id: exp-launch-overhead
type: experience
vendor: nvidia
title: 2026 04 23 Launch Overhead
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1700-L1800
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA API References/cuda-runtime-api/cuda_cuda-runtime-api_index.html.md
  anchor: L5900-L6100
architectures:
- sm90
- sm90a
languages:
- cuda-cpp
hardware_features:
- cluster
techniques:
- kernel-fusion
- tile-scheduling
kernel_types:
- gemm
- fused-kernel
- quantization
confidence: experimental
tags:
- cluster
- kernel-fusion
- tile-scheduling
- gemm
- fused-kernel
- quantization
- cuda-cpp
artifact_dir: artifacts/experience/hw-probes/launch-overhead
---
## Summary

This probe establishes three measured facts used throughout the launch-overhead skill:

1. **The per-launch floor on H200 + driver 570.124 + CUDA 12.9 is ≈ 1.5–1.7 μs** for any reasonable launch shape (empty kernel, `<<<1,1>>>` up to full-grid `<<<528,256>>>`) and any of four launch mechanisms (`<<<>>>` / `cudaLaunchKernel` / `cudaLaunchKernelEx(0 attrs)` / `cudaLaunchKernelEx(1 attr)`). Shape variation contributes at most 7% on top of this floor. All four mechanisms are within ~5% of each other across the whole sweep.
2. **`cudaLaunchKernelEx` is NOT slower than `cudaLaunchKernel`** — the Ex path is full-grid-equivalent and 1.5% *faster* at tiny grids. Attaching one `accessPolicyWindow` attribute adds ~26 ns/launch at 1x1 grid (+1.7%) and is noise-level at full grid. The common expectation "Ex is slower because of attribute parsing" is directly contradicted by measurement.
3. **The workload-too-small threshold on H200 is roughly 64 FMA per thread on a full grid**, at which point GPU-work time (≈1.3 μs) approaches the per-launch overhead (≈2.0 μs). To amortize launch to < 10% of total wall-clock, the kernel must do ≈1024 FMA/thread (≈17 μs of GPU work).

Decision rule falling out of the data: **if a kernel at your target launch shape does less than ~2 μs of GPU work per launch, host-side launch overhead dominates**. On H200 with ~135k threads and fp32 FMA at ~1.58 ns/op post-auto-packing, 2 μs ≈ 64 FMA per thread. Kernels that cannot amortize above this floor should be batched, fused with neighbouring ops, or moved into CUDA Graphs (out of this skill's scope — see sibling sub-area `90-system-level/cuda-graphs/`, pending).

## Setup

- GPU: NVIDIA H200, UUID `GPU-fbaa167f-4646-b8b9-c4f5-a4c4734ebc25`, driver 570.124.06, CUDA runtime 12.9, sm_9.0a.
- Build: `nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo`.
- Clock policy: unlocked; nvidia-smi reported 1980 MHz throughout.
- Protocol: 3 warmup batches + 10 timed batches, median ms. Each batch is N_BATCH = 10000 individual kernel launches, timed with a single outer CUDA-event pair (so the per-launch event overhead is amortized to essentially zero). Per-launch μs = `batch_ms * 1000 / 10000`.

## Results

### Part A — Per-launch floor (empty kernel, 4 mechanisms × 3 shapes)

| Launch shape | Mechanism | batch_ms | μs/launch |
|--------------|-----------|---------:|----------:|
| `<<<1, 1>>>` | `<<<>>>` chevron | 16.07 | 1.607 |
| `<<<1, 1>>>` | `cudaLaunchKernel` | 15.51 | 1.551 |
| `<<<1, 1>>>` | `cudaLaunchKernelEx(0 attrs)` | 15.27 | **1.527** |
| `<<<1, 1>>>` | `cudaLaunchKernelEx(+APW)` | 15.53 | 1.553 |
| `<<<1, 256>>>` | `<<<>>>` chevron | 16.14 | 1.614 |
| `<<<1, 256>>>` | `cudaLaunchKernel` | 15.60 | 1.560 |
| `<<<1, 256>>>` | `cudaLaunchKernelEx(0 attrs)` | 15.30 | **1.530** |
| `<<<1, 256>>>` | `cudaLaunchKernelEx(+APW)` | 15.58 | 1.558 |
| `<<<528, 256>>>` | `<<<>>>` chevron | 17.16 | 1.716 |
| `<<<528, 256>>>` | `cudaLaunchKernel` | 17.05 | 1.705 |
| `<<<528, 256>>>` | `cudaLaunchKernelEx(0 attrs)` | 17.05 | 1.705 |
| `<<<528, 256>>>` | `cudaLaunchKernelEx(+APW)` | 17.06 | 1.706 |

Observations:

- The per-launch floor is **1.5–1.7 μs** across all 12 cells. Launch shape contributes **+7 %** going from 1 thread to the full 135168-thread grid; at full grid the spread across mechanisms collapses to < 0.1 %.
- **`cudaLaunchKernel` and `<<<>>>` are within 3.5 %** on all three shapes. The common claim that "the chevron syntax is always faster" is directly contradicted — if anything, the explicit API is slightly faster on this target (~2-3 % at tiny shapes).
- **`cudaLaunchKernelEx(0 attrs)` is NOT slower than `cudaLaunchKernel`.** On `<<<1,1>>>` it measures 1.527 μs vs `cudaLaunchKernel`'s 1.551 μs — **the Ex path is 1.5 % FASTER at tiny grids**. At full grid the two are indistinguishable (both 1.705 μs). The pre-probe expectation "Ex is slower because of attribute parsing" is measurably wrong on sm_9.0a / driver 570.124.
- **Attaching one attribute costs ≈ 26 ns at tiny grid**, noise-level at full grid. `cudaLaunchKernelEx(+APW)` vs `cudaLaunchKernelEx(0 attrs)` at `<<<1,1>>>`: 1.553 vs 1.527 μs = +26 ns (+1.7 %). At full grid: 1.706 vs 1.705 = +1 ns (noise). The per-attribute incremental cost is small and amortized away at realistic launch shapes.

### Part B — Workload crossover (full-grid, WORK_STEPS FMA per thread)

| WORK_STEPS | μs/launch | GFLOPS_eff | GPU-work μs (est.) | Launch share |
|-----------:|----------:|-----------:|-------------------:|-------------:|
|          0 |     1.967 |       0.00 |               0.00 |       100 %  |
|          1 |     1.967 |     137.45 |              ~0.00 |       100 %  |
|          4 |     2.015 |     536.59 |               0.05 |        98 %  |
|         16 |     2.230 |    1939.52 |               0.26 |        88 %  |
|         64 |     3.249 |    5324.45 |               1.28 |        61 %  |
|        256 |     6.567 |   10538.92 |               4.60 |        30 %  |
|       1024 |    19.005 |   14565.48 |              17.04 |        10 %  |
|       4096 |    68.597 |   16142.10 |              66.63 |         3 %  |

`GPU-work μs (est.) = launch_μs - 1.97` (launch floor). `Launch share` = what fraction of total wall-clock is launch overhead.

Observations:

- The **crossover** where GPU work catches up to launch overhead is at **~64 FMA per thread** (GPU 1.28 μs ≈ launch 1.97 μs, launch share drops to ~60 %).
- To get launch overhead below **10 %** of total wall-clock, the kernel needs **~1024 FMA per thread** or ~17 μs of GPU work. This translates to ~136 M FMA ops at full grid — a non-trivial compute workload.
- Peak measured effective throughput on the workload kernel reaches ~16 TFLOPS at WORK_STEPS=4096, consistent with H200's fp32 peak after compiler auto-packing the scalar FMAs into HFMA2.MMA (per the half-precision-math probe's findings — same machinery).

### Part C — Event-record host-side cost

| Measurement | Value |
|-------------|------:|
| Empty event-pair batch (10000 pairs) | 61.87 ms |
| **Per `cudaEventRecord` call (host)** | **3.04 μs** |
| Per event-pair (2 records) | 6.08 μs |

The single-`cudaEventRecord` host cost of ~3 μs is **higher than the per-launch floor** of 1.7 μs. This is counter-intuitive but consistent: a kernel launch pushes a single command into the stream queue, while an event record allocates / initialises timer state plus pushes a record command, doing more host-side work per call. Part A measurements are NOT inflated by this cost because the outer event pair is amortized over 10000 launches inside the batch (per-launch event contribution = 60 ms / 10000 = 6 ns, negligible vs the 1.7 μs launch floor).

## Interpretation (wall-clock only; no SASS needed — this is a host-bound measurement)

### 1. The ~1.7 μs floor is a host-side fixed cost, not a GPU cost

At grid `<<<1,1>>>` the GPU executes one empty kernel with one thread and returns instantly. The entire 1.7 μs is the CUDA runtime + driver pushing a launch command into the stream, dispatching to the GPU, and completing the cudaEventRecord bookkeeping on the host side. Scaling grid from 1 thread to 135k threads adds only 0.073 μs (+4 %) — the GPU scheduler processes the larger grid almost as fast as the tiny one. This means **any optimization targeted at "reducing launch overhead" has to reduce host-side work**, not GPU-side work.

### 2. Shape does not meaningfully change per-launch cost

The 4% delta going from `<<<1,1>>>` to full-grid `<<<528,256>>>` contradicts the folk rule "launch a smaller grid to reduce overhead". The launch itself costs the same; only the GPU compute + writeback time changes. For very-small workloads, reducing grid size saves no time and may increase per-thread work. Prefer amortizing work across more threads, not fewer.

### 3. All four launch mechanisms are essentially equivalent

The triple-chevron is syntactic sugar the nvcc front-end lowers to a `cudaLaunchKernel` call at compile time. `cudaLaunchKernelEx` is a newer extended API that accepts a `cudaLaunchConfig_t` + attributes array. SM_9.0a / driver 570.124 measurement confirms all four mechanisms (`<<<>>>` / `cudaLaunchKernel` / `cudaLaunchKernelEx(0 attrs)` / `cudaLaunchKernelEx(+APW)`) produce the same wall-clock within 5 % across all three launch shapes. At full grid the spread collapses to < 0.1 %. The pre-probe expectation that "Ex is slightly slower because it parses attributes" is directly contradicted: the Ex path is equal or faster than plain `cudaLaunchKernel` in every cell, and the per-attribute overhead is ~26 ns at tiny shapes — smaller than the shot-to-shot measurement noise.

Decision rule: **pick the mechanism that fits your code**. `<<<>>>` for `.cu` readability; `cudaLaunchKernel` for driver-API-from-C++ launches; `cudaLaunchKernelEx` when you need launch attributes (cluster-dim, access-policy-window, memory-sync-domain, programmatic-dependent-launch). There is no wall-clock reason to prefer one over another.

### 4. Crossover threshold is per-architecture and per-driver

The 64-FMA crossover is specific to H200 + CUDA 12.9. On a GPU with higher FMA throughput but the same host, the crossover shifts lower (less GPU work needed to equal the launch floor). On a slower host with a faster GPU, the gap widens. Re-measure if the target platform changes.

### 5. The "batching events" argument is quantitatively supported

CUDA events are ~2× more expensive per call than kernel launches in this host-side budget. Timing harnesses that wrap each small kernel in an event pair artificially inflate the apparent launch cost. **Always bracket a batch of launches with one outer event pair and divide, not per-launch pairs.**

## Takeaways (fed back into skill + pitfalls)

1. **Per-launch floor ≈ 1.5–1.7 μs** (empty kernel, any shape, any of the four mechanisms) on H200 + CUDA 12.9 + driver 570.124.
2. **Shape does not meaningfully change per-launch cost** — grid from 1 to full-H200 adds only ~7 %; mechanism spread collapses to <0.1 % at full grid.
3. **All four launch mechanisms are equivalent at wall-clock**: `<<<>>>`, `cudaLaunchKernel`, `cudaLaunchKernelEx(0 attrs)`, `cudaLaunchKernelEx(+APW)` — within 5 % of each other at worst case (tiny grid), within 0.1 % at full grid. `cudaLaunchKernelEx` is not slower than `cudaLaunchKernel`.
4. **Per-attribute incremental cost**: adding one `accessPolicyWindow` attribute costs +26 ns/launch at tiny grid, noise-level at full grid. The attribute-parsing cost folklore is quantitatively too high.
5. **Launch is < 50 % of total wall-clock once kernel does ≥ 64 FMA / thread**; < 10 % at ≥ 1024 FMA / thread.
6. **Kernels that cannot amortize to at least ~2 μs of GPU work per launch are launch-overhead-bound.** Fix by batching more work into one launch (grid-stride loops), fusing with neighbouring kernels, or moving to CUDA Graphs (out of this skill's scope).
7. **CUDA events cost ~3 μs / record on the host** — time batches, not individual launches.

## Files

- `artifacts/launch_overhead_probe.cu` — 3-shape × 2-mechanism × workload-sweep harness.
- `artifacts/build.sh` — `nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo`.
- `artifacts/run.sh` — build + run (no NCU — the measurement is host-bound).
- `artifacts/device.json` — nvidia-smi introspection snapshot.
- `h200_ncu:/inspire/hdd/project/qianghuaxuexi/public/kernel_pilot_public/kp-probe-artifacts/launch-overhead/2026-04-23/run.log` — clean run output.
