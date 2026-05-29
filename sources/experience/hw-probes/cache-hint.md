---
api: __ldg/__ldca/__ldcg/__ldcs/__ldcv (plus default)
namespace: cuda-runtime
probe_slug: cache-hint
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
  code: sources/experience/hw-probes/cache-hint/artifacts/cache_hint_probe.cu
  build: sources/experience/hw-probes/cache-hint/artifacts/build.sh
  introspection: sources/experience/hw-probes/cache-hint/artifacts/device.json
  profile: ''
  ncu_report_host_path: h200_ncu:/inspire/hdd/project/qianghuaxuexi/public/kernel_pilot_public/kp-probe-artifacts/cache-hint/2026-04-23/cache_hint.ncu-rep
  ncu_txt_host_path: h200_ncu:/inspire/hdd/project/qianghuaxuexi/public/kernel_pilot_public/kp-probe-artifacts/cache-hint/2026-04-23/ncu.txt
  ncu_csv_host_path: h200_ncu:/inspire/hdd/project/qianghuaxuexi/public/kernel_pilot_public/kp-probe-artifacts/cache-hint/2026-04-23/ncu_metrics.csv
  run_log_host_path: h200_ncu:/inspire/hdd/project/qianghuaxuexi/public/kernel_pilot_public/kp-probe-artifacts/cache-hint/2026-04-23/run.log
source:
- path: spec
  anchor: Reference
conclusions:
  workload: 'grid-stride read kernel, grid=528 blocks x 256 threads. Two regimes:
    DRAM (256 MiB buffer > 60 MiB L2, single pass) and L2 (8 MiB buffer << L2, 16
    inner passes).'
  dram_default_ms: 0.1949
  dram_default_gbps: 1377.44
  dram_ldg_ms: 0.194
  dram_ldg_gbps: 1383.35
  dram_ldca_ms: 0.1943
  dram_ldca_gbps: 1381.52
  dram_ldcg_ms: 0.1941
  dram_ldcg_gbps: 1382.66
  dram_ldcs_ms: 0.1939
  dram_ldcs_gbps: 1384.26
  dram_ldcv_ms: 0.1939
  dram_ldcv_gbps: 1384.26
  dram_spread_pct: 0.5
  l2_default_ms: 0.0222
  l2_default_gbps: 6043.67
  l2_ldg_ms: 0.0222
  l2_ldg_gbps: 6052.39
  l2_ldca_ms: 0.0222
  l2_ldca_gbps: 6034.97
  l2_ldcg_ms: 0.0503
  l2_ldcg_gbps: 2668.13
  l2_ldcs_ms: 0.0222
  l2_ldcs_gbps: 6043.67
  l2_ldcv_ms: 0.0502
  l2_ldcv_gbps: 2673.23
  l2_ldcg_slowdown_vs_default: 2.265
  l2_ldcv_slowdown_vs_default: 2.262
  l2_ldcs_vs_default: 1.0
  ncu_default_dram_bytes_per_launch: 268455000
  ncu_default_dram_sol_pct: 26.6
  ncu_default_l2_hit_sectors: 0
open_questions:
- clock_policy is `unlocked-logged-only` — H200 ran at max graphics clock (1980 MHz)
  but was not explicitly locked. Absolute GB/s subject to boost-clock variation; ratios
  across (regime, variant) cells are robust.
- NCU --launch-count 12 captured only the first variant's 12 launches (5 warmup +
  7 of 20 timed iters of `default` variant at DRAM regime). To get per-variant NCU
  metrics we would need --launch-count 300 (12 variants x 25 launches each) or to
  wrap each (regime, variant) invocation in a separate binary. Wall-clock GB/s is
  authoritative here; NCU only corroborates that the DRAM-regime default variant generates
  ~268 MB DRAM reads per launch (matches the 256 MiB × 1 pass = 268.4 MB expected).
- '**RESOLVED (PTX/SASS audit 2026-04-23)**: all 6 variants emit distinct PTX (`ld.global.{nc,ca,cg,cs,cv}`)
  AND distinct SASS opcodes (`LDG.E.CONSTANT` / `LDG.E.STRONG.SM` / `LDG.E.STRONG.GPU`
  / `LDG.E.EF` / `LDG.E.STRONG.SYS`). The DRAM-regime 0.5% collapse is ''all hints
  honoured identically'', not ''all hints ignored'' — confirmed by opcode distinctness.
  Note the non-obvious finding: default (`*p` with `const __restrict__`) lowers to
  `ld.global.nc` / `LDG.E.CONSTANT`, identical to `__ldg` — that is why default ≡
  `__ldg` at wall-clock.'
- 'Legacy P6 / Q2 answered: `__ldg` vs default is 0.4% on H200 DRAM regime and 0.1%
  on L2 regime — within measurement noise. On sm_90a with `const __restrict__`, explicit
  `__ldg` is unnecessary. Keep explicit `__ldg` only for non-const pointers the compiler
  cannot prove read-only.'
- 'Legacy P1 (staleness from __ldg non-coherent cache): not re-measured. Single-kernel
  harness reads but never writes the buffer, so the coherency hazard never materializes.
  Pitfall retained as legacy-anecdotal.'
- Store hints (`__stcs`, `__stwb`, `__stwt`) are out of scope for this probe — the
  kernel never writes to the hot buffer. Legacy skill's Skill 4 sub-skill retained
  as inferred pending a dedicated store-hint probe (`sources/experience/hw-probes/store-hint/`,
  open).
- '`__ldlu` (last-use) not measured. Its effect is coupled with subsequent kernels''
  L2 pressure; single-kernel microbench cannot reveal benefit. Same regime-gap pattern
  as the `l2-residency` probe. Follow-up: `sources/experience/hw-probes/cache-hint-contended/`
  (open).'
- 'The L2 regime''s `__ldcs` (evict-first) result is *identical* to default. Reason:
  with an 8 MiB buffer in a 60 MiB L2 with no competing traffic, the evict-first tag
  never triggers an actual eviction across the 16 inner passes. A contended regime
  (8 MiB hot + another stream touching 50+ MiB) would expose the degradation. Belongs
  in the same contended follow-up probe.'
id: exp-cache-hint
type: experience
vendor: nvidia
title: 2026 04 23 Cache Hint
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L25083-L25130
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/parallel-thread-execution/cuda_parallel-thread-execution_index.html.md
  anchor: L10400-L10490
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L25083-L25130
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/parallel-thread-execution/cuda_parallel-thread-execution_index.html.md
  anchor: L10400-L10490
---
## Summary

This probe backs the cost model in [wiki/nvidia/foundations/memory/cache-load-hints/skill.md](../../../wiki/nvidia/foundations/memory/cache-load-hints.md) by sweeping six load variants — default (`*p`), `__ldg`, `__ldca`, `__ldcg`, `__ldcs`, `__ldcv` — across two working-set regimes on H200 sm_9.0a. The canonical claim (PG §5.4.8.3, L25083-L25130) is that cache operators let the programmer steer a load into the desired cache level; the legacy skill extrapolated that `__ldcg` (L2-only) saves L1 pressure and `__ldcs` (streaming) reduces cache pollution. Neither source quantifies the effect on Hopper sm_9.0a, and the legacy L3-sandbox work flagged P6 ("`const __restrict__` already emits the non-coherent path; `__ldg` is redundant") as inferred. This probe answers three decision-level questions:

1. **Is `__ldg` still meaningful on H200's unified cache?** No. Measured 0.1-0.4% spread between `default` and `__ldg` across both regimes — within noise. The compiler-auto-emit claim is empirically correct.
2. **Do evict-first hints (`__ldcs`) speed up streaming?** No. In a single-kernel DRAM-bound scan they are identical to default; in an L2-resident reuse pattern they are also identical because the L2 is under-subscribed.
3. **Does L1 bypass (`__ldcg`) hurt when data is L2-resident?** Yes. **2.26× slowdown** at L2 regime because reads route L2→registers without L1 staging; the L2 pipe cannot match the L1 throughput. This is the only measurably bad hint in the probe.

## Setup

- GPU: NVIDIA H200, UUID `GPU-fbaa167f-4646-b8b9-c4f5-a4c4734ebc25`, driver 570.124.06, CUDA runtime 12.9, sm_9.0a. L2 = 60 MiB.
- Build: `nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo`.
- Clock policy: **unlocked**; nvidia-smi reported 1980 MHz throughout.
- Protocol: 5 warmup launches + 20 timed launches per (regime, variant) pair, CUDA events, median ms reported. Buffer initialised once with `1.0f`.
- Kernel: `stream_read<V><<<528, 256>>>(in, N, N_REPEATS, sink)` where `V` is a template parameter selecting the load variant; inner body is a grid-stride read + `acc += load_v<V>(&in[i])`; outer `acc *= 0.999f` prevents hoisting across `N_REPEATS`. Sentinel write to `sink` only when `acc == -1.0f` — prevents DCE without contaminating the load-dominated hot path.
- Regime A: **DRAM** — N = 64 Mi elements (256 MiB buffer, > 60 MiB L2), N_REPEATS = 1.
- Regime B: **L2** — N = 2 Mi elements (8 MiB buffer, fits L2), N_REPEATS = 16.

## Results — median ms / effective GB/s

### DRAM regime (256 MiB buffer, single pass)

| variant   | median_ms | GB/s_eff | vs default |
| --------- | --------: | -------: | ---------: |
| default   |    0.1949 |  1377.44 |      1.000 |
| `__ldg`   |    0.1940 |  1383.35 |      1.004 |
| `__ldca`  |    0.1943 |  1381.52 |      1.003 |
| `__ldcg`  |    0.1941 |  1382.66 |      1.004 |
| `__ldcs`  |    0.1939 |  1384.26 |      1.005 |
| `__ldcv`  |    0.1939 |  1384.26 |      1.005 |

All six variants within **0.5%** — indistinguishable from noise. The kernel is DRAM-bound at ~1.38 TB/s (≈29% of H200 HBM3e peak), and the hint choice has no measurable effect because each sector is touched once and spends negligible time in any cache.

### L2 regime (8 MiB buffer, 16 inner passes)

| variant   | median_ms | GB/s_eff | vs default |
| --------- | --------: | -------: | ---------: |
| default   |    0.0222 |  6043.67 |      1.000 |
| `__ldg`   |    0.0222 |  6052.39 |      1.001 |
| `__ldca`  |    0.0222 |  6034.97 |      0.998 |
| **`__ldcg`**  | **0.0503** | **2668.13** |  **0.441** |
| `__ldcs`  |    0.0222 |  6043.67 |      1.000 |
| **`__ldcv`**  | **0.0502** | **2673.23** |  **0.442** |

Two variants drop to ~44% of default: **`__ldcg`** (cache-global / L2 only / bypass L1) and **`__ldcv`** (cache-volatile / always-fetch). The other four (default, `__ldg`, `__ldca`, `__ldcs`) sit at ~6 TB/s — consistent with L1 staging of a hot buffer that fits L2 comfortably.

Effective GB/s = (WS × N_REPEATS) / median_ms with 1 GB = 10⁹ B.

## Interpretation

### 1. Unified cache collapses `__ldg` onto default

On sm_9.0a the L1/TEX unit is unified; the former texture-cache path that `__ldg` used to route through is now the same physical cache as the default load. With `const __restrict__` parameters the compiler already emits `ld.global.nc` automatically (as PG §5.4.8.3 notes). The measured 0.4% spread in DRAM regime and 0.1% in L2 regime is below noise. Legacy pitfall P6 is **upgraded from inferred to measured** — explicit `__ldg` is redundant on H200 with properly qualified pointers.

### 2. Evict-first (`__ldcs`) is a null when L2 is under-subscribed

Legacy Skill 3's claim was that `.cs` (cache-streaming / evict-first) "marks data as evict-first" to reduce cache pollution. Measured in the L2 regime: `__ldcs` is **identical** to default (6043.67 GB/s vs 6043.67 GB/s — same median ms to 4 decimals). Reason: with an 8 MiB buffer in a 60 MiB L2 under no competing traffic, the evict-first tag never triggers an eviction across the 16 inner passes — there is no pressure to select against. This directly confirms legacy P4's concern ("streaming hint on reused data") as a non-issue in single-kernel microbench and upgrades legacy P7 from inferred to measured ("simple streaming kernels: default path already behaves like `__ldcg`" — actually better: default behaves like `__ldca`).

The interesting case — `__ldcs` in a **contended L2** regime — is not measurable here and is an open follow-up.

### 3. L1 bypass (`__ldcg`) costs 2.26× on L2-resident reuse

This is the probe's one measurably bad hint. `__ldcg` maps to PTX `ld.global.cg` which caches in L2 but **bypasses L1**. When the buffer is L2-resident and re-read 16 times, each inner pass routes L2→registers without L1 staging. L1 BW per SM on H200 is higher than L2 BW per SM, so bypassing L1 halves the effective throughput (6043 → 2668 GB/s). The legacy skill's Skill 2 recommends `__ldcg` when "data is not reused by the same SM but may be reused by other SMs" — the probe refines this: **any reuse within the SM** (including the natural multi-pass pattern common in reductions and normalization kernels) makes `__ldcg` the wrong choice on H200. See skill §S3 for the concrete guidance.

### 4. Volatile (`__ldcv`) collapses to the same slowdown as `__ldcg`

`__ldcv` should bypass *all* caches (each load goes to DRAM), so the expected slowdown is even larger than `__ldcg`. Measured: **same ~44% of default** (2673 vs 2668 GB/s). Hypothesis: H200's L2 is unbypassable in the load path — the volatile modifier causes the L2 tag comparison to always miss, but the physical DRAM fetch still streams through the L2 cache on its way to registers. The wall-clock cost ends up the same as L1-bypass. A PTX-level inspection would confirm whether `ld.global.cv` emits a distinct instruction from `ld.global.cg` on sm_90a; not in scope.

### 5. DRAM regime: all variants flat at ~1.38 TB/s (~29% HBM SOL)

The kernel reads a 256 MiB buffer in one pass at ~1.38 TB/s. HBM3e peak on H200 is ~4.8 TB/s, so the probe runs at ~29% DRAM SOL — not saturating DRAM. Cause: grid-stride loop with sequential access generates stream-friendly requests, but a single accumulator per thread limits outstanding loads in flight. This is expected for a simple read-sum kernel; cranking up ILP or widening to `float4` would close the gap but is out of scope (those are the `ilp` and `vectorized-access` skills). The DRAM-regime result is still valid for the *variant-comparison* purpose: the comparison is at a fixed load-issue rate, and all six variants hit the same ceiling.

NCU DRAM reads per launch: **268 MB** ≈ 256 MiB, matches the single pass. NCU DRAM SOL: 26.6%, matches the wall-clock-derived 29% (difference from the slightly idle launch-overhead slice).

## Takeaways (fed back into skill + pitfalls)

1. **`__ldg` is dead on H200 sm_9.0a for `const __restrict__` pointers.** Keep explicit `__ldg` only for non-const pointers where the compiler cannot prove immutability. Legacy P6 upgraded to measured.
2. **`__ldcg` hurts on any L2-resident reuse pattern** — 2.26× at 16 passes. Use only when the data is truly non-reused per SM (which is rare in typical cuda-core kernels; the pattern occurs in multi-kernel pipelines where a later kernel consumes the same buffer from a different SM).
3. **`__ldcs` is a null in under-contended L2** — identical to default in the probe's L2 regime. Useful only when there is competing traffic that would otherwise evict the hot buffer. Same regime-gap as the `l2-access-policy` skill.
4. **`__ldcv` is a 2.26× loss with no compensating benefit** for any read pattern. Use only when required for correctness (polling a flag written by another kernel); never as a performance knob.
5. **For one-shot DRAM-streaming**, the hint choice is free. The probe measured < 0.5% spread across all six variants. Time spent picking a hint is wasted — the bottleneck is elsewhere (likely ILP or vector-load width).

## PTX / SASS audit (verified 2026-04-23)

`nvcc -arch=sm_90a -O3 -std=c++17 -ptx cache_hint_probe.cu` produced six distinct `ld.global.*` instructions, one per template specialization. `cuobjdump --dump-sass` on the linked binary confirmed six distinct SASS opcodes as well. Per-variant mapping:

| `V` | C intrinsic | PTX emitted | SASS emitted |
|---:|------------|-------------|--------------|
| 0 | `*p` (default, with `const float* __restrict__`) | `ld.global.nc.f32` | `LDG.E.CONSTANT` |
| 1 | `__ldg(p)` | `ld.global.nc.f32` | `LDG.E.CONSTANT` |
| 2 | `__ldca(p)` | `ld.global.ca.f32` | `LDG.E.STRONG.SM` |
| 3 | `__ldcg(p)` | `ld.global.cg.f32` | `LDG.E.STRONG.GPU` |
| 4 | `__ldcs(p)` | `ld.global.cs.f32` | `LDG.E.EF` |
| 5 | `__ldcv(p)` | `ld.global.cv.f32` | `LDG.E.STRONG.SYS` |

Two findings from this audit:

1. **Default and `__ldg` are literally the same SASS** (`LDG.E.CONSTANT`) when the pointer is `const __restrict__`. This is the hardware-level reason wall-clock was identical to 0.1-0.4%. Earlier skill drafts that implied "default emits `ld.global.ca`" were wrong for this pointer-qualification; the compiler promotes `const __restrict__` loads to the non-coherent read-only path (`.nc`) and the SASS skips the L1 STRONG.SM variant in favour of the CONSTANT cache path.
2. **The 2.26× slowdown of `__ldcg` / `__ldcv` at L2-resident reuse has a clean SASS explanation**: both emit `LDG.E.STRONG.*` variants (GPU-scope and system-scope respectively) that bypass the SM-local L1 staging that `LDG.E.CONSTANT` and `LDG.E.STRONG.SM` use. Not a pipe fallback, not a compiler artifact — a direct hardware-level consequence of the chosen consistency scope.

PTX file committed to `artifacts/cache_hint_probe.ptx`. SASS (~120 KB) kept on host only at `h200_ncu:/inspire/hdd/project/qianghuaxuexi/public/kernel_pilot_public/kp-probe-src/cache-hint/cache_hint.sass`.

## Files

- `artifacts/cache_hint_probe.cu` — 6-variant × 2-regime harness.
- `artifacts/cache_hint_probe.ptx` — per-template PTX (6 entries, ld.global.{nc,ca,cg,cs,cv} verified).
- `artifacts/build.sh` — `nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo`.
- `artifacts/run.sh` — build + run + NCU capture (12-launch budget).
- `artifacts/device.json` — nvidia-smi introspection snapshot (repo).
- `h200_ncu:/inspire/hdd/project/qianghuaxuexi/public/kernel_pilot_public/kp-probe-artifacts/cache-hint/2026-04-23/` — host-only:
  - `cache_hint.ncu-rep` — binary NCU report.
  - `ncu_metrics.csv` — per-launch CSV (12 launches, default variant only).
  - `ncu.txt` — NCU log.
  - `run.log` — clean (non-NCU) output.
  - `cache_hint.sass` — full disassembly (cuobjdump --dump-sass), ~120 KB.
