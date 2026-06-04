---
api: __hfma / __hfma2 (fp16 and bf16) + fmaf
namespace: cuda-runtime
probe_slug: half2-throughput
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
  code: artifacts/experience/hw-probes/half2-throughput/half2_throughput_probe.cu
  build: artifacts/experience/hw-probes/half2-throughput/build.sh
  introspection: artifacts/experience/hw-probes/half2-throughput/device.json
  profile: ''
  ncu_report_host_path: h200_ncu:/inspire/hdd/project/qianghuaxuexi/public/kernel_pilot_public/kp-probe-artifacts/half2-throughput/2026-04-23/half2_throughput.ncu-rep
  ncu_csv_host_path: h200_ncu:/inspire/hdd/project/qianghuaxuexi/public/kernel_pilot_public/kp-probe-artifacts/half2-throughput/2026-04-23/ncu_metrics.csv
  run_log_host_path: h200_ncu:/inspire/hdd/project/qianghuaxuexi/public/kernel_pilot_public/kp-probe-artifacts/half2-throughput/2026-04-23/run.log
source:
- path: spec
  anchor: Reference
conclusions:
  workload: 'fma_bench_<V>: grid=528 blocks x 256 threads = 135168 threads; each thread runs N_CHAINS=4 independent accumulator chains of N_FMA_ITERS=2048 FMAs (8192 FMAs/thread). Compute-bound (NCU fp32 variant: SM SOL = 78.4%).'
  fp32_scalar_ms: 0.0773
  fp32_scalar_gflops: 28633.0
  fp16_scalar_ms: 0.0488
  fp16_scalar_gflops: 45351.3
  fp16_packed_ms: 0.0842
  fp16_packed_gflops: 52608.1
  bf16_scalar_ms: 0.0488
  bf16_scalar_gflops: 45381.0
  bf16_packed_ms: 0.0953
  bf16_packed_gflops: 46478.2
  fp16_scalar_vs_fp32: 1.584
  fp16_packed_vs_fp32: 1.837
  bf16_scalar_vs_fp32: 1.585
  bf16_packed_vs_fp32: 1.623
  fp16_packed_vs_fp16_scalar: 1.16
  bf16_packed_vs_fp16_packed: 0.884
  h200_peak_fp32_tflops: 67
  compute_sol_fp32_pct: 78.4
open_questions:
- clock_policy is `unlocked-logged-only` — H200 ran at max graphics clock (1980 MHz) but was not explicitly locked. Absolute GFLOPS subject to boost-clock variation; ratios across variants are robust (same launch context).
- 'Legacy claim: `__hfma2` gives 2x throughput over `__hfma`. Measured on H200 sm_9.0a: only **1.16x** (52608 vs 45351 GFLOPS at scalar op count). The source of the gap is not confirmed by this probe; hypotheses: (a) scalar `__hfma` already emits the FP16 SIMD pipe on sm_9.0a, so the second packing axis does not add 2x; (b) `hfma2` has higher latency per instruction than scalar `hfma` so the benefit partially cancels; (c) register pressure for h2 is 2x the scalar form. A PTX / SASS inspection (nvcc -Xptxas=-v + nvcc -ptx) would distinguish (a) from (b). Out of scope for this probe.'
- bf16 packed (46478 GFLOPS) is **12% slower** than fp16 packed (52608 GFLOPS) on H200 — legacy 'equal throughput' claim contradicted. bf16 scalar equals fp16 scalar (both 45 TFLOPS), so the packed-pipeline path differs between fp16x2 and bf16x2 on sm_9.0a. Possibly a result of the bf16 packed FMA using the same FP32 issue slot (since bf16 is essentially a truncated fp32) rather than the dedicated fp16x2 pipe. Not confirmed; a SASS inspection would show which pipe each variant dispatches to.
- Only fp32 variant's NCU metrics were captured (--launch-count 1). Re-running with --launch-count 5 or structuring the harness so each variant runs in a single binary would give per-variant Compute SOL / instruction mix. Wall-clock remains authoritative.
- Transcendentals (hexp, h2exp, hlog, h2log, hrsqrt, h2rsqrt, tanh.approx.f16/bf16) NOT measured in this probe. Legacy pitfall P10 flags that `h2exp` may decompose into fp32 ops on some architectures; H200 coverage is an open follow-up (`sources/experience/hw-probes/half-transcendental/`, open).
- __hfma2_relu (fused FMA+relu) NOT measured. Legacy Skill 5 / pitfall P11 claim 4% instruction reduction but no wall-clock benefit in memory-bound kernels. Confirm/refute on H200 in a follow-up (`sources/experience/hw-probes/half-fma-relu/`, open).
- Atomic fp16/bf16 add NOT measured. Legacy pitfall P12 claims native fp16 atomicAdd has worse contention than fp32 atomicAdd. half-precision-math skill retains this as inferred; see `sources/experience/hw-probes/atomic-reduction-contention/` (smem-tile-reuse / earlier probe measured fp32 only).
id: exp-half2-throughput
type: experience
vendor: nvidia
title: 2026 04 23 Half2 Throughput
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1370-L1440
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L25350-L25420
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1370-L1440
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L25350-L25420
architectures:
- sm90
- sm90a
languages:
- ptx
- cuda-cpp
techniques:
- pipeline-stages
- vectorized-loads
- register-budgeting
- data-reuse
- kernel-fusion
- shared-memory-optimization
kernel_types:
- gemm
- fused-kernel
confidence: experimental
tags:
- pipeline-stages
- vectorized-loads
- register-budgeting
- data-reuse
- kernel-fusion
- shared-memory-optimization
- gemm
- fused-kernel
- ptx
- cuda-cpp
artifact_dir: artifacts/experience/hw-probes/half2-throughput
---
## Summary

This probe backs the compute-throughput claims in [wiki/nvidia/foundations/compute/half-precision-math/skill.md](../../../wiki/nvidia/foundations/compute/half-precision-math.md) by sweeping five FMA variants — fp32 scalar, fp16 scalar, fp16 packed (`__hfma2`), bf16 scalar, bf16 packed — in a compute-bound 4-chain ILP harness on H200 sm_9.0a. The legacy skill makes two canonical claims worth re-measuring on H200:

1. **"`__hfma2` gives 2× throughput over `__hfma`"** (Skill 1). Measured on H200: only **1.16×**. The "2× from packing" framing is largely false on sm_9.0a — scalar `__hfma` already runs faster than scalar `fmaf` (1.58× measured), and the additional packing only gains 16% more.
2. **"bf16 packed has equal throughput to fp16 packed"** (Skill 2). Measured: **bf16 packed is 12% slower than fp16 packed** (46.5 vs 52.6 TFLOPS scalar-equivalent). Scalar bf16 and fp16 are identical at ~45 TFLOPS, so the fp16x2 and bf16x2 packed pipelines differ on H200.

Both measurements refine the legacy expectations and feed directly into the skill's decision rules ("when is packing worth the extra register pressure?") and the precision-impact annotation users get in the pattern-level INDEX.md.

## Setup

- GPU: NVIDIA H200, UUID `GPU-fbaa167f-4646-b8b9-c4f5-a4c4734ebc25`, driver 570.124.06, CUDA runtime 12.9, sm_9.0a.
- Build: `nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo`.
- Clock policy: **unlocked**; nvidia-smi reported 1980 MHz throughout.
- Protocol: 5 warmup launches + 20 timed launches per variant, CUDA events, median ms reported.
- Kernel shape: `fma_bench_<V><<<528, 256>>>(out, N)`; `N = 528 × 256 = 135168` threads; each thread runs `N_CHAINS=4` independent chains of `N_FMA_ITERS=2048` FMAs. Outputs are summed back into one value per thread and written — this prevents DCE and keeps the FMAs necessary.
- Five variants:
  - `fp32-scalar`: `float`, `fmaf`.
  - `fp16-scalar`: `__half`, `__hfma`.
  - `fp16-packed`: `__half2`, `__hfma2` (2 FP ops per instruction).
  - `bf16-scalar`: `__nv_bfloat16`, `__hfma`.
  - `bf16-packed`: `__nv_bfloat162`, `__hfma2` (2 FP ops per instruction).
- "Scalar-equivalent GFLOPS" counts SCALAR FP ops (so packed variants receive ×2 credit for their dual-lane instructions).

## Results

| variant       | median_ms | scalar GFLOPS | vs fp32 | vs own-scalar |
|---------------|----------:|--------------:|--------:|--------------:|
| fp32-scalar   |    0.0773 |         28633 |   1.00× |          —    |
| fp16-scalar   |    0.0488 |         45351 | **1.58×** |          —    |
| fp16-packed   |    0.0842 |         52608 | **1.84×** |    **1.16×** (vs fp16-scalar) |
| bf16-scalar   |    0.0488 |         45381 |   1.59× |          —    |
| bf16-packed   |    0.0953 |         46478 |   1.62× |    **1.02×** (vs bf16-scalar) |

H200 datasheet peak fp32 FMA = 67 TFLOPS. fp32 scalar variant hit **28.6 TFLOPS = 43% of peak**; NCU Compute SM SOL = 78% — the probe is compute-bound on the FMA issue slot but not saturating the FP register-file bandwidth. The variant-to-variant ratios are the measurement of interest, not the absolute peak.

## Interpretation (wall-clock + audited PTX/SASS)

Per-variant instruction mapping from `nvcc -arch=sm_90a -O3 -ptx` + `cuobjdump --dump-sass` (2026-04-23):

| Kernel template | PTX emitted | SASS emitted |
|-----------------|-------------|--------------|
| `fma_bench_fp32`        | `fma.rn.f32`    | `FFMA` |
| `fma_bench_fp16_scalar` | `fma.rn.f16`    | `HFMA2.MMA` |
| `fma_bench_fp16_packed` | `fma.rn.f16x2`  | `HFMA2.MMA` |
| `fma_bench_bf16_scalar` | `fma.rn.bf16`   | `HFMA2.MMA.BF16_V2` (+ a couple `HFMA2.MMA`) |
| `fma_bench_bf16_packed` | `fma.rn.bf16x2` | `HFMA2.MMA.BF16_V2` + `HFMA2.BF16_V2` |

### 1. fp16 scalar ≈ fp16 packed at SASS — the compiler already packs

`fma_bench_fp16_scalar` and `fma_bench_fp16_packed` emit the **same `HFMA2.MMA` SASS opcode**. nvcc auto-packs the 4 independent scalar `__hfma` chains into HFMA2.MMA instructions given the ILP harness. "Scalar vs packed" at C-source is largely erased at SASS. The measured 1.16× packed-over-scalar residual (52608 / 45351 GFLOPS) is register-layout / alignment overhead, not the packing gain itself. The **real packing gain is fp32 → fp16 scalar at 1.58×** (FFMA → HFMA2.MMA); the explicit `__hfma2` path adds only 16% on top.

Consequence: stop framing `__hfma2` as a 2× throughput knob. The 1.84× fp32-scalar → fp16-packed total is mostly available from just switching to fp16 scalar under `const __restrict__` with ILP.

### 2. bf16 packed is 12% slower than fp16 packed — distinct HFMA2 variant

`fma_bench_bf16_packed` emits `HFMA2.MMA.BF16_V2` / `HFMA2.BF16_V2`, **distinct from** fp16's `HFMA2.MMA` and **not** a fall-through to `FFMA`. Both variants stay within the half-precision HFMA2 family; the BF16_V2 form simply has lower measured throughput (46478 vs 52608 GFLOPS = 0.88×) on sm_9.0a hardware. Root cause (multiplier width, pipeline latency, register port allocation) is not resolvable from SASS alone and would need an isolated-instruction microbench — out of scope for this probe.

Practical consequence: prefer fp16 packed when dynamic range permits (values in fp16's [6.1e-5, 6.55e4]). bf16's 12% cost is the price of the wider dynamic range; accept it for training / unnormalised activations.

### 3. fp16 scalar = bf16 scalar at 45 TFLOPS — choice is precision/range

Both scalar paths get auto-packed into HFMA2-family instructions (fp16 scalar → `HFMA2.MMA`, bf16 scalar → `HFMA2.MMA.BF16_V2`). Wall-clock is identical at 45.4 / 45.4 TFLOPS. There is no throughput tie-breaker at scalar level — the choice is purely the precision vs range trade-off documented in §Precision of the skill.

### 4. bf16 scalar ≈ bf16 packed — explicit packing is a near-null for bf16

bf16 scalar (45381 GFLOPS) and bf16 packed (46478 GFLOPS) differ by only **2.4%** — far less than fp16's 16% scalar-vs-packed gap, and close to the probe's noise floor under unlocked clocks. Three contributing factors, in decreasing order of confidence:

1. **Compiler already auto-packs the scalar bf16 path** — `fma_bench_bf16_scalar`'s inner loop emits `HFMA2.MMA.BF16_V2` at SASS (identical to the packed template's inner loop). The 4-chain ILP gives nvcc pairs of independent scalar FMAs and it opportunistically fuses them into the packed SASS form. Explicit `__hfma2` on `__nv_bfloat162` doesn't unlock a qualitatively new opcode — it just guarantees the packing decision.
2. **The BF16_V2 HFMA2 pipe is narrower** — the *same* compiler auto-packing lifted fp16 scalar from 45 → 52.6 TFLOPS (+16%) in the explicit packed variant, but bf16's ceiling stays at ~46 TFLOPS regardless. The hardware pipeline for `HFMA2.MMA.BF16_V2` appears to be throughput-bound at a lower point than `HFMA2.MMA` (fp16). Whatever register-layout / alignment residual makes fp16 explicit packing win 16% on fp16, that residual gets absorbed below the BF16_V2 pipe's saturation point — so explicit packing for bf16 cannot extract the same headroom.
3. **2.4% is within measurement noise** under `clock_policy: unlocked-logged-only`. H200 boost-clock variation across launches can drift 1-2% at sustained load, and the probe's warm-up budget (5 launches) is not enough to null this out. Strictly speaking, the scalar-vs-packed bf16 delta might not even be statistically significant — a locked-clock re-run would be needed to bisect "small real gain" from "just noise".

**Practical implication**: for bf16 compute on H200, writing `__nv_bfloat162` + `__hfma2` is code complexity that buys essentially nothing — the compiler already does the packing when ILP permits. Prefer scalar `__nv_bfloat16` + `__hfma` for readability and keep the 4-chain (or wider) ILP structure so the compiler has pairs to fuse. This is different from fp16, where explicit packing still buys a real 16%. Put another way: **the bf16 `__hfma2` intrinsic is a correctness-preservation tool for bf16x2 *storage* layouts, not a performance knob**.

### 5. Probe runs at 78% Compute SOL (register pressure bound)

NCU reports fp32 variant's Compute (SM) Throughput at 78.4%. The probe is compute-bound (4 independent accumulator chains × 2048 iters, no memory traffic in the inner loop) but the SM issue slot is not fully saturated — register pressure from the 4-chain ILP + accumulator spills at the inner-loop boundary. Measured ratios across variants are valid at this 78% operating point; absolute peaks would be higher if saturated (wider ILP, see `ilp` skill).

## Takeaways (fed back into skill + pitfalls)

1. **fp16 scalar is already 1.58× fp32 scalar** at `HFMA2.MMA` SASS thanks to compiler auto-packing. Explicit `__hfma2` adds only 16% on top (total 1.84×). Legacy "2× from packing" is false on H200.
2. **fp16 packed beats bf16 packed by 12%** (`HFMA2.MMA` vs `HFMA2.MMA.BF16_V2`). Prefer fp16 when dynamic range permits; accept the 12% for bf16 when range forces it.
3. **fp16 scalar = bf16 scalar at 45 TFLOPS.** Format choice at scalar level is pure precision vs range; no throughput tie-breaker.
4. **bf16 scalar ≈ bf16 packed (2.4%, near noise).** Explicit `__hfma2` on `__nv_bfloat162` buys essentially nothing — the compiler already packs scalar bf16 into `HFMA2.MMA.BF16_V2`, and the BF16_V2 pipe's lower saturation ceiling hides whatever alignment residual makes fp16 explicit packing worth 16%. Prefer scalar `__nv_bfloat16` + `__hfma` for bf16 compute.
5. **fp32 scalar is 28.6 TFLOPS at 78% SM SOL.** The half-precision advantage (1.58×–1.84× vs fp32) is real but smaller than "half precision is 2× fp32" folklore.
6. **Legacy P10 (h2exp decomposition), P11 (hfma_relu benefit), P12 (fp16 atomic contention)** are open — this probe measured only the FMA inner loop.

PTX committed to `artifacts/half2_throughput_probe.ptx`. SASS (~50 KB) kept on host at `h200_ncu:/inspire/hdd/project/qianghuaxuexi/public/kernel_pilot_public/kp-probe-src/half2-throughput/half2.sass`.

## Files

- `artifacts/half2_throughput_probe.cu` — 5-variant FMA throughput harness.
- `artifacts/half2_throughput_probe.ptx` — per-kernel PTX (5 entries, `fma.rn.{f32,f16,f16x2,bf16,bf16x2}` verified).
- `artifacts/build.sh` — `nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo`.
- `artifacts/run.sh` — build + run + NCU capture (1 launch per variant).
- `artifacts/device.json` — nvidia-smi introspection snapshot (repo).
- `h200_ncu:/inspire/hdd/project/qianghuaxuexi/public/kernel_pilot_public/kp-probe-artifacts/half2-throughput/2026-04-23/` — host-only:
  - `half2_throughput.ncu-rep`, `ncu_metrics.csv`, `ncu.txt`, `run.log`.
  - `half2.sass` — full disassembly (cuobjdump --dump-sass), kept on host at `h200_ncu:/inspire/hdd/project/qianghuaxuexi/public/kernel_pilot_public/kp-probe-src/half2-throughput/`.
