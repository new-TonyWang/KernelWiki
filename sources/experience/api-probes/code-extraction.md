---
api: meta-skill — `wiki/nvidia/foundations/meta/code-extraction/skill.md` applied to `cutlass@f74fea9c`
namespace: meta
probe_slug: cutlass-meta-extraction-walkthrough
status: partial
kind: api-end-to-end
trigger: meta-skill walkthrough applied to cutlass end-to-end
evidence_level: measured
clock_policy: as-applied
measured_on: developer workstation + H200-SXM | sm_90a | cuda 12.9.86
verdict: partial
source:
- path: wiki/nvidia/foundations/meta/code-extraction/skill.md
  anchor: the procedure under test
artifacts:
  code: wiki/nvidia/foundations/meta/code-extraction/skill.md
  build: sources/experience/api-probes/code-extraction/2026-04-28-cutlass-walkthrough.md
  introspection: wiki/nvidia/foundations/meta/code-extraction/skill.md
  profile: sources/experience/api-probes/gemm/artifacts/profiles/2026-04-28-gemm-aligned.csv
upstream_repo: cutlass@f74fea9c
id: exp-code-extraction
type: experience
vendor: nvidia
title: 2026 04 28 Cutlass Walkthrough
---
# Probe — Meta-skill walkthrough applied to cutlass end-to-end

**Goal**: execute the meta-skill procedure (`wiki/nvidia/foundations/meta/code-extraction/skill.md`) against cutlass `f74fea9c` and record the exact commands + evidence paths produced. The output of this probe is what an agent following the meta-skill procedure should produce; the per-topic landings (TMA, wgmma, aligned GEMM, non-aligned tail, warp-specialization, persistent-kernel, fused GEMM) are the verifiable downstream artifacts.

## Method

The 6-step procedure from the meta-skill was executed in order against `cutlass@f74fea9c` on a developer workstation, with H200-SXM (sm_90a, cuda 12.9.86) used for the on-hardware verification of every step's downstream skill. Each step's exact commands + outputs are recorded below.

## Step 1: Repository orientation

```bash
cd {{CUTLASS_REPO_REF}}
git log --oneline -1
# 9f74fea9c <commit message>

# Locate kernel sources:
find . -name '*.cu' -o -name '*.cuh' -o -name '*.hpp' | head -20
# revealed: include/cutlass/, examples/, tools/util/include/

# Identify build system:
ls CMakeLists.txt examples/48_hopper_warp_specialized_gemm/CMakeLists.txt
# CMake; example 48 has its own CMake target.

# Confirm one example builds + runs on the target hardware:
ssh h200_ncu '
  rsync -az --exclude=".git" --exclude="build" \
    {{CUTLASS_REPO_REF}}/ \
    h200_ncu:{{CUTLASS_REPO_REF}}/
'
ssh h200_ncu 'cd {{CUTLASS_REPO_REF}} && mkdir -p build_ex48 && cd build_ex48 && \
  nvcc -std=c++17 -O3 -arch=sm_90a -lineinfo -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 \
       --expt-relaxed-constexpr --expt-extended-lambda \
       -I../include -I../tools/util/include \
       ../examples/48_hopper_warp_specialized_gemm/48_hopper_warp_specialized_gemm.cu \
       -o ex48_hopper_ws_gemm && ./ex48_hopper_ws_gemm'
# -> Disposition: Passed; Avg runtime ~0.01 ms; cutlass at f74fea9c builds + runs on H200.
```

**Output**: cutlass tree confirmed at `f74fea9c`, build pipeline confirmed, sm_90a target confirmed via single example. Pre-condition for steps 2-4 is met.

## Step 2: Hardware-feature scan

For each hardware-feature pattern in the meta-skill's table, `grep`-located in cutlass and landed:

```bash
# TMA primitive scan:
rg -l 'cp\.async\.bulk\.tensor|cuTensorMapEncodeTiled|SM90_TMA_LOAD' include/
# -> include/cute/arch/copy_sm90_tma.hpp, include/cute/atom/copy_traits_sm90_tma.hpp,
#    include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp, ...
# Landed at: wiki/nvidia/hardware/tma/skill.md + sources/experience/api-probes/gemm/2026-04-28-tma-bandwidth-counters.md

# wgmma atom scan:
rg -l 'wgmma\.mma_async|MMA_64xNxK|cute::SM90::GMMA::MMA_' include/
# -> include/cute/arch/mma_sm90_gmma.hpp, include/cute/atom/mma_traits_sm90_gmma.hpp, ...
# Landed at: wiki/nvidia/hardware/wgmma/skill.md + sources/experience/api-probes/gemm/2026-04-28-wgmma-counters.md
# Plus atom-shape sweep: sources/experience/api-probes/gemm/2026-04-28-wgmma-atom-shape-sweep.md
```

**Output**: Two hardware-feature skills landed (TMA, wgmma) with hw-probes capturing measured TMA bandwidth (5.37 GB/launch at the 5120×4096×4096 anchor) and wgmma instruction counts. Atom-shape sweep covers `MMA_64x64x8`, `MMA_64x128x8`, `MMA_64x256x8` with measured TFLOPS.

## Step 3: Algorithm-extraction scan

For each end-to-end optimization pattern, `grep`-located + landed:

```bash
# Warp-specialization scan:
rg -l 'KernelTmaWarpSpecialized|MainloopSm90TmaGmmaWarpSpecialized' include/ examples/
# -> include/cutlass/gemm/dispatch_policy.hpp, include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized*.hpp,
#    examples/49_hopper_gemm_with_collective_builder/49_collective_builder.cu (multi-schedule sweep)
# Landed at: wiki/nvidia/techniques/warp-specialization/skill.md + sources/experience/api-probes/gemm/2026-04-28-warp-specialization-ablation.md

# Persistent-kernel scan:
rg -l 'PersistentScheduler|StreamKScheduler|TileSchedulerType' include/
# -> include/cutlass/gemm/kernel/tile_scheduler.hpp,
#    include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp,
#    include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp
# Landed at: wiki/nvidia/techniques/persistent-kernel/skill.md + sources/experience/api-probes/gemm/2026-04-28-persistent-kernel-ablation.md
```

**Output**: Two classical-algo skills landed (warp-specialization, persistent-kernel) with measured A/B at 2048³: TMA-only (warp-spec OFF) 178.9 TFLOPS, plain WS 195.4, pingpong 193.8, cooperative 186.9 — all bit-identical to cuBLAS, +9.2% from enabling warp-specialization.

## Step 4: Operator-skill scan

For each operator-level pattern, `grep`-located + landed:

```bash
# Aligned GEMM scan: (use TMA + wgmma primitives directly)
# Landed at: wiki/nvidia/foundations/compute/gemm/aligned/skill.md + sources/experience/api-probes/gemm/2026-04-28-gemm-aligned.md
# Measured at 512^3 / 2048^3 / 8192^3, all bit-identical to cuBLAS.

# Non-aligned tail handling scan:
rg -l 'predicate\|kErrorInvalidProblem\|gemm\.can_implement' include/cutlass/gemm/
# -> include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp (predicate-tail mainloop)
# Landed at: wiki/nvidia/foundations/compute/gemm/non-aligned-tail/skill.md + sources/experience/api-probes/gemm/2026-04-28-gemm-tail.md
# (status: partial — strategy A/B documented as adjacent-shape proxy with confounders)

# Fused GEMM scan:
rg -l 'LinCombEltAct|LinearCombinationRelu|FusionOperation|MainloopMixedDtype' include/
# -> include/cutlass/epilogue/fusion/operations.hpp (LinCombEltAct),
#    include/cutlass/epilogue/thread/linear_combination_relu.h, activation.h,
#    include/cutlass/gemm/collective/...sm90_mixed_input...hpp (mixed-dtype mainloop),
#    examples/{50,55,61}_hopper_*.cu (fusion examples)
# Landed at: wiki/nvidia/foundations/compute/gemm-fused/cutlass-epilogue-prologue/skill.md +
#            sources/experience/api-probes/gemm/2026-04-28-gemm-fused.md (epilogue verified, prologue follow-up)
# Measured at 2048^3: fused 86.83 us vs non-fused 100.21 us = 12.65% savings.
```

**Output**: Three operator skills landed (aligned GEMM, non-aligned-tail, fused GEMM). Aligned fully verified at 3 shapes; non-aligned-tail status: partial (structural-infeasibility argument for the same-shape A/B); fused-GEMM epilogue fusion verified, prologue follow-up.

## Step 5: On-hardware verification

For every skill in steps 2-4, the canonical artifact bundle was built and run on H200 with the direct cuBLAS comparator gate:

```bash
ssh h200_ncu '
  cd {{CUTLASS_REPO_REF}}/build_ac5
  bash build_warpspec.sh   # builds gemm_compare, gemm_compare_tma, gemm_compare_ws, gemm_compare_pingpong
  bash build_fused.sh      # builds gemm_compare_relu, relu_kernel
  bash run.sh              # aligned-shape sweep
  bash run_tail.sh         # non-aligned-tail sweep
  ./gemm_compare_relu --m=2048 --n=2048 --k=2048 --iterations=20  # fused
'
# Each binary's output is captured in the per-topic probe markdown's "Run-log excerpt".
```

**Output verifications by probe**:

| Skill | Probe path | Key measurement |
|---|---|---|
| `wiki/nvidia/hardware/tma/skill.md` | `sources/experience/api-probes/gemm/2026-04-28-tma-bandwidth-counters.md` | TMA load bw 5.37 GB at 5120×4096×4096 |
| `wiki/nvidia/hardware/wgmma/skill.md` | `sources/experience/api-probes/gemm/2026-04-28-wgmma-counters.md` | wgmma instruction count via ncu `smsp__inst_executed_pipe_tensor_op_hmma_cycles_active.sum` |
| `wiki/nvidia/hardware/wgmma/skill.md` (atom-shape sweep) | `sources/experience/api-probes/gemm/2026-04-28-wgmma-atom-shape-sweep.md` | M64xN{64,128,256}xK8 throughput sweep |
| `wiki/nvidia/foundations/compute/gemm/aligned/skill.md` | `sources/experience/api-probes/gemm/2026-04-28-gemm-aligned.md` | 512³ 21.7, 2048³ 188.7, 8192³ 292.7 TFLOPS, all max_abs=0 vs cuBLAS |
| `wiki/nvidia/foundations/compute/gemm/non-aligned-tail/skill.md` | `sources/experience/api-probes/gemm/2026-04-28-gemm-tail.md` | 80³ 0.185, 200³ 1.85, 1440³ 130.2 TFLOPS, all max_abs=0 vs cuBLAS |
| `wiki/nvidia/techniques/warp-specialization/skill.md` | `sources/experience/api-probes/gemm/2026-04-28-warp-specialization-ablation.md` | TMA-only 178.9, plain WS 195.4 → +9.2% from warp-spec |
| `wiki/nvidia/techniques/persistent-kernel/skill.md` | `sources/experience/api-probes/gemm/2026-04-28-persistent-kernel-ablation.md` | pingpong 193.8 vs cooperative 186.9 vs plain WS 195.4 TFLOPS at 2048³ |
| `wiki/nvidia/foundations/compute/gemm-fused/cutlass-epilogue-prologue/skill.md` (status: partial) | `sources/experience/api-probes/gemm/2026-04-28-gemm-fused.md` (epilogue path independently verified: fused 86.83 vs non-fused 100.21 μs = ~13% savings; bit-identical to the fused-vs-fused-reference oracle) | Prologue path (cutlass example 55 mixed-dtype int4 × bf16) is **not yet implemented**, so the skill stays partial |

## Step 6: Cross-reference

The meta-skill's "Cross-references" section links into all per-topic probe paths. Each downstream skill's "Cross-references" section in turn links back to siblings (e.g. the non-aligned-tail skill links to aligned, warp-specialization, etc.). Bidirectional cross-references are present in every landed skill.

## Verdict

**verified for the procedure itself.** The meta-skill procedure has been applied end-to-end against cutlass `f74fea9c` on H200-SXM. Every step produced the artifacts the meta-skill predicted (hw-probe → classical-algo skill → operator skill → on-hardware verification). The procedure is repeatable: the rg/build/run commands above can be re-executed by another agent on the same hardware.

**Downstream skill statuses are NOT all "verified" — the meta-skill produces the right shape of artifacts, but each downstream skill's promotion to `verified` depends on its own on-hardware outcome**:

| Downstream skill | Status | Note |
|---|---|---|
| TMA | verified (measured hw-probe at production scale) | TMA bandwidth measurement |
| wgmma | verified (measured hw-probe) | wgmma instruction count via ncu |
| wgmma atom-shape sweep | verified (measured atom-shape sweep) | M64xN{64,128,256}xK8 |
| Aligned GEMM | verified (measured at 3 shapes vs cuBLAS) | 512³, 2048³, 8192³ |
| Non-aligned-tail | **partial** | strategy A/B is documented adjacent-shape proxy with confounders; structural-infeasibility argument for same-shape on/off |
| Warp-specialization | verified (measured ON/OFF A/B) | TMA-only 178.9 → plain WS 195.4 = +9.2% |
| Persistent-kernel | verified (measured pingpong vs cooperative vs plain WS) | shape-dependent crossover |
| Fused GEMM | **partial** | skill is partial because the prologue half (mixed-dtype int4 × bf16, cutlass example 55) is not yet implemented. The epilogue **probe** (`2026-04-28-gemm-fused.md`) is independently verified with a real fused-vs-fused-reference correctness gate at 2048³ on H200. |

The meta-skill **does not** grant `verified` status to downstream skills — that requires each skill's own on-hardware verification. The walkthrough's claim is "the procedure executes correctly", not "every produced skill is verified".

## Known caveats

- The meta-skill does **not** grant `verified` status to its downstream skills. Each skill's promotion requires its own on-hardware verification (canonical-bundle reproduction + numeric-correctness gate); following the meta-skill is necessary but not sufficient.
- The non-aligned-tail same-shape A/B is structurally infeasible for this kernel family; the meta-skill's pitfalls.md lists this as Pitfall #2.
- The PTX-down track (cutlass-free TMA / wgmma / GEMM) is explicitly out of scope for this meta-skill (it covers the cutlass-API extraction path only). The PTX-down workflow re-uses the algorithms identified here as references but writes new code without `cutlass::` / `cute::` includes.
- Compiler/hardware version: cuda 12.9.86 + driver 570.124.06 + sm_90a. Behavior may differ at older toolchains (e.g. cuda 12.4) or other Hopper sub-archs (sm_90 without -a, sm_90a with -a). The meta-skill's procedure is version-agnostic; the specific TFLOPS numbers are not.
