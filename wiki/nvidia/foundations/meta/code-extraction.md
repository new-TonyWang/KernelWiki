---
title: 'Agent meta-skill: CUDA code crawling + operator-optimization-algorithm identification'
status: partial
evidence_level: measured
applies_to_pattern_class:
- cuda-core
- tensor-core
- tensor-core/gemm
- fused
applies_to_ops:
- meta
requires_sm: '>=9.0'
single_kernel_useful: true
cuda_version_tested: 12.9.86
driver_version_tested: 570.124.06
toolchain: nvcc 12.9 + ptxas 12.9 + ripgrep 14.x
measured_on: developer workstation + H200-SXM | sm_90a | cuda 12.9.86 | driver 570.124.06
source:
- path: spec
  anchor: Reference
related_skills:
- tma
- wgmma
- gemm-aligned
- non-aligned-tail
- warp-specialization
- persistent-kernel
- gemm-fused
artifacts:
  code: wiki/nvidia/foundations/meta/code-extraction/skill.md
  build: sources/experience/api-probes/code-extraction/2026-04-28-cutlass-walkthrough.md
  introspection: sources/experience/api-probes/code-extraction/2026-04-28-cutlass-walkthrough.md
  ablation: sources/experience/api-probes/code-extraction/2026-04-28-cutlass-walkthrough.md
id: skill-code-extraction
type: skill
vendor: nvidia
tags:
- cuda-cpp
applies_to:
- general
source_refs:
- source_id: blogs/colfax
  path: developing-cuda-kernels-for-gemm-on-nvidia-hopper-architecture-using-cutlass
  anchor: Hopper warp-specialized GEMM walkthrough — the canonical worked-example
    target
- source_id: blogs/colfax
  path: cutlass-tutorial-mastering-the-nvidia-tensor-memory-accelerator-tma
  anchor: TMA walkthrough
- source_id: blogs/colfax
  path: cutlass-tutorial-fast-matrix-multiplication-with-wgmma-on-nvidia-hopper-gpus
  anchor: wgmma walkthrough
- source_id: blogs/colfax
  path: cutlass-tutorial-persistent-kernels-and-stream-k
  anchor: persistent + stream-K walkthrough
- source_id: blogs/colfax
  path: epilogue-fusion-in-cutlass-with-epilogue-visitor-trees
  anchor: epilogue fusion walkthrough
---
# Agent meta-skill: CUDA code crawling + operator-optimization-algorithm identification

## What it is

A structured procedure for an agent (or human) to land in a new GPU-kernel codebase, identify the candidate optimization algorithms encoded in it, and back-fill them into the layered structure of this KB. The procedure is operator- and library-agnostic; the worked example below threads the procedure through cutlass end-to-end.

## When to use it

- A new GPU-kernel repository (cutlass, FlashAttention, Flash-DMattn, vLLM, cccl, …) needs to be onboarded and its optimizations distilled into the KB.
- A target operator class (GEMM, attention, top-K, GroupedGEMM, …) needs an evidence-backed skill set landed.
- A mature subsystem in the agent's existing KB needs a "what optimizations did we miss" audit against an upstream reference.

## When NOT to use it

- The repository is not GPU-kernel code (e.g. pure host-side scheduling). The procedure presupposes kernel-level optimization patterns.
- The target operator is not represented in the KB's `wiki/nvidia/foundations/compute/<op>/` taxonomy. Add the taxonomy entry first; this meta-skill assumes a target landing slot exists.

## Procedure

### Step 1: Repository orientation

1. Clone the repository and run `git log --oneline -10` to identify the head commit.
2. Locate the kernel sources: `find . -name '*.cu' -o -name '*.cuh' -o -name '*.hpp' | head -100` typically reveals the kernel directories. For cutlass: `include/cutlass/`, `examples/`. For Flash: `csrc/`. For cccl: `cub/`, `thrust/`.
3. Identify the build system (CMake, Bazel, custom). Confirm one example kernel can be built and run on the target hardware before continuing.

### Step 2: Hardware-feature scan

Before extracting algorithms, identify which hardware primitives the kernels use. These map onto the `wiki/nvidia/hardware/` layer of the KB:

| Pattern in code | Hardware feature | KB landing |
|---|---|---|
| `cp.async.bulk.tensor.*`, `cuTensorMapEncodeTiled`, `SM90_TMA_LOAD*` | TMA | `wiki/nvidia/hardware/tma/` |
| `wgmma.mma_async.*`, `MMA_64xNxK_*_SS_TN`, `MainloopSm90Tma*` | wgmma | `wiki/nvidia/hardware/wgmma/` |
| `mbarrier.init.shared`, `cute::Pipeline*` | mbarrier | `wiki/nvidia/hardware/mbarrier/` (when added) |
| `cluster.sync`, `__cluster_dim_is_specified`, `ClusterShape <...>` | cluster-multicast | `wiki/nvidia/hardware/cluster/` (when added) |
| `cp.async.commit_group`, `cp.async.wait_group` | Ampere cp.async | `wiki/nvidia/hardware/cp.async/` (when added) |
| `ldmatrix.sync.aligned.m8n8.x4.shared` | ldmatrix | `wiki/nvidia/hardware/ldmatrix/` (when added) |

For each primitive found, run a hw-probe (canonical artifact-bundle layout: `<probe>.cu` + `helper.h` + `build.sh` + `run.sh` + `device.json` + `profiles/*.csv`) and land at `sources/experience/hw-probes/<feature>/`. TMA and wgmma are the worked examples in this KB.

### Step 3: Algorithm-extraction scan

Search for end-to-end optimization patterns. These map onto the `wiki/nvidia/techniques/` layer:

| Pattern in code | Algorithm | KB landing |
|---|---|---|
| `KernelTmaWarpSpecialized*` dispatch policy + producer/consumer warpgroups + mbarrier-pipelined TMA→wgmma | warp-specialization | `wiki/nvidia/techniques/warp-specialization/` |
| `PersistentScheduler` + SM-count grid + tile-loop-inside-kernel | persistent-kernel | `wiki/nvidia/techniques/persistent-kernel/` |
| `StreamKScheduler` + work-stealing tile assignment | stream-K | `wiki/nvidia/techniques/stream-k/` (when added) |
| Online running statistics for normalization (Welford, log-sum-exp) | online-softmax / online-norm | `wiki/nvidia/techniques/online-softmax/` (when added) |
| Output-tile decomposition with cross-CTA atomics | split-K | `wiki/nvidia/techniques/split-k/` (when added) |

For each algorithm: extract a buildable code skeleton to `wiki/nvidia/code-walkthroughs/<source-repo>/<algo>/`, write a `skill.md` + `pitfalls.md` at `wiki/nvidia/techniques/<algo>/`, and run an on/off ablation experiment under `sources/experience/api-probes/<op>/`. Warp-specialization + persistent-kernel are the worked examples.

### Step 4: Operator-skill scan

Search for operator-level patterns. These map onto `wiki/nvidia/foundations/compute/<op>/`:

| Pattern in code | Skill class | KB landing |
|---|---|---|
| Aligned GEMM fast-path (no boundary handling) | aligned GEMM | `wiki/nvidia/foundations/compute/gemm/aligned/` |
| Predicate-tail / boundary-tile masking | non-aligned tail handling | `wiki/nvidia/foundations/compute/gemm/non-aligned-tail/` |
| Mainloop-fused dequant (mixed-dtype, e.g. int4 × bf16) | prologue fusion | `wiki/nvidia/foundations/compute/gemm-fused/cutlass-epilogue-prologue/` (prologue side) |
| Epilogue-fused activation / bias / residual / topK / softmax | epilogue fusion | `wiki/nvidia/foundations/compute/gemm-fused/cutlass-epilogue-prologue/` (epilogue side) |
| GroupedGEMM with per-group meta-tensor | grouped-GEMM | `wiki/nvidia/foundations/compute/gemm-grouped/` (when added) |

For each operator skill: write `skill.md` + `pitfalls.md`, land a buildable code anchor under `wiki/nvidia/code-walkthroughs/<source-repo>/<op>/`, and an experience record under `sources/experience/api-probes/<op>/`. Aligned GEMM, non-aligned tail, and fused-GEMM are the worked examples.

### Step 5: On-hardware verification

For every skill landed in steps 2-4: build the code anchor on the target hardware, run the experiment, capture the canonical artifact bundle (`<probe>.cu` + `helper.h` + `build.sh` + `run.sh` + `device.json` + `profiles/*.csv`), and report verdict in the experience record. The skill's `status` is promoted to `verified` only when:

1. `evidence_level: measured` is set (with `measured_on`, `cuda_version_tested`, `toolchain`, `driver_version_tested` populated).
2. The experience record's `verdict` is `verified`.
3. At least one source backlink to `corpus/nvidia/blogs/colfax/...` (or equivalent third-party walkthrough) is present.
4. The numeric correctness gate is the strongest available — direct same-input comparison against a reference (e.g. cuBLAS for GEMM), not an analytic surrogate.

### Step 6: Cross-reference

Link the new skill from related skills' `Cross-references` section. Add the new probe path to any meta skills (this one) that should know about it.

## Measured Characteristics

The procedure was applied end-to-end against `cutlass@f74fea9c` on H200-SXM (cuda 12.9.86, driver 570.124.06) and produced **seven distinct downstream skill landings** across the cutlass-API track (corresponding to the 7 rows in the worked-example table below; the wgmma atom-shape sweep lives under the wgmma skill so they share a row). Verification status: **5 verified** (TMA, wgmma + atom-shape sweep, aligned GEMM, warp-specialization, persistent-kernel); **2 partial** (non-aligned tail — boundary-strategy A/B documented as adjacent-shape proxy with confounders; the same-logical-shape padded harness is follow-up work, not yet on disk. Fused GEMM — skill is partial because the prologue half remains unimplemented; the epilogue probe is independently verified via a fused-vs-fused-reference correctness gate at 2048³). The measurement record is `sources/experience/api-probes/code-extraction/2026-04-28-cutlass-walkthrough.md`; key per-step outputs:

| Procedure step | Downstream skill produced | Status | Measured anchor |
|---|---|---|---|
| Step 2 (TMA hw-feature scan) | `wiki/nvidia/hardware/tma/skill.md` | verified | TMA load 5.37 GB at 5120×4096×4096 |
| Step 2 (wgmma hw-feature scan) | `wiki/nvidia/hardware/wgmma/skill.md` | verified | M64xN{64,128,256}xK8 atom-shape sweep |
| Step 3 (warp-spec algo extraction) | `wiki/nvidia/techniques/warp-specialization/skill.md` | verified | TMA-only 178.9 → plain WS 195.4 TFLOPS = +9.2% |
| Step 3 (persistent-kernel algo extraction) | `wiki/nvidia/techniques/persistent-kernel/skill.md` | verified | pingpong 193.8 / cooperative 186.9 / plain WS 195.4 TFLOPS at 2048³ |
| Step 4 (aligned GEMM operator) | `wiki/nvidia/foundations/compute/gemm/aligned/skill.md` | verified | 512³ 21.7, 2048³ 188.7, 8192³ 292.7 TFLOPS, max_abs=0 vs cuBLAS |
| Step 4 (non-aligned-tail operator) | `wiki/nvidia/foundations/compute/gemm/non-aligned-tail/skill.md` | partial | 80³, 200³, 1440³ all max_abs=0; A/B is documented proxy |
| Step 4 (fused GEMM operator) | `wiki/nvidia/foundations/compute/gemm-fused/cutlass-epilogue-prologue/skill.md` | **partial** (prologue half unimplemented; the `2026-04-28-gemm-fused.md` epilogue probe itself is independently verified with a fused-vs-fused-reference correctness gate at 2048³, fused 86.83 vs non-fused 100.21 μs = ~13% savings, 40% less DRAM traffic) |

The procedure is repeatable: the rg/build/run commands in `sources/experience/api-probes/code-extraction/2026-04-28-cutlass-walkthrough.md` can be re-executed by another agent on the same hardware to reproduce the outputs.

## Worked example: cutlass end-to-end

The procedure above was applied to cutlass `f74fea9c` on H200-SXM. Each step landed the following:

| Step | Output | KB paths |
|---|---|---|
| 2 (TMA) | TMA hw-probe | `wiki/nvidia/hardware/tma/skill.md`, `sources/experience/api-probes/gemm/2026-04-28-tma-bandwidth-counters.md`, `wiki/nvidia/code-walkthroughs/cutlass-cute/example48-hopper-warp-specialized-gemm/` |
| 2 (wgmma) | wgmma hw-probe + atom-shape sweep | `wiki/nvidia/hardware/wgmma/skill.md`, `sources/experience/api-probes/gemm/2026-04-28-wgmma-counters.md`, `sources/experience/api-probes/gemm/2026-04-28-wgmma-atom-shape-sweep.md`, `wiki/nvidia/code-walkthroughs/cutlass-cute/wgmma-atom-decoding/` |
| 4 (aligned GEMM) | aligned GEMM skill at 512³/2048³/8192³ | `wiki/nvidia/foundations/compute/gemm/aligned/`, `wiki/nvidia/code-walkthroughs/cutlass-cute/gemm-aligned/`, `sources/experience/api-probes/gemm/2026-04-28-gemm-aligned.md` |
| 4 (non-aligned tail) | non-aligned tail skill at 80³/200³/1440³ + adjacent-shape proxy | `wiki/nvidia/foundations/compute/gemm/non-aligned-tail/`, `wiki/nvidia/code-walkthroughs/cutlass-cute/gemm-tail/`, `sources/experience/api-probes/gemm/2026-04-28-gemm-tail.md` |
| 3 (warp-spec) | warp-specialization classical-algo skill + on/off A/B at 2048³ | `wiki/nvidia/techniques/warp-specialization/`, `wiki/nvidia/code-walkthroughs/cutlass-cute/example48-hopper-warp-specialized-gemm/`, `sources/experience/api-probes/gemm/2026-04-28-warp-specialization-ablation.md` |
| 3 (persistent kernel) | persistent-kernel classical-algo skill + pingpong-vs-cooperative A/B at 2048³ | `wiki/nvidia/techniques/persistent-kernel/`, `wiki/nvidia/code-walkthroughs/cutlass-cute/persistent-kernel/`, `sources/experience/api-probes/gemm/2026-04-28-persistent-kernel-ablation.md` |
| 4 (fused) | fused-GEMM (skill partial; prologue not yet implemented) | `wiki/nvidia/foundations/compute/gemm-fused/cutlass-epilogue-prologue/` (status: partial), `wiki/nvidia/code-walkthroughs/cutlass-cute/gemm-fused/`, `sources/experience/api-probes/gemm/2026-04-28-gemm-fused.md` (probe is epilogue-scoped and independently verified) |

Each landing follows the canonical artifact-bundle layout (cf. `sources/experience/hw-probes/warp-divergence-cost/artifacts/` reference); the cutlass-vs-cuBLAS direct-comparator harness (`sources/experience/api-probes/gemm/artifacts/gemm_compare.cu`) is the numeric-correctness gate for the aligned, non-aligned-tail, and warp-spec/persistent-kernel skills.

## Non-coverage (explicit)

This meta-skill **does not** replace:

- **On-hardware verification**: even after the procedure above is followed, every skill must independently pass `kp_introspect` + canonical-bundle reproduction + numeric-correctness gate before being promoted to `status: verified`. The meta-skill is a procedure; it doesn't grant verification.
- **Hardware-specific tuning**: tile/stage/cluster choices remain per-shape and per-SM. The meta-skill enumerates which knobs exist; it does not pick values for them.
- **PTX-down work**: this meta-skill describes the cutlass-API extraction path. Cutlass-free PTX reimplementations are a separate workflow that re-uses the algorithms identified here as references but writes new code without `cutlass::` / `cute::` includes.

## Cross-references

- TMA: `wiki/nvidia/hardware/tma/skill.md` + `sources/experience/api-probes/gemm/2026-04-28-tma-bandwidth-counters.md`
- wgmma: `wiki/nvidia/hardware/wgmma/skill.md` + `sources/experience/api-probes/gemm/2026-04-28-wgmma-counters.md` + `2026-04-28-shape-sweep.md`
- Aligned GEMM: `wiki/nvidia/foundations/compute/gemm/aligned/skill.md` + `sources/experience/api-probes/gemm/2026-04-28-gemm-aligned.md`
- Non-aligned tail: `wiki/nvidia/foundations/compute/gemm/non-aligned-tail/skill.md` + `sources/experience/api-probes/gemm/2026-04-28-gemm-tail.md`
- Warp-specialization: `wiki/nvidia/techniques/warp-specialization/skill.md` + `sources/experience/api-probes/gemm/2026-04-28-warp-specialization-ablation.md`
- Persistent-kernel: `wiki/nvidia/techniques/persistent-kernel/skill.md` + `sources/experience/api-probes/gemm/2026-04-28-persistent-kernel-ablation.md`
- Fused-GEMM (epilogue verified; prologue queued): `wiki/nvidia/foundations/compute/gemm-fused/cutlass-epilogue-prologue/skill.md` + `sources/experience/api-probes/gemm/2026-04-28-gemm-fused.md`

## Failure modes

See `wiki/nvidia/foundations/meta/code-extraction/pitfalls.md`.
