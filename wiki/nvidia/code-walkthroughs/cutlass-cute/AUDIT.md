---
id: code-cutlass-cute-AUDIT
type: code-walkthrough
vendor: nvidia
title: Audit
upstream_repo: NVIDIA/cutlass-cute
---
<!--
lint-exempt: this file lives under 60-code/ which AGENTS.md exempts from frontmatter
requirements. A YAML block is included below for human readers; lint will skip it.
-->

---
title: "Gap audit: CUTLASS GEMM/Attention coverage in kernel-kb-mvp/knowledge"
audit_round: rlcr-2026-05-08-round-2
target_plan: docs/cutlass_explore_update_01.md
classification_keys: [complete, partial, missing, redundant]
upstream_repo_pinned: cutlass@f74fea9c
auditor_inputs:
  - knowledge/40-hardware-feature/
  - knowledge/30-skill/compute/
  - knowledge/30-skill/memory/
  - knowledge/30-skill/sync/
  - knowledge/30-skill/meta/
  - knowledge/50-classical-algo/
  - knowledge/60-code/cutlass-cute/
  - knowledge/80-experience/api-probes/
  - knowledge/80-experience/hw-probes/
  - knowledge/corpus/nvidia/MANIFEST.yaml
  - knowledge/templates/frontmatter/
upstream_root: "{{CUTLASS_REPO_REF}}"
---

# Gap Audit: CUTLASS GEMM/Attention Coverage

This file is the deliverable for the gap-audit task in `docs/cutlass_explore_update_01.md`. It catalogs every existing GEMM/wgmma/TMA/attention-relevant entry in `knowledge/`, classifies each, and lists concrete additions needed. All "missing | partial" rows cite a specific cutlass source path under `{{CUTLASS_REPO_REF}}/` that justifies the gap, and propose a target KB layer path.

The classification rubric is:
- **complete** — entry exists, satisfies its layer's contract (skill+pitfalls+optional apis for skill layer; README+tuning+optional skeleton for `60-code/`), and its referenced sources/artifacts are present.
- **partial** — entry exists but is missing a required companion file, has thin content (<≈50 lines for skill body), or its `tuning.md` is absent where convention requires one.
- **missing** — no entry exists for a topic the plan requires; cite cutlass file(s) justifying the gap.
- **redundant** — overlapping coverage that should be consolidated; cite both paths.

All paths are relative to `knowledge/` unless prefixed with `<path-removed> (absolute upstream paths).

---

## Section A — `40-hardware-feature/` (Hardware primitives)

| Entry | Files | Status | Evidence | Classification | Notes / Gap |
|-------|-------|--------|----------|----------------|-------------|
| `40-hardware-feature/tma/` | skill.md, pitfalls.md | verified | measured | complete | TMA cute end-to-end on Hopper; pitfalls list 8 items; could add a measured 3D TMA case but not blocking. |
| `40-hardware-feature/tma-ptx/` | skill.md, pitfalls.md | verified | measured | complete | Raw PTX TMA (cutlass-free); has 4 hw-probe records (hello, throughput, multicast v1, multicast v2). |
| `40-hardware-feature/wgmma/` | skill.md, pitfalls.md | verified | measured | complete | **Round 1 closed**: Added "## Atomic Usage (cute)" section with 11-row mapping table (MMA_Atom → PTX mnemonic → probe config → probe record) plus cute-API snippet. Cutlass anchor: `{{CUTLASS_REPO_REF}}/include/cute/atom/mma_traits_sm90_gmma.hpp`. |
| `40-hardware-feature/wgmma-ptx/` | skill.md, pitfalls.md, atomic_skeleton.md | verified | measured | complete | **Round 1 closed**: Added "## Atomic Usage (PTX)" section with raw PTX template + build/run commands, and created `atomic_skeleton.md` documenting the full single-atom sequence (fence→mma→commit→wait) with descriptor construction and fragment layout. |
| `40-hardware-feature/mma-sync-ptx/` | skill.md, pitfalls.md | draft | spec | complete | **Round 1 closed**: Created skill.md with full atom shape table (12 families), PTX mnemonic template, and 5 pitfalls. `status: draft`, `evidence_level: spec`, `requires_sm: ">=8.0"`. |
| `40-hardware-feature/ldmatrix-ptx/` | skill.md, pitfalls.md | draft | spec | complete | **Round 1 closed**: Created skill.md with all 6 ldmatrix variants (x1/x2/x4 non-transposed + x2/x4/x8 transposed) plus movmatrix, and 5 pitfalls. Uses corrected paths (`copy_traits_sm75.hpp`, not `mma_traits_sm80*.hpp`). |
| `40-hardware-feature/tcgen05-ptx/` | skill.md, pitfalls.md | draft | spec | complete | **Round 1 closed**: Created skill.md with `not_measured_on_target: true`, `requires_sm: ">=10.0"`. Documents tcgen05.cp, tcgen05.ld, and SM100_MMA_* traits. Cites Blackwell whitepaper and cutlass sm100 headers. 3 pitfalls. |

---

## Section B — `30-skill/compute/gemm*` and `30-skill/compute/attention/` (Operator skills)

| Entry | Files | Status | Evidence | Classification | Notes / Gap |
|-------|-------|--------|----------|----------------|-------------|
| `30-skill/compute/gemm/aligned/` | skill.md, pitfalls.md, tuning.md | verified | measured | complete | **Round 1 closed**: Created sibling `tuning.md` with compile-time and runtime parameter tables. Added back-link from `skill.md` body. |
| `30-skill/compute/gemm/non-aligned-tail/` | skill.md, pitfalls.md, tuning.md | partial | measured | partial | **Round 1 partial**: Created sibling `tuning.md` with tail-strategy knobs. Added back-link from `skill.md`. Status remains partial (coverage incomplete). |
| `30-skill/compute/gemm-fused/cutlass-epilogue-prologue/` | skill.md, pitfalls.md, tuning.md | partial | measured | partial | **Round 1 partial**: Created sibling `tuning.md` with EVT and prologue fusion knobs. Added back-link from `skill.md`. Status remains partial (prologue coverage incomplete). |
| `30-skill/compute/gemm-ptx/` | skill.md, pitfalls.md | partial | measured | partial | Cutlass-free PTX GEMM track. No apis.md (acceptable — pure PTX). Cross-link gap to any tuning.md is absent because no `60-code/.../gemm-ptx/` partner exists. Out of plan scope unless the Round-N tuning sweep targets PTX-GEMM. |
| `30-skill/compute/attention/mvp-minimal/` | skill.md, pitfalls.md, tuning.md | verified | measured | complete | **Round 9 verified**: TMA+wgmma kernel passes correctness on H200 (max_abs_err=0.000061, 0/16384 mismatches, sanitizer 0 errors). Uses canonical CUTLASS smem layout (Layout_K_SW128_Atom + Swizzle<3,4,3> + CU_TENSOR_MAP_SWIZZLE_128B). Evidence logs checked in. |
| `30-skill/compute/attention/cutlass-fmha/` | skill.md, pitfalls.md, tuning.md | verified | measured | complete | **Round 13 measured**: H200-verified via repo-local wrapper. Example: 376.1 TFLOPS (B2 H16 S1024 D128 full). Collective: 348.5 TFLOPS (B2 H16 S2048 D128 causal). |

Other compute skills (`branch-elimination`, `compiler-hints`, `fast-math`, `half-precision-math`, `occupancy-tuning`, `warp-divergence`, `warp-primitives`, `ilp`) are out of plan scope; only `ilp` is missing apis.md and that is unrelated to this plan.

---

## Section C — `50-classical-algo/` (End-to-end algorithm patterns)

| Entry | Files | Status | Evidence | Classification | Notes / Gap |
|-------|-------|--------|----------|----------------|-------------|
| `50-classical-algo/warp-specialization/` | skill.md, pitfalls.md | verified | measured | complete | Producer/consumer warpgroup pattern; pinned to `cutlass@f74fea9c`. Pitfalls is short (<50 lines) but covers the canonical hazards. |
| `50-classical-algo/persistent-kernel/` | skill.md, pitfalls.md | verified | measured | complete | Persistent-kernel ping-pong; same depth/notes as above. |
| `50-classical-algo/online-softmax/` | — | — | — | out-of-scope | Online-softmax is documented inline within `30-skill/compute/attention/mvp-minimal/skill.md` (algorithmic skeleton) and `80-experience/api-probes/attention/artifacts/flash_attn_tma_wgmma.cu` (implementation). A standalone `50-classical-algo/` entry is optional — the algorithm is already covered by the attention skill entries. Not required by any plan AC. |
| `50-classical-algo/split-k/` | — | — | — | absent (out of plan scope) | Listed in `AGENTS.md` as a future classical algo; not required by current plan ACs. Recorded for reference only. |

---

## Section D — `60-code/cutlass-cute/` (Code-extraction layer)

This subtree is lint-exempt per AGENTS.md; entries carry README.md, tuning.md, and optional `*_skeleton.md`.

| Entry | Files | Classification | Notes / Gap |
|-------|-------|----------------|-------------|
| `60-code/cutlass-cute/example48-hopper-warp-specialized-gemm/` | README.md, mainloop_skeleton.md | complete | Reading guide for cutlass example 48; pairs with `30-skill/compute/gemm/aligned/`. Verify: `{{CUTLASS_REPO_REF}}/examples/48_hopper_warp_specialized_gemm/`. |
| `60-code/cutlass-cute/gemm-aligned/` | README.md, tuning.md | complete | Cross-linked from this audit; needs back-link from `30-skill/compute/gemm/aligned/skill.md` body (closes AC-9 cross-link). |
| `60-code/cutlass-cute/gemm-fused/` | README.md, tuning.md | complete | Same cross-link note. |
| `60-code/cutlass-cute/gemm-tail/` | README.md, tuning.md | complete | Same cross-link note. |
| `60-code/cutlass-cute/persistent-kernel/` | README.md, tuning.md | complete | Documents persistent-kernel knobs; pairs with `50-classical-algo/persistent-kernel/`. |
| `60-code/cutlass-cute/wgmma-atom-decoding/` | README.md, wgmma_skeleton.md | complete | Skeleton walkthrough for wgmma atom decoding; pairs with `40-hardware-feature/wgmma/`. |
| `60-code/cutlass-cute/attention-fmha-example/` | README.md, fmha_skeleton.md, tuning.md | complete | **Round 13**: H200-measured 88_hopper_fmha (376.1 TFLOPS ping-pong at B2 H16 S1024 D128). Build/run scripts + probe record. 41_fused fallback context documented. |
| `60-code/cutlass-cute/attention-fmha-collective/` | README.md, tuning.md | complete | **Round 13**: H200-measured collective config (348.5 TFLOPS ping-pong at B2 H16 S2048 D128 causal). Probe record with collective settings cited. |
| `60-code/flash-attention-v3/` | README.md, tuning.md | complete | **Round 1 closed**: Registered in MANIFEST.yaml (commit `bbda031f`, BSD-3-Clause, local path `{{FLASH_ATTENTION_REPO_REF}}`). Created README (source files, algorithmic comparison vs cutlass FMHA and MVP minimal) and tuning.md (compile-time + runtime knobs, differences from cutlass FMHA). |

---

## Section E — `80-experience/hw-probes/` (Microbench records)

| Entry | Records | Classification | Notes / Gap |
|-------|---------|----------------|-------------|
| `80-experience/hw-probes/wgmma-ptx/` | hello, zoo (11 configs) | complete | Already exercises the canonical zoo. AC-2.1 needs explicit 1:1 mapping table in `40-hardware-feature/wgmma/skill.md`, not new probe records. |
| `80-experience/hw-probes/tma-ptx/` | hello, throughput, multicast-v1, multicast-v2 | complete | No gap. |
| `80-experience/hw-probes/canonical/` | — (empty) | partial | Empty placeholder; clarify with maintainers whether this is a stub for future canonical microbenches or stale. Out of plan scope; flagged for visibility. |
| `80-experience/hw-probes/attention-tuning/` | 2026-05-08-attention-tile-sweep.md, artifacts/profiles/ | complete | **Round 0 created**: 6-config BLOCK_M x BLOCK_N tile-shape sweep at fixed B=1,H=2,S=256,D=64 on H200. CSV includes throughput_gflops column. |
| `80-experience/hw-probes/gemm-tuning/` (or operator-specific) | — | missing-or-deferred | AC-9 requires ≥1 measured sweep across operators; if attention-tuning is the chosen sweep, this stays optional. Plan lower bound permits one measured sweep total. |

Other probe directories (smem-bank-conflict, occupancy-sweep, warp-divergence-cost, fast-math, register-pressure, etc.) are out of plan scope.

---

## Section F — `80-experience/api-probes/` (API-probe records)

| Entry | Files | Classification | Notes / Gap |
|-------|-------|----------------|-------------|
| `80-experience/api-probes/gemm/` | 10 dated records, artifacts/ (14 .cu, build/run scripts, device.json, profiles/) | complete | Hot path; existing artifacts are the reference for AC-9 cross-linking. |
| `80-experience/api-probes/attention/` | artifacts/{flash_attn_tma_wgmma.cu, flash_attn_minimal.cu, wgmma_desc_test.cu, cutlass_88_hopper_fmha.cu, build*.sh, run*.sh, device.json, profiles/}, 4 dated probe records | complete | **Round 14**: repo-local FMHA wrapper added. TMA+wgmma (0.35 TFLOPS), FMHA example (376.1 TFLOPS), FMHA collective (348.5 TFLOPS). All with H200 evidence. |

---

## Section G — `corpus/nvidia/MANIFEST.yaml` (Source registry)

| Source | Present? | Classification | Notes / Gap |
|--------|----------|----------------|-------------|
| cuda-official/toolkit-docs-13.2 | yes | complete | — |
| blogs/colfax | yes | complete | Tags: tma, wgmma, async-pipeline. |
| whitepapers/gpu-wite-paper | yes | complete | Hopper + Blackwell whitepapers (covers tcgen05 spec citation). |
| source-code/cutlass | yes | complete | Pinned at `cutlass@f74fea9c`. |
| source-code/cuda-samples | yes | complete | — |
| legacy-knowledge | yes | complete | KernelPilot pre-refactor; not truth source. |
| FlashAttention v3 (Tri Dao) | yes | complete | **Round 1 closed**: Registered as `source-code/flash-attention` with commit `bbda031f`, BSD-3-Clause license, local path `{{FLASH_ATTENTION_REPO_REF}}`. |

---

## Summary

### Targeted closures by AC
- **AC-1**: this file.
- **AC-2.1**: extend `40-hardware-feature/wgmma/skill.md` with `## Atomic Usage (cute)` + 1:1 mapping table to `80-experience/hw-probes/wgmma-ptx/zoo/`.
- **AC-2.2**: extend `40-hardware-feature/wgmma-ptx/skill.md` with `## Atomic Usage (PTX)` + create `40-hardware-feature/wgmma-ptx/atomic_skeleton.md` (companion `.cu` referenced from `80-experience/api-probes/`).
- **AC-3**: create `40-hardware-feature/mma-sync-ptx/{skill.md,pitfalls.md}` and `40-hardware-feature/ldmatrix-ptx/{skill.md,pitfalls.md}`.
- **AC-4**: create `40-hardware-feature/tcgen05-ptx/{skill.md,pitfalls.md}` (spec-only).
- **AC-5**: create `30-skill/compute/attention/mvp-minimal/{skill.md,pitfalls.md,tuning.md}` + artifacts at `80-experience/api-probes/attention/artifacts/`.
- **AC-6**: create `60-code/cutlass-cute/attention-fmha-example/{README.md,fmha_skeleton.md,tuning.md}` + **measured artifact** for `88_hopper_fmha` on H200 (at least one correctness/timing datapoint).
- **AC-7**: create `60-code/cutlass-cute/attention-fmha-collective/{README.md,tuning.md}` + **measured collective datapoint** on H200 (at least one configuration with measured_on + artifact paths).
- **AC-8**: register FAv3 in `corpus/nvidia/MANIFEST.yaml` and create `60-code/flash-attention-v3/{README.md,tuning.md}`.
- **AC-9**: cross-link `60-code/cutlass-cute/<op>/tuning.md` from each `30-skill/compute/gemm*/skill.md`; add `tuning.md` to new attention skill (AC-5); land ≥1 measured sweep at `80-experience/hw-probes/<operator>-tuning/`.
- **AC-10**: run `python -m tools.lint_knowledge` and confirm exit 0 across full tree and per-layer.
- **AC-11**: grep produced artifacts to confirm no plan-document terminology leaked.

### Out-of-scope items flagged for visibility (not closed by this plan)
- `30-skill/compute/ilp/` missing apis.md.
- `30-skill/memory/bank-conflict/` missing apis.md.
- `30-skill/sync/memory-ordering/` is draft + spec-only.
- `80-experience/hw-probes/canonical/` empty placeholder.
- `10-api-raw/{ptx,math-intrinsics,cccl/cub,_classified}/` empty namespaces.

### Reproducibility note
Re-running this audit against the current `knowledge/` tree should reproduce the same classification provided no entries are added or moved. The classification uses observable filesystem state only (presence of `skill.md`, `pitfalls.md`, `tuning.md`, frontmatter `status`/`evidence_level`, hw-probe record counts) and verified upstream paths under `{{CUTLASS_REPO_REF}}/`.
