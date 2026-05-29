---
id: pitfall-code-extraction
type: pitfall
vendor: nvidia
title: Pitfalls
---
# Code-extraction meta-skill — pitfalls

## 1. Don't conflate "non-aligned to canonical kernel" with "non-aligned to executing kernel"

When extracting a non-aligned-tail-handling skill from cutlass, verify the chosen test shape is non-aligned to the **kernel template that actually executes it**, not just the canonical cooperative kernel. A 64³ datapoint, for example, is a perfect tile match for the small-tile fallback (`<64, 64, 32>`), so the predicate-tail path never fires there even though 64³ is non-aligned to the cooperative `<128, 128, 32>` tile. The lesson generalizes: for any extraction that exercises a fallback / smaller-tile / different-cluster code path, the alignment check must be re-run against the executing kernel's CtaTile, not the canonical one.

(BitLesson reference: `BL-20260428-shape-non-aligned-on-executing-kernel`.)

## 2. Don't conflate scale effects with strategy cost in A/B comparisons

When the "without strategy" anchor is at a different problem size than the "with strategy" target, the throughput delta confounds the strategy cost with absolute-scale effects (wave quantization, occupancy ramp, cluster-multicast efficiency). For compile-time strategies (no runtime toggle), the closest practical A/B is the same kernel binary at adjacent shapes; the extraction must explicitly document this as a proxy with remaining confounders, not a clean isolation.

(BitLesson reference: `BL-20260428-isolate-boundary-from-scale`.)

## 3. Direct same-input numeric comparison vs analytic surrogate

When verifying the extracted skill's correctness, prefer a direct same-A-and-B comparison against a reference implementation (e.g. `cublasGemmEx` for GEMM) over an analytic surrogate (e.g. "the all-ones input gives `C[0] = K`"). The analytic surrogate misses any per-element bug that happens not to affect `C[0]`. The aligned-GEMM landing in this KB uses `gemm_compare.cu` for exactly this reason.

## 4. Source-corpus backlinks are a hard requirement, not a nice-to-have

Every extracted skill must cite at least one third-party walkthrough in `corpus/nvidia/...` (typically a Colfax tutorial or NVIDIA blog). Citing only the upstream cutlass `*.hpp` sources is insufficient because cutlass headers are dense template-metaprogramming and don't explain the algorithmic intent. The third-party walkthrough is the explanation; the cutlass header is the implementation.

## 5. The canonical artifact-bundle layout is fixed

Every probe lands at `80-experience/<api-probes|hw-probes>/<topic>/artifacts/{<probe>.cu, helper.h, build.sh, run.sh, device.json, profiles/*.csv}`. Don't deviate. The reference is `80-experience/hw-probes/warp-divergence-cost/artifacts/`. Reviewers will reject probes with a non-standard layout, even if the actual measurements are correct.

## 6. `evidence_level: measured` requires environment fields

When you set `evidence_level: measured`, the linter will require `cuda_version_tested`, `driver_version_tested`, `toolchain`, and `measured_on` to be non-empty. Don't set `measured` without filling these in; use `evidence_level: spec` for documentation-only landings.

## 7. Meta-skill does not grant verification

This skill describes the procedure for extracting and landing skills. Following the procedure does NOT promote a downstream skill to `status: verified`; that promotion still requires on-hardware verification (canonical-bundle reproduction + numeric-correctness gate + reviewer approval). The meta-skill is the recipe, not the certification.
