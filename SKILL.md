---
name: KernelWiki
description: >-
  Use when the user asks about optimizing NVIDIA GPU kernels (Blackwell SM100,
  Hopper SM90, or general CUDA), foundational CUDA optimization skills
  (coalescing, warp primitives, shared memory, etc.), hardware feature
  characterization (tcgen05, TMA, wgmma), operator routing (elementwise,
  reduction, GEMM), API definitions, or wants PR references from
  CUTLASS/SGLang/vLLM/FlashInfer/PyTorch. Future: Huawei Ascend NPU, Biren GPU.
  Do NOT use for host-side framework integration or distributed systems
  (DeepEP/EPLB/DualPipe).
argument-hint: "[natural-language-question] | [--vendor nvidia --tag foo --type skill] | [page-id]"
allowed-tools: "Bash Read Grep Glob"
---

# KernelWiki — GPU Kernel Optimization Knowledge Base

> **Knowledge cutoff: 2026-04-27.** All upstream PR data, blog summaries, and version-claim entries reflect upstream state on or before this date (per `data/refresh-cutoff.yaml`). Re-run the refresh tooling to advance the cutoff.

Query a structured, cross-referenced knowledge base of GPU kernel optimization — 2179 merged PRs, 200+ wiki synthesis pages, foundational CUDA skills, API definitions, operator routing guides, code walkthroughs, and measurement records.

## When To Use This Skill

Trigger this skill when the user asks about:

- **Blackwell/SM100 kernel programming** — tcgen05.mma, TMEM, CLC, 2-SM cooperative, NVFP4, FP8/FP4 block scaling, PDL/GDC
- **Kernel implementations** — FlashAttention-4, DeepGEMM, FlashMLA, NSA, GatedDeltaNet, NVFP4 GEMM/GEMV, fused MoE, gated dual GEMM
- **Foundational CUDA optimization skills** — coalescing, warp primitives, bank conflicts, shared memory, register pressure, vectorized access, occupancy tuning
- **Operator routing** — elementwise, reduction, normalization, GEMM, pooling, scan/cumulative decision trees
- **API definitions** — CUDA Runtime API semantics, PTX instructions, intrinsics
- **Performance patterns** — low SM utilization, memory-bound, register pressure, compute-bound, tail effects, pipeline stalls
- **DSLs for Blackwell** — CuTe DSL, CUDA C++ with PTX inline, Triton on Blackwell
- **Hopper → Blackwell migration** — wgmma → tcgen05, register → TMEM accumulators
- **PR references** — "how did vLLM/SGLang/FlashInfer/CUTLASS/PyTorch implement X for SM100?"
- **Competition solutions** — GPU Mode NVFP4 hackathon, FlashInfer MLSys 2026 submissions

Do NOT use this skill for:

- Host-side framework integration (model loading, request routing, scheduling policy)
- Distributed systems topics — DeepEP, EPLB, DualPipe are out of scope

## How To Query

All commands below run from the skill directory (the clone root — the directory this `SKILL.md` lives in). The scripts auto-resolve the wiki root; **no environment variable required**.

### Path 1: Unified search (preferred for natural language)

```bash
python3 scripts/query.py "how to fuse gate-up dual GEMM on Blackwell"
python3 scripts/query.py --tag nvfp4 --type kernel
python3 scripts/query.py --repo cutlass --limit 20
python3 scripts/query.py --symptom tail-effect --compact
```

Filters: `--type`, `--tag`, `--repo`, `--language`, `--architecture`,
`--symptom`, `--confidence`, `--limit`, `--compact`, `--paths-only`. `--tag`
and `--architecture` accept aliases — `--tag UMMA` matches `tcgen05`,
`--architecture B200` matches `sm100`, etc.

### Path 2: Fetch a specific page by id or path

```bash
python3 scripts/get_page.py kernel-flash-attention-4
python3 scripts/get_page.py pr-cutlass-2472
python3 scripts/get_page.py kernel-flash-attention-4 --follow-sources
python3 scripts/get_page.py kernel-flash-attention-4 --body-only
```

### Path 3: Regex text search across wiki bodies and PR pages

```bash
python3 scripts/grep_wiki.py "tcgen05\\.fence"
python3 scripts/grep_wiki.py "2-CTA backward" --only wiki
python3 scripts/grep_wiki.py "nvfp4" "block_scale" --any
```

### Path 4: Pre-built cross-reference indices

Auto-generated under `queries/`:

- `queries/by-problem.md` — symptom → pattern page → candidate techniques
- `queries/by-technique.md` — 15 techniques with architectures, confidence, reproducibility, source count
- `queries/by-hardware-feature.md` — tcgen05/tmem/clc/tma/nvfp4/etc. → related wiki + PR pages
- `queries/by-kernel-type.md` — gemm/attention/moe/mla/gated-delta-net → pages
- `queries/by-language.md` — cute-dsl/cuda-cpp/ptx/triton → guide page + related kernels/sources
- `queries/by-repo.md` — all 2179 PRs across cutlass/sglang/vllm/flashinfer/pytorch/DeepGEMM

### Path 5: Primer, schema, examples

Companion docs under `references/`:

- `references/primer.md` — topic map: hardware features, techniques, symptoms, canonical page IDs. Read this first when the question is broad.
- `references/schema.md` — condensed frontmatter schema, confidence rules, reproducibility ladder, controlled vocabulary, canonical aliases.
- `references/examples.md` — 10 worked query patterns mapping user questions → command sequences → synthesis.

### Hopper / SM90 example queries

Use these examples when the user asks for Hopper-specific kernel guidance:

```bash
# Hopper wgmma / TMA GEMM implementation paths
python3 scripts/query.py "Hopper wgmma TMA GEMM" --architecture sm90 --type skill
python3 scripts/get_page.py wiki/nvidia/foundations/compute/gemm.md --follow-sources
python3 scripts/get_page.py wiki/nvidia/foundations/compute/gemm-ptx.md

# Raw PTX primitives on Hopper
python3 scripts/query.py "cutlass-free wgmma ptx" --architecture sm90 --language ptx
python3 scripts/query.py "TMA PTX cuTensorMapEncodeTiled mbarrier" --architecture sm90 --language ptx
python3 scripts/grep_wiki.py "wgmma.mma_async" --only wiki

# Hopper optimization symptoms
python3 scripts/query.py --symptom pipeline-stalls --architecture sm90
python3 scripts/query.py --symptom tail-effect --architecture sm90
python3 scripts/query.py "Hopper warp specialization pingpong cooperative" --architecture sm90

# Hopper migration context
python3 scripts/query.py "Hopper wgmma to Blackwell tcgen05 migration"
python3 scripts/query.py "register accumulators to TMEM" --architecture sm100
```

## Output Pattern

When answering from this KB:

1. **Cite specific pages** with paths (e.g., `wiki/nvidia/kernels/flash-attention-4.md`) and IDs (`kernel-flash-attention-4`).
2. **Follow `sources:` fields** to trace claims back to PRs/blogs/docs.
3. **Respect confidence levels** — `verified` > `source-reported` > `inferred` > `experimental`. Call out when a claim is `experimental` or `inferred`.
4. **Include code snippets** from wiki pages when they exist — technique/kernel/language pages are guaranteed `snippet`-reproducibility (validator-enforced).
5. **Report performance claims with all six fields** — `gpu`, `dtype`, `shape`, `metric`, `value`, `source_id`.

## Knowledge Base Contents (knowledge cutoff: 2026-04-27)

- **2474 total markdown pages — 2179 PR references + 200+ wiki synthesis (hardware, techniques, kernels, foundations, API definitions, operator routing, code walkthroughs) + 48 experience records + 20 blogs + 11 docs + 7 contests
- **6 candidate ledgers** in `candidates/` — 4,222 merged PRs classified (include/defer/exclude) Jan 2025 – Apr 2026
- **117 verbatim/extracted/derived asset bundles** in `artifacts/` (PR diffs, kernel files, blog code) — pinned to upstream SHAs via `PROVENANCE.yaml`
- **6 auto-generated query indices** in `queries/`
- **Controlled vocabulary** (80+ tags) in `data/tags.yaml`, alias map in `data/aliases.yaml`
- **Hybrid version-claim registry** — per-page `version_sensitive: <id>` pointers + `data/version-claims.yaml` central registry, validated for bidirectional consistency
- **Validator** `scripts/validate.py` — 2474 files / 117 bundles / 6 ledgers / 0 errors
- **Architecture-neutral** — SM90/Hopper, SM100/Blackwell, and general CUDA pages are all first-class when tagged with `architectures`.

The knowledge cutoff date is the last day on which upstream PRs / blog snapshots were refreshed. To advance it: run `scripts/refresh_candidate_ledger.py`, regenerate PR pages, then bump `data/refresh-cutoff.yaml::cutoff_date`.

## Quality Guarantees

- Every `verified` page has official-doc + upstream-code evidence
- Every technique/kernel/language page has a compilable snippet
- Every PR page has `inclusion_reason` and `status: merged`
- Architecture coverage is expressed through `architectures` fields; no architecture requires a special relevance justification.
