---
id: code-cutlass-cute-AGENTS
type: code-walkthrough
vendor: nvidia
title: Agents
upstream_repo: NVIDIA/cutlass-cute
---
# 60-code / cutlass-cute

This directory holds **CUTLASS / CuTe source-reading notes on Hopper**: how specific upstream examples and template fragments express TMA, WGMMA, warp-specialized mainloops, tails, and fused epilogues. The prose half (what the technique *is*) lives in `wiki/nvidia/hardware/<feature>/`, `wiki/nvidia/techniques/<algo>/`, and `wiki/nvidia/foundations/compute/<gemm-variant>/`.

`tools/lint_knowledge.py` skips the entire `wiki/nvidia/code-walkthroughs/` subtree by default (see SKIP_DIRS), so the markdown skeletons, READMEs, and tuning logs are not subjected to the `skill / api-raw / experience` frontmatter contracts. If a topic directory introduces a knowledge-prose document with frontmatter (e.g. an embedded skill), it must move to the right `wiki/nvidia/foundations/` / `wiki/nvidia/hardware/` / `wiki/nvidia/techniques/` location or be flagged for the linter explicitly.

## Source-driven layout

The second-level directory always names a **source repository**:

```
wiki/nvidia/code-walkthroughs/
├── cutlass-cute/        # this directory — extracted from cutlass / CuTeDSL
│   ├── example48-hopper-warp-specialized-gemm/
│   │                         # ex48: TMA + WGMMA + cooperative WS mainloop
│   ├── wgmma-atom-decoding/ # CUTLASS WGMMA atom decoding
│   ├── gemm-aligned/        # aligned cooperative GEMM template parameters
│   ├── gemm-tail/           # tail-handling template selection by shape
│   ├── persistent-kernel/   # pingpong / cooperative persistent schedules
│   └── gemm-fused/          # fused epilogue / prologue patterns
└── ptx-gemm/                # cutlass-free PTX track — minimal compilable repro lives here
```

`wiki/nvidia/code-walkthroughs/ptx-gemm/` is a different shape (cutlass-free track keeps its working `.cu` here, since it is the single source of truth for that path — there is no upstream cutlass example to point at). All other topics under `cutlass-cute/` are documentation-only.

## Where the reproducible artifacts actually live

| Topic | Canonical artifact path |
|---|---|
| Example 48 Hopper WS GEMM | `artifacts/experience/api-probes/gemm/gemm_aligned.cu` |
| TMA cutlass-free | `artifacts/experience/hw-probes/tma-ptx/{tma_hello.cu, tma_throughput_probe.cu}` |
| wgmma cutlass-API | `artifacts/experience/api-probes/gemm/gemm_aligned.cu` (same harness) |
| wgmma cutlass-free | `artifacts/experience/hw-probes/wgmma-ptx/{wgmma_hello.cu, wgmma_zoo.cu}` |
| Aligned GEMM | `artifacts/experience/api-probes/gemm/gemm_aligned.cu` |
| Tail GEMM | `artifacts/experience/api-probes/gemm/{gemm_tail.cu, gemm_tail_small.cu}` |
| Plain WS schedule | `artifacts/experience/api-probes/gemm/gemm_compare_ws.cu` |
| Pingpong schedule | `artifacts/experience/api-probes/gemm/gemm_compare_pingpong.cu` |
| Cooperative schedule | `artifacts/experience/api-probes/gemm/gemm_aligned.cu` (auto-selects cooperative at large aligned shapes) |
| Cutlass-free WS | `artifacts/experience/kernel-records/2026-04-29-gemm-ws-ptx/gemm_ws_ptx.cu` |
| Fused-GEMM (ex50) | upstream `examples/50_hopper_gemm_with_epilogue_swizzle/` (built in place by `artifacts/experience/api-probes/gemm/run_fused.sh`) |

To rebuild any topic's binary: `cd artifacts/experience/<api-probes|hw-probes>/<topic> && bash build.sh && bash run.sh`.

## Current contents

```
wiki/nvidia/code-walkthroughs/cutlass-cute/
├── README.md                 (this file)
├── example48-hopper-warp-specialized-gemm/
│                              (ex48 TMA + WGMMA + cooperative WS mainloop)
├── wgmma-atom-decoding/      (CUTLASS WGMMA atom decoding + skeleton)
├── gemm-aligned/             (aligned-GEMM template params + tuning)
├── gemm-tail/                (tail-handling template selection + tuning)
├── persistent-kernel/        (persistent-schedule selection + tuning)
└── gemm-fused/               (fused epilogue / prologue patterns + tuning)
```
