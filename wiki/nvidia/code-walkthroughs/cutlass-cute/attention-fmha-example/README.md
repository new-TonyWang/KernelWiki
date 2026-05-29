---
id: code-cutlass-cute-attention-fmha-example-README
type: code-walkthrough
vendor: nvidia
title: Readme
upstream_repo: NVIDIA/cutlass-cute
---
# cutlass example 88: Hopper FMHA

This directory documents the extraction of cutlass `examples/88_hopper_fmha/` — the primary sm_90a flash multi-head attention implementation in the cutlass repository.

## Source location

```
{{CUTLASS_REPO_REF}}/examples/88_hopper_fmha/
├── 88_hopper_fmha.cu          # Runner: CLI parsing, reference comparison, timing
├── CMakeLists.txt
├── README.md                  # NVIDIA's own documentation
├── collective/                # FMHA collective layer (16 headers)
│   ├── fmha_collective_tma_warpspecialized.hpp   # Primary forward collective
│   ├── fmha_collective_bwd_tma_warpspecialized.hpp
│   ├── fmha_collective_load.hpp
│   ├── fmha_collective_softmax.hpp
│   ├── fmha_collective_tma.hpp
│   ├── fmha_common.hpp
│   ├── fmha_epilogue.hpp
│   ├── fmha_epilogue_bwd.hpp
│   └── fmha_fusion.hpp
├── kernel/                    # Kernel layer: builder, scheduler, options
│   ├── fmha_kernel_builder.hpp       # Dispatches to fmha_kernel_tma_warpspecialized
│   ├── fmha_kernel_tma_warpspecialized.hpp
│   ├── fmha_kernel_tma.hpp
│   ├── fmha_options.hpp              # Tag-based option system
│   ├── fmha_tile_scheduler.hpp
│   ├── fmha_kernel_bwd_convert.hpp
│   └── fmha_kernel_bwd_sum_OdO.hpp
├── device/                    # Device layer: workspace orchestration
│   ├── device_universal.hpp
│   └── fmha_device_bwd.hpp
└── reference/                 # CPU/GPU reference implementations for verification
```

Upstream commit: `cutlass@f74fea9c`.

## Architecture overview

The example follows cutlass's standard 4-layer architecture:

1. **Runner** (`88_hopper_fmha.cu`): Parses command line (`--b`, `--h`, `--q`, `--k`, `--d`), allocates tensors, runs the kernel, compares against reference, reports timing.

2. **Device layer** (`device/device_universal.hpp`): Orchestrates workspace allocation, kernel launch configuration, and multi-kernel scheduling for backward pass.

3. **Kernel layer** (`kernel/fmha_kernel_builder.hpp` → `fmha_kernel_tma_warpspecialized.hpp`): The kernel entry point. `FmhaKernelBuilder` dispatches to the warp-specialized kernel template based on options. The kernel template manages the main loop, CTA scheduling, and interfaces with the collective layer.

4. **Collective layer** (`collective/fmha_collective_tma_warpspecialized.hpp`): Contains the core FMHA logic — TMA-based Q/K/V loads, wgmma-based S=Q@K^T and O=P@V tiles, online softmax, and epilogue writeback. This is the most complex layer and is documented separately at `wiki/nvidia/code-walkthroughs/cutlass-cute/attention-fmha-collective/`.

## Key types and entry points

- **Runner entry**: `run()` in `88_hopper_fmha.cu` calls `device.run(args, workspace, stream)`.
- **Kernel builder**: `cutlass::fmha::kernel::FmhaKernelBuilder<Options...>` in `fmha_kernel_builder.hpp`. This is the primary compile-time dispatch point.
- **Options system**: `cutlass::fmha::kernel::Tag` enum in `fmha_options.hpp` defines compile-time knobs: `kIsPersistent`, `kNumMmaWarpGroups`, `kStagesQ`, `kStagesKV`, `kBlocksPerSM`, `kClusterM`, `kAccQK`, etc.
- **Forward collective**: `cutlass::fmha::collective::FmhaCollectiveTmaWarpSpecialized` in `fmha_collective_tma_warpspecialized.hpp`.

## Supported features

- Forward pass (fp16, bf16, fp8)
- Backward pass (fp16, bf16)
- Causal masking
- Warp-specialized (producer/consumer) mainloop
- Persistent kernel scheduling
- Multi-stage TMA pipeline (configurable via `kStagesQ`, `kStagesKV`)

## sm_80 fallback context: `41_fused_multi_head_attention`

Cutlass also ships `examples/41_fused_multi_head_attention/`, an older FMHA example targeting sm_80 (Ampere) using `mma.sync` instead of wgmma. This example:
- Uses the pre-Hopper collective architecture (no TMA, no warp-specialization)
- Supports fp16 and bf16 forward pass with causal masking
- Source: `{{CUTLASS_REPO_REF}}/examples/41_fused_multi_head_attention/`

It is not the primary extraction target (sm_90a/H200 requires `88_hopper_fmha`) but provides architectural context for understanding how FMHA evolved from Ampere to Hopper.

## How to build and run

Repo-local extracted wrapper: `sources/experience/api-probes/attention/artifacts/cutlass_88_hopper_fmha.cu`
Build script: `sources/experience/api-probes/attention/artifacts/build_cutlass_fmha.sh`

```bash
# Set CUTLASS_DIR to your cutlass checkout (default: {{CUTLASS_REPO_REF}})
cd sources/experience/api-probes/attention/artifacts/
CUTLASS_DIR=/path/to/cutlass bash build_cutlass_fmha.sh
./88_hopper_fmha --b=2 --h=16 --q=1024 --k=1024 --d=128 --verify
```

The wrapper includes the upstream source via `#include "88_hopper_fmha.cu"` and requires the cutlass include tree at build time.

## H200 measured results

See [sources/experience/api-probes/attention/2026-05-08-cutlass-88-hopper-fmha.md](../../../sources/experience/api-probes/attention/2026-05-08-cutlass-88-hopper-fmha.md).

| Config | Shape | TFLOPS/s | Correctness |
|--------|-------|----------|-------------|
| cooperative 128x128x128 | B2 H16 S1024 D128 full | 366.6 | OK |
| ping-pong 128x128x128 | B2 H16 S1024 D128 full | 376.1 | OK |

## Docs

1. Read `tuning.md` for the compile-time and runtime knobs.
2. Read `fmha_skeleton.md` for a guided tour of the code flow.
