---
id: code-cutlass-cute-README
type: code-walkthrough
vendor: nvidia
title: Readme
upstream_repo: NVIDIA/cutlass-cute
---
# cutlass FMHA Collective Layer

This directory documents the collective-builder layer that implements the core FMHA algorithm in cutlass `examples/88_hopper_fmha/collective/`.

## Source location

```
{{CUTLASS_REPO_REF}}/examples/88_hopper_fmha/collective/
├── fmha_collective_tma_warpspecialized.hpp   # Primary forward collective
├── fmha_collective_bwd_tma_warpspecialized.hpp  # Backward collective
├── fmha_collective_load.hpp                  # TMA load helpers
├── fmha_collective_softmax.hpp               # Online softmax implementation
├── fmha_collective_tma.hpp                   # TMA descriptor helpers
├── fmha_common.hpp                           # Common types and utilities
├── fmha_epilogue.hpp                         # Output writeback
├── fmha_epilogue_bwd.hpp                     # Backward epilogue
└── fmha_fusion.hpp                           # Fusion interface (causal, etc.)
```

Upstream commit: `cutlass@f74fea9c`.

## API surface

### Entry types

- `FmhaCollectiveTmaWarpSpecialized<ElementQK, ElementVO, kHeadDim, Options...>` — the primary forward collective. Uses TMA for data movement and wgmma for compute.
- `FmhaCollectiveBwdTmaWarpSpecialized` — the backward-pass collective.

### Builder dispatch

The collective is instantiated by `FmhaKernelBuilder` in `kernel/fmha_kernel_builder.hpp`. The builder resolves compile-time options (see `fmha_options.hpp`) and passes them through as template parameters.

### Warp-specialization variants

The forward collective uses a **producer/consumer** warp-specialization pattern:

1. **Producer warpgroup** (warpgroup 0): Issues TMA loads for K and V tiles into shared memory pipeline stages. Manages mbarrier-based pipeline synchronization. Optionally loads Q separately (`kLoadsQSeparately`).

2. **Consumer warpgroup(s)** (warpgroup 1+): Execute the attention computation:
   - `S = Q @ K^T` via wgmma (shared-memory Q × shared-memory K^T)
   - Online softmax (per-row max tracking, exp, rescale)
   - `O += P @ V` via wgmma (register-resident P × shared-memory V)

This mirrors the GEMM warp-specialization pattern from `examples/48_hopper_warp_specialized_gemm/`.

### Core methods

| Method | Role |
|---|---|
| `load(pipeline, ...)` | Producer: TMA loads K/V tiles into smem stages |
| `mma(pipeline, ...)` | Consumer: wgmma S=Q@K^T, softmax, wgmma O=P@V |
| `epilogue(...)` | Final normalization and global store |

## Differences from GEMM collective

| Aspect | GEMM collective | FMHA collective |
|---|---|---|
| Inner loop | K-dimension tiles | Sequence-dimension (KV) tiles |
| Softmax | None | Online softmax with accumulator rescaling |
| Output accumulator | Single, grows with K tiles | Rescaled every KV tile (FlashAttention-2) |
| TMA loads per iteration | A + B tiles | K + V tiles (Q loaded once, reused) |
| Epilogue | Direct store or fused op | Normalize by softmax denominator + store |

## Builder-level vs kernel-level knobs

See `tuning.md` for the full table. Key distinction:

- **Builder-level knobs** (resolved by `FmhaKernelBuilder`): `kNumMmaWarpGroups`, `kIsPersistent`, `kStagesQ`, `kStagesKV`, `kClusterM`. These determine the collective's template instantiation.
- **Kernel-level knobs** (inside the collective): Tile shapes (BLOCK_M, BLOCK_N derived from warpgroup count and head dim), pipeline depth, softmax precision.

## H200 measured collective datapoint

See [80-experience/api-probes/attention/2026-05-08-cutlass-fmha-collective.md](../../../80-experience/api-probes/attention/2026-05-08-cutlass-fmha-collective.md).

Configuration: B=2 H=16 Q=2048 K=2048 D=128, causal mask, kNumMmaWarpGroups=2 (BLOCK_M=128).

| Config | TFLOPS/s | Correctness |
|--------|----------|-------------|
| cooperative 128x128x128 | 307.1 | OK |
| ping-pong 128x128x128 | 348.5 | OK |

Measured on H200-SXM, sm_90a, CUDA 12.9.

## How this fits in the KB

- Detailed per-knob documentation: `60-code/cutlass-cute/attention-fmha-example/tuning.md`
- Code flow skeleton: `60-code/cutlass-cute/attention-fmha-example/fmha_skeleton.md`
- Hand-built minimal attention kernel (no cutlass dependency): `30-skill/compute/attention/mvp-minimal/skill.md`
