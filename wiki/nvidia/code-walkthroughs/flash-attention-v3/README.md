---
id: code-flash-attention-v3-README
type: code-walkthrough
vendor: nvidia
title: Readme
upstream_repo: NVIDIA/flash-attention-v3
architectures:
- sm90
- sm90a
languages:
- ptx
- cuda-cpp
- cute-dsl
- python
hardware_features:
- wgmma
- tma
- fp8
techniques:
- pipeline-stages
- kernel-fusion
- shared-memory-optimization
- software-exp
- conditional-rescaling
kernel_types:
- gemm
- attention
- flash-attention
- fused-kernel
- quantization
- mla
- decode
confidence: inferred
tags:
- wgmma
- tma
- fp8
- pipeline-stages
- kernel-fusion
- shared-memory-optimization
- software-exp
- conditional-rescaling
- gemm
- attention
- flash-attention
- fused-kernel
- quantization
- mla
- decode
- ptx
- cuda-cpp
- cute-dsl
- python
---
# FlashAttention v3 (Tri Dao et al.)

## Source

- Repository: https://github.com/Dao-AILab/flash-attention
- Local clone: `{{FLASH_ATTENTION_REPO_REF}}`
- Commit: `bbda031f1cd1adf1a57a3b3e8cdc3db2b54d3994`
- License: BSD-3-Clause
- MANIFEST registration: `corpus/MANIFEST.yaml` → `source-code/flash-attention`

## What it is

FlashAttention v3 is Tri Dao's reference implementation of fused multi-head attention with online softmax (the "FlashAttention" algorithm). It supports:
- Forward and backward passes
- Hopper (sm_90a) optimized kernels using wgmma and TMA
- Ampere (sm_80) fallback kernels using mma.sync
- FP16, BF16, and FP8 input types
- Causal masking, variable-length (varlen), split-KV, and MLA (Multi-head Latent Attention) decode
- Multi-head, grouped-query, and multi-query attention patterns

## Key source files

| File | Role |
|---|---|
| `hopper/flash_fwd_kernel_sm90.h` | Hopper forward kernel (wgmma-based) |
| `hopper/flash_bwd_kernel_sm90.h` | Hopper backward kernel |
| `hopper/flash_api.cpp` | C++ API entry point |
| `hopper/block.h` | Block/tile configuration |
| `flash_attn/flash_attn_interface.py` | Python API |
| `hopper/epilogue_fwd.hpp` | Forward epilogue (writeback + rescale) |
| `hopper/epilogue_bwd.hpp` | Backward epilogue |

## Algorithmic approach

FlashAttention v3 implements the same core algorithm as FlashAttention-2:

1. **Tiled online softmax**: Process KV tiles sequentially, maintaining a running max and running sum per Q-row. Rescale the output accumulator whenever the max updates.
2. **IO-aware tiling**: Tile sizes are chosen to minimize HBM reads by keeping Q/O tiles in SRAM (registers + smem) while streaming K/V tiles from HBM.
3. **Fused kernel**: The entire attention computation (Q@K^T, softmax, P@V) runs in a single kernel launch, avoiding materialization of the S or P matrices in HBM.

### Comparison with cutlass FMHA (`examples/88_hopper_fmha`)

| Aspect | FlashAttention v3 | cutlass FMHA |
|---|---|---|
| Origin | Tri Dao (academic/industry) | NVIDIA (in-tree cutlass) |
| Template system | Hand-rolled C++ templates | cutlass `CollectiveBuilder` + `TiledMMA` |
| Hopper path | Direct wgmma/TMA inline PTX via cutlass cute atoms | Full cutlass collective/kernel/device stack |
| Ampere fallback | Yes (sm_80 kernels) | No (Hopper only in example 88) |
| Variable-length | Yes (varlen API) | Partial (batch-only) |
| FP8 | Yes | Yes |
| MLA decode | Yes (specialized kernel) | No |
| Backward | Yes | Yes |

### Comparison with MVP minimal attention (`wiki/nvidia/foundations/compute/attention/mvp-minimal`)

| Aspect | FlashAttention v3 | MVP minimal |
|---|---|---|
| Purpose | Production performance | Correctness reference + learning |
| Code complexity | ~10K lines | ~200 lines |
| Tiling | Multiple tuned tile sizes | Fixed BLOCK_M=64, BLOCK_N=64 |
| Pipeline | Multi-stage TMA pipeline | Single-buffer |
| Performance | Optimized for production (no KB measurement yet) | Correctness reference |
| Dependencies | cutlass cute atoms | None (raw CUDA) |

## Tuning parameters

See `tuning.md` for the parameter space.

## Status

- Registered in `corpus/MANIFEST.yaml` with commit hash and local path.
- No measured artifacts from this KB's H200 infrastructure yet. If measured, artifacts would follow the standard layout at `artifacts/experience/api-probes/attention/`.
