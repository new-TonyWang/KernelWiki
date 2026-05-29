# GPU Kernel Optimization Knowledge Base

> Multi-vendor knowledge base for GPU kernel optimization. Currently covering NVIDIA Blackwell (SM100), Hopper (SM90), and general CUDA.
> Optimized for LLM agent retrieval. See [CLAUDE.md](CLAUDE.md) for schema and conventions.
> **For Claude Code agents**: this repository is a Claude Code skill — see [SKILL.md](SKILL.md).

## Recommended Query Tools (for LLM agents)

```bash
python3 scripts/query.py "<natural language>" [--tag <t>] [--type <kernel|technique|pr|...>]
python3 scripts/get_page.py <page-id-or-path> [--follow-sources]
python3 scripts/grep_wiki.py "<regex>" [--only wiki|sources]
```

See [references/examples.md](references/examples.md) for 10 worked query patterns.

## Quick Navigation

| I want to... | Go to |
|---|---|
| Fix a performance problem | [queries/by-problem.md](queries/by-problem.md) |
| Learn a specific technique | [queries/by-technique.md](queries/by-technique.md) |
| Use a hardware feature | [queries/by-hardware-feature.md](queries/by-hardware-feature.md) |
| See what a repo contributed | [queries/by-repo.md](queries/by-repo.md) |
| Write a specific kernel type | [queries/by-kernel-type.md](queries/by-kernel-type.md) |
| Use a specific language/DSL | [queries/by-language.md](queries/by-language.md) |
| Browse by vendor | [queries/by-vendor.md](queries/by-vendor.md) |
| Learn a foundational skill | `wiki/nvidia/foundations/` |
| Route an operator | `wiki/nvidia/operator-routing/` |
| Look up an API | `wiki/nvidia/api-definitions/` |

## Hardware Features

- [hw-tcgen05-mma](wiki/nvidia/hardware/tcgen05-mma.md) — Blackwell MMA instruction (replaces wgmma)
- [hw-tmem](wiki/nvidia/hardware/tmem.md) — Tensor Memory (256KB dedicated accumulator storage)
- [hw-clc](wiki/nvidia/hardware/clc.md) — Cluster Launch Control (dynamic tile scheduling)
- [hw-tma](wiki/nvidia/hardware/tma.md) — Tensor Memory Accelerator (async bulk loads)
- [hw-2sm-cooperative](wiki/nvidia/hardware/2sm-cooperative.md) — Two-SM cooperative MMA
- [hw-nvfp4](wiki/nvidia/hardware/nvfp4.md) — NVFP4 and block-scaled narrow precision
- [hw-pdl-gdc](wiki/nvidia/hardware/pdl-gdc.md) — Programmatic Dependent Launch / Grid Dependency Control

## Optimization Techniques

- [technique-warp-specialization](wiki/nvidia/techniques/warp-specialization.md) — Warp role assignment
- [technique-persistent-kernels](wiki/nvidia/techniques/persistent-kernels.md) — Persistent kernel patterns with CLC
- [technique-swizzling](wiki/nvidia/techniques/swizzling.md) — Shared memory swizzling
- [technique-pipeline-stages](wiki/nvidia/techniques/pipeline-stages.md) — Software pipelining
- [technique-epilogue-fusion](wiki/nvidia/techniques/epilogue-fusion.md) — Fusing epilogue with mainloop
- [technique-tile-scheduling](wiki/nvidia/techniques/tile-scheduling.md) — Tile scheduling strategies
- [technique-double-buffering](wiki/nvidia/techniques/double-buffering.md) — Double/multi-buffering
- [technique-software-exp](wiki/nvidia/techniques/software-exp.md) — Software-emulated exponential
- [technique-fine-grained-quant](wiki/nvidia/techniques/fine-grained-quantization.md) — Fine-grained FP8/FP4 quantization
- [technique-vectorized-loads](wiki/nvidia/techniques/vectorized-loads.md) — Wide vectorized loads and cache policies

## Kernel Case Studies

- [kernel-flash-attention-4](wiki/nvidia/kernels/flash-attention-4.md) — FlashAttention-4 (1605 TFLOPS on B200)
- [kernel-deepgemm](wiki/nvidia/kernels/deepgemm.md) — DeepGEMM FP8 GEMM (1550 TFLOPS on H800)
- [kernel-flashmla](wiki/nvidia/kernels/flashmla.md) — FlashMLA sparse/dense MLA decoding
- [kernel-nsa](wiki/nvidia/kernels/nsa.md) — Native Sparse Attention (9x fwd speedup)
- [kernel-gated-delta-net](wiki/nvidia/kernels/gated-delta-net.md) — Gated Delta Net linear attention
- [kernel-nvfp4-gemm](wiki/nvidia/kernels/nvfp4-gemm.md) — NVFP4 GEMM from GPU Mode hackathon
- [kernel-nvfp4-gemv](wiki/nvidia/kernels/nvfp4-gemv.md) — NVFP4 batched GEMV optimization
- [kernel-grouped-gemm](wiki/nvidia/kernels/grouped-gemm.md) — Grouped GEMM for MoE
- [kernel-fused-moe](wiki/nvidia/kernels/fused-moe.md) — Fused MoE with FP8

## Problem → Solution Patterns

- [pattern-low-sm-utilization](wiki/nvidia/patterns/low-sm-utilization.md) — SM utilization is low
- [pattern-memory-bound](wiki/nvidia/patterns/memory-bound.md) — Kernel is memory bandwidth limited
- [pattern-register-pressure](wiki/nvidia/patterns/register-pressure.md) — Too many registers → low occupancy
- [pattern-compute-bound](wiki/nvidia/patterns/compute-bound.md) — Not reaching peak FLOPS
- [pattern-tail-effect](wiki/nvidia/patterns/tail-effect.md) — Last wave underutilizes GPU

## Languages & DSLs

- [lang-cute-dsl](wiki/nvidia/languages/cute-dsl.md) — CuTe DSL for Blackwell
- [lang-cuda-cpp](wiki/nvidia/languages/cuda-cpp.md) — CUDA C++ with PTX inline
- [lang-ptx](wiki/nvidia/languages/ptx-sm100.md) — PTX instructions for SM100
- [lang-triton](wiki/nvidia/languages/triton-blackwell.md) — Triton on Blackwell

## Migration Guides

- [migration-wgmma-to-tcgen05](wiki/nvidia/migration/wgmma-to-tcgen05.md) — Hopper wgmma → Blackwell tcgen05
- [migration-register-to-tmem](wiki/nvidia/migration/register-to-tmem.md) — Register accumulators → TMEM

## Source Repositories

| Repository | Focus |
|---|---|
| [NVIDIA/cutlass](queries/by-repo.md#nvidiacutlass) | CUTLASS 4.x Blackwell support |
| [sgl-project/sglang](queries/by-repo.md#sgl-projectsglang) | SGLang Blackwell integration |
| [vllm-project/vllm](queries/by-repo.md#vllm-projectvllm) | vLLM Blackwell support |
| [flashinfer-ai/flashinfer](queries/by-repo.md#flashinfer-aiflashinfer) | FlashInfer Blackwell kernels |
| [pytorch/pytorch](queries/by-repo.md#pytorchpytorch) | PyTorch/Inductor Blackwell |

## Competitions

- [GPU Mode NVFP4 Hackathon](sources/contests/gpu-mode-nvfp4/) — 4 NVFP4 kernel challenges on B200
- [FlashInfer MLSys 2026](sources/contests/flashinfer-mlsys26/) — MoE, Sparse Attention, GatedDeltaNet
