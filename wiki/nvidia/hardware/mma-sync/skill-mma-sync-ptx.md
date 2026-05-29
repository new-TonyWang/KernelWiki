---
title: 'mma.sync: Warp-level MMA on Ampere+ via PTX'
status: draft
evidence_level: spec
applies_to_pattern_class:
- tensor-core
applies_to_ops:
- gemm
- attention
requires_sm: '>=8.0'
requires_features:
- mma.sync
- ldmatrix
single_kernel_useful: true
source:
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/parallel-thread-execution/cuda_parallel-thread-execution_index.html.md
  anchor: L28200-L28250
  excerpt: 9.7.14. Matrix Multiply-Accumulate Operation using mma.sync instruction
    — The mma operation is performed by all threads in a warp, and is therefore a
    warp-level instruction.
related_apis: []
related_skills:
- wgmma
- wgmma-ptx
- ldmatrix-ptx
upstream_repo: cutlass@f74fea9c
id: skill-mma-sync-ptx
type: skill
vendor: nvidia
tags:
- cuda-cpp
applies_to:
- general
source_refs:
- source_id: source-code/cutlass
  path: include/cute/atom/mma_traits_sm80.hpp
  anchor: SM80_16x8x8_F32F16F16F32_TN .. SM80_16x8x32_S32S8S8S32_TN
- source_id: source-code/cutlass
  path: include/cute/arch/mma_sm80.hpp
  anchor: SM80_16x8x8_F32F16F16F32_TN inline PTX wrappers
---
# mma.sync — Warp-level MMA on Ampere+ via PTX

## What it is

`mma.sync` is the Ampere-era (sm_80+) warp-level matrix-multiply-accumulate instruction. Unlike Hopper's wgmma (which operates at warpgroup granularity = 128 threads), `mma.sync` is a **single-warp** (32 threads) synchronous instruction. Each `mma.sync` call computes a small tile (typically m16n8k8 for fp16/bf16 or m16n8k16 for the double-K variant) and returns immediately — there is no asynchronous issue/commit/wait protocol.

In cutlass-cute the atoms live in `MMA_Traits<SM80_16x8x{K}_{Acc}{A}{B}{Acc}_TN>` structs inside `mma_traits_sm80.hpp`.

## Atom shapes supported on Ampere (sm_80+)

| Family | M | N | K | Acc dtype | Input dtypes | cute atom struct |
|---|---|---|---|---|---|---|
| F16/BF16 → F16 | 16 | 8 | 8 | f16 | f16×f16 | `SM80_16x8x8_F16F16F16F16_TN` |
| F16/BF16 → F16 (double-K) | 16 | 8 | 16 | f16 | f16×f16 | `SM80_16x8x16_F16F16F16F16_TN` |
| F16 → F32 | 16 | 8 | 8 | f32 | f16×f16 | `SM80_16x8x8_F32F16F16F32_TN` |
| F16 → F32 (double-K) | 16 | 8 | 16 | f32 | f16×f16 | `SM80_16x8x16_F32F16F16F32_TN` |
| BF16 → F32 | 16 | 8 | 8 | f32 | bf16×bf16 | `SM80_16x8x8_F32BF16BF16F32_TN` |
| BF16 → F32 (double-K) | 16 | 8 | 16 | f32 | bf16×bf16 | `SM80_16x8x16_F32BF16BF16F32_TN` |
| TF32 → F32 | 16 | 8 | 4 | f32 | tf32×tf32 | `SM80_16x8x4_F32TF32TF32F32_TN` |
| TF32 → F32 (double-K) | 16 | 8 | 8 | f32 | tf32×tf32 | `SM80_16x8x8_F32TF32TF32F32_TN` |
| F64 → F64 | 8 | 8 | 4 | f64 | f64×f64 | `SM80_8x8x4_F64F64F64F64_TN` |
| S8 → S32 | 8/16 | 8 | 16/32 | s32 | s8×s8 | `SM80_8x8x16_S32S8S8S32_TN` / `SM80_16x8x16_S32S8S8S32_TN` / `SM80_16x8x32_S32S8S8S32_TN` |
| Mixed S8/U8 → S32 | 8/16 | 8 | 16/32 | s32 | s8×u8 / u8×s8 | `SM80_*_S32S8U8S32_TN` / `SM80_*_S32U8S8S32_TN` |

## PTX mnemonic template

```asm
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    {%f0, %f1, %f2, %f3},       // 4 f32 accumulators per thread
    {%r0, %r1, %r2, %r3},       // 4 uint32 registers holding A fragment (8 fp16 values packed)
    {%r4, %r5},                  // 2 uint32 registers holding B fragment (4 fp16 values packed)
    {%f4, %f5, %f6, %f7};       // 4 f32 source accumulators (C for D = A*B + C)
```

Key differences from wgmma:
- **Synchronous**: no fence/commit/wait; result is in registers immediately after the instruction retires.
- **Per-warp**: 32 threads, not 128 (warpgroup).
- **Register-resident operands**: A and B are in registers (typically loaded via `ldmatrix`), not in smem descriptors.
- **Multiple issues needed**: to cover an M=64 tile a kernel must issue 4 consecutive m16n8k16 calls (one per 16-row slice), versus one wgmma.mma_async for the whole M=64 tile on Hopper.

## When to use it

- Targets sm_80 / sm_86 / sm_89 (Ampere / Ada Lovelace) where wgmma is not available.
- Porting legacy Ampere kernels to understand the hardware before migrating to Hopper wgmma.
- Operators on Hopper that cannot use wgmma (e.g. blocks with fewer than 128 threads).

## When NOT to use it

- On Hopper (sm_90a): use wgmma instead. wgmma covers the entire warpgroup in one instruction and pairs with TMA for higher throughput.
- When you can use cutlass's `TiledMMA<MMA_Atom<SM80_*>>` abstraction instead of raw PTX.

## How it connects to the rest of the KB

- Predecessor to `wiki/nvidia/hardware/wgmma-ptx/` — wgmma replaces mma.sync on Hopper with warpgroup-level granularity and an async issue model.
- Pairs with `wiki/nvidia/hardware/ldmatrix-ptx/` — ldmatrix loads A/B fragments from smem into registers in the layout mma.sync expects.
- cutlass-cute wraps mma.sync in `MMA_Atom<MMA_Traits<SM80_*>>` (see `mma_traits_sm80.hpp`).
