---
title: 'ldmatrix: Shared-memory to register matrix load for tensor cores'
status: draft
evidence_level: spec
applies_to_pattern_class:
- tensor-core
applies_to_ops:
- gemm
- attention
requires_sm: '>=7.5'
requires_features:
- ldmatrix
single_kernel_useful: true
source:
- path: spec
  anchor: Reference
related_apis: []
related_skills:
- mma-sync-ptx
- wgmma-ptx
upstream_repo: cutlass@f74fea9c
id: skill-ldmatrix-ptx
type: skill
vendor: nvidia
tags:
- cuda-cpp
- wgmma
- tma
- ldmatrix
- shared-memory-optimization
- gemm
- attention
- ptx
- cute-dsl
applies_to:
- general
source_refs:
- source_id: source-code/cutlass
  path: include/cute/atom/copy_traits_sm75.hpp
  anchor: SM75_U32x1_LDSM_N .. SM75_U16x8_LDSM_T
- source_id: source-code/cutlass
  path: include/cute/arch/copy_sm75.hpp
  anchor: ldmatrix PTX wrappers
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/parallel-thread-execution/cuda_parallel-thread-execution_index.html.md
  anchor: L27850-L27890
architectures:
- sm90
- sm90a
languages:
- ptx
- cuda-cpp
- cute-dsl
hardware_features:
- wgmma
- tma
- ldmatrix
techniques:
- shared-memory-optimization
kernel_types:
- gemm
- attention
confidence: source-reported
---
# ldmatrix — Shared-memory to register matrix load for tensor cores

## What it is

`ldmatrix` is a warp-level instruction (sm_75+, Turing and later) that loads 8x8 matrix tiles from shared memory into registers, distributing fragments across the 32 threads of a warp in the exact layout that `mma.sync` expects. It replaces the manual "load from smem + cross-lane shuffle" pattern with a single hardware instruction.

The instruction comes in two variants:
- **Non-transposed** (`ldmatrix.sync.aligned.x{1,2,4}.m8n8.shared.b16`): loads 1, 2, or 4 contiguous 8x8 tiles without transposition.
- **Transposed** (`ldmatrix.sync.aligned.x{1,2,4}.m8n8.trans.shared.b16`): loads with an implicit transpose, useful for the B operand in TN layout.

In cutlass-cute, these are wrapped as:
- `SM75_U32x1_LDSM_N`, `SM75_U32x2_LDSM_N`, `SM75_U32x4_LDSM_N` (non-transposed)
- `SM75_U16x2_LDSM_T`, `SM75_U16x4_LDSM_T`, `SM75_U16x8_LDSM_T` (transposed)

A companion instruction, `movmatrix` (`SM75_U32x1_MOVM_T`), transposes a matrix already in registers.

## PTX mnemonic template

```asm
// Load 4 × 8x8 tiles (the most common variant for m16n8k16):
ldmatrix.sync.aligned.x4.m8n8.shared.b16
    {%r0, %r1, %r2, %r3},   // 4 uint32 destination registers per thread
    [smem_ptr];               // shared memory address (each thread provides its row's address)
```

Each thread in the warp provides the smem address for its own row's 8-element strip. The hardware performs a cross-lane gather so that the resulting register contents match what `mma.sync` expects.

## Variants

| Variant | PTX suffix | Tiles loaded | Regs/thread | cute atom |
|---|---|---|---|---|
| x1 non-transposed | `.x1.m8n8` | 1 × 8x8 | 1 × uint32 | `SM75_U32x1_LDSM_N` |
| x2 non-transposed | `.x2.m8n8` | 2 × 8x8 | 2 × uint32 | `SM75_U32x2_LDSM_N` |
| x4 non-transposed | `.x4.m8n8` | 4 × 8x8 | 4 × uint32 | `SM75_U32x4_LDSM_N` |
| x2 transposed | `.x2.m8n8.trans` | 2 × 8x8 (transposed) | 2 × uint16 pairs | `SM75_U16x2_LDSM_T` |
| x4 transposed | `.x4.m8n8.trans` | 4 × 8x8 (transposed) | 4 × uint16 pairs | `SM75_U16x4_LDSM_T` |
| x8 transposed | `.x8.m8n8.trans` | 8 × 8x8 (transposed) | 8 × uint16 pairs | `SM75_U16x8_LDSM_T` |

## When to use it

- Loading A/B operands for `mma.sync` on Ampere (sm_80+). ldmatrix handles the smem→register layout mapping that mma.sync requires.
- On Turing (sm_75) / Volta for older MMA instructions.

## When NOT to use it

- On Hopper with wgmma in SS mode: wgmma reads A and B directly from smem via descriptors — no ldmatrix needed. The RS variant of wgmma uses register-resident A but still does not use ldmatrix (the agent must fill the A fragment layout manually or via TMA → register staging).
- When the smem layout is incompatible with ldmatrix's 16-byte alignment requirement.

## How it connects to the rest of the KB

- Required companion to `wiki/nvidia/hardware/mma-sync-ptx/` — mma.sync expects fragments in ldmatrix's output layout.
- Superseded on Hopper by TMA + smem descriptors for the wgmma path.
- cutlass-cute wraps ldmatrix in `Copy_Traits<SM75_U32x{1,2,4}_LDSM_N>` (see `copy_traits_sm75.hpp`).
