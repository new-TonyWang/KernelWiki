---
title: 'tcgen05: Blackwell 5th-generation Tensor Core via PTX'
status: draft
evidence_level: spec
not_measured_on_target: true
applies_to_pattern_class:
- tensor-core
applies_to_ops:
- gemm
- attention
requires_sm: '>=10.0'
requires_features:
- tcgen05
single_kernel_useful: true
source:
- path: whitepapers/gpu-wite-paper/BlackWell/nvidia-blackwell-architecture-technical-brief.pdf
  anchor: 5th Generation Tensor Cores
  excerpt: Blackwell introduces 5th-generation Tensor Cores with 2x throughput over
    Hopper and new micro-tensor scaling (MX formats).
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/parallel-thread-execution/cuda_parallel-thread-execution_index.html.md
  anchor: tcgen05 instruction family
related_apis: []
related_skills:
- wgmma-ptx
- wgmma
upstream_repo: cutlass@f74fea9c
id: skill-tcgen05-ptx
type: skill
vendor: nvidia
tags:
- cuda-cpp
applies_to:
- general
source_refs:
- source_id: source-code/cutlass
  path: include/cute/arch/copy_sm100.hpp
  anchor: tcgen05.cp and tcgen05.ld inline PTX wrappers
- source_id: source-code/cutlass
  path: include/cute/atom/mma_traits_sm100.hpp
  anchor: SM100_MMA_TF32_SS, SM100_MMA_F16BF16_SS, SM100_MMA_*_TS, SM100_MMA_*_SCALED
---
# tcgen05 — Blackwell 5th-generation Tensor Core via PTX

**This entry is spec-only. H200 (sm_90a) cannot run Blackwell instructions. No measured artifacts exist; all content is derived from the PTX ISA specification, the Blackwell architecture whitepaper, and cutlass source inspection.**

## What it is

`tcgen05` is the Blackwell-generation (sm_100+) tensor core instruction family that replaces Hopper's `wgmma` as the primary matrix-multiply interface. The name stands for "tensor core generation 05" (5th generation, counting from Volta's 1st gen through Turing/Ampere/Hopper).

Key differences from Hopper's wgmma:

- **2x peak throughput** over Hopper at the same clock (Blackwell whitepaper).
- **New copy path**: `tcgen05.cp` instructions load matrix tiles from smem into an opaque "tcgen05 register file" that is separate from the standard register file. This replaces the wgmma smem-descriptor path.
- **New load path**: `tcgen05.ld` instructions move data from the tcgen05 register file back into standard registers.
- **Micro-tensor scaling**: SCALED variants (`SM100_MMA_F16BF16_SS_SCALED`, `SM100_MMA_TF32_TS_SCALED`) support per-element or per-block scaling for MX-format quantized inference.
- **Tile shapes**: larger base tiles (128x256b, 128x128b per CTA group), with warp-cooperative modes (`warpx2`, `warpx4`).

## PTX instruction families

### tcgen05.cp — smem → tcgen05 register file

```asm
tcgen05.cp.cta_group::1.128x256b [tcgen05_addr], smem_desc;
tcgen05.cp.cta_group::2.128x256b [tcgen05_addr], smem_desc;
tcgen05.cp.cta_group::1.128x128b [tcgen05_addr], smem_desc;
tcgen05.cp.cta_group::1.4x256b   [tcgen05_addr], smem_desc;
tcgen05.cp.cta_group::1.32x128b.warpx4 [tcgen05_addr], smem_desc;
tcgen05.cp.cta_group::1.64x128b.warpx2::02_13 [tcgen05_addr], smem_desc;
```

### tcgen05.ld — tcgen05 register file → standard registers

```asm
tcgen05.ld.sync.aligned.16x256b.x1.b32 {regs...}, [tcgen05_addr];
tcgen05.ld.sync.aligned.16x256b.x2.b32 {regs...}, [tcgen05_addr];
tcgen05.ld.sync.aligned.16x256b.x4.b32 {regs...}, [tcgen05_addr];
tcgen05.ld.sync.aligned.16x256b.x8.b32 {regs...}, [tcgen05_addr];
```

The `.pack::16b` variants pack 16-bit elements.

### tcgen05.mma — the MMA itself

The MMA instruction is issued via cutlass's `SM100_MMA_*` traits which dispatch to the underlying PTX. The atom shapes include:

| cutlass trait | A/B dtype | Acc dtype | Layout |
|---|---|---|---|
| `SM100_MMA_TF32_SS` | tf32 (32-bit) | f32 | SS (both from smem) |
| `SM100_MMA_F16BF16_SS` | f16/bf16 (16-bit) | f32 | SS |
| `SM100_MMA_TF32_TS` | tf32 | f32 | TS (A from tcgen05 regs, B from smem) |
| `SM100_MMA_F16BF16_TS` | f16/bf16 | f32 | TS |
| `SM100_MMA_F16BF16_SS_SCALED` | f16/bf16 | f32 | SS + per-element scaling |
| `SM100_MMA_TF32_TS_SCALED` | tf32 | f32 | TS + per-element scaling |

## When to use it

- Future Blackwell (sm_100+) deployments where maximum tensor core throughput is required.
- MX-format quantized inference using the SCALED variants.

## When NOT to use it

- H200 or any sm_90a device — **tcgen05 instructions are not available on Hopper**. Attempting to compile with `-arch=sm_90a` will fail at ptxas.
- Any target older than sm_100.

## How it connects to the rest of the KB

- **Successor to** `wiki/nvidia/hardware/wgmma-ptx/` — tcgen05 replaces wgmma as the primary tensor core interface on Blackwell.
- **Same conceptual pattern**: load tiles into an opaque register space (tcgen05 regs vs wgmma smem descriptors), issue async MMA, wait for completion.
- cutlass-cute wraps tcgen05 in `MMA_Traits<SM100_MMA_*>` (see `mma_traits_sm100.hpp`) and `Copy_Traits` for tcgen05.cp/ld (see `copy_traits_sm100.hpp`).
