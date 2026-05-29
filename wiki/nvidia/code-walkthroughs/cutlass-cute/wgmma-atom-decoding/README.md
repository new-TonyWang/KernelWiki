---
id: code-cutlass-cute-wgmma-atom-decoding-README
type: code-walkthrough
vendor: nvidia
title: Readme
upstream_repo: NVIDIA/cutlass-cute
---
# wiki/nvidia/code-walkthroughs/cutlass-cute/wgmma-atom-decoding — understanding CUTLASS WGMMA atoms

This directory is a **CUTLASS reading aid**. Its purpose is to explain how CUTLASS / CuTe represents Hopper WGMMA instructions in C++ types, especially in `48_hopper_warp_specialized_gemm.cu`.

It is not a handwritten-kernel tuning guide. Performance rules, atom recommendations, and register/smem budgeting belong outside this folder.

The canonical runnable artifact for the CUTLASS path lives at:

- `sources/experience/api-probes/gemm/artifacts/gemm_aligned.cu`

## Reader Question

This folder answers one question:

> In CUTLASS / CuTe, how do I describe the single hardware WGMMA operation that the consumer warpgroup will repeatedly issue?

That single operation is called an **atom**. The surrounding producer / consumer mainloop is tracked separately under `../example48-hopper-warp-specialized-gemm/mainloop_skeleton.md`. No buildable code lives here.

## Use This For

Use this directory when you are reading CUTLASS code and need to decode:

- what `MMA_64x128x8_F32TF32TF32_SS_TN` means
- why that type is wrapped in `MMA_Atom<...>`
- how `TiledMMA` repeats the atom across a CTA tile
- why `cute::gemm(tiled_mma, sA, sB, accum)` lowers to `wgmma.mma_async`
- how to read atom names in profiler/kernel-signature output

Do not use this directory to understand the full GEMM schedule. That belongs to `../example48-hopper-warp-specialized-gemm/`.

## What Atom Means Here

An atom is CuTe's type-level description of **one architecture instruction family instance**.

For WGMMA on Hopper, one atom answers:

| Question | Example answer |
|---|---|
| Which instruction shape? | `64x128x8` |
| Which accumulator / input types? | `F32 <- TF32 x TF32` |
| Where do operands come from? | `SS` = A and B both from shared memory |
| Which logical layouts? | `TN` = A row-major-ish, B column-major-ish for this atom |
| Which PTX instruction is emitted? | `wgmma.mma_async.sync.aligned.m64n128k8.f32.tf32.tf32.f32` |

So:

```cpp
MMA_Atom<MMA_64x128x8_F32TF32TF32_SS_TN<1, 1>>
```

means "the smallest WGMMA unit CuTe will tile with is a Hopper warpgroup MMA instruction that computes a 64 x 128 output slice over K=8 using TF32 inputs and F32 accumulation."

It is **not** the whole GEMM. It is also **not** the whole CTA tile. A CTA tile such as `Shape<_128,_128,_32>` is built by repeating this atom across M and K:

```text
CTA tile:       128 x 128 x 32
WGMMA atom:      64 x 128 x  8
Needed repeats:   2 x   1 x  4 atom issues
```

`TiledMMA` is the CuTe object that says how to repeat the atom. `cute::gemm(tiled_mma, sA, sB, accum)` is the call that emits those repeated atom instructions.

## Name Decoding

The naming pattern is:

```text
MMA_<M>x<N>x<K>_<Acc><A><B>_<OperandSource>_<Layout>
```

For the production TF32 atom:

```text
MMA_64x128x8_F32TF32TF32_SS_TN
    64x128x8      one hardware WGMMA instruction shape
    F32           accumulator type
    TF32TF32      A and B input types
    SS            A and B are read from shared memory
    TN            A/B logical layout variant
```

## Atom Name Catalogue

CuTe exposes one atom type per `(M, N, K, accumulator-dtype, A-dtype, B-dtype, layout, A-source)` combination. This table is for decoding CUTLASS symbols, not for recommending which atom to choose.

The naming pattern is `MMA_<M>x<N>x<K>_<accD><dtA><dtB>_<SS|RS>_<TN|NT|NN|TT>` where `SS` = both A and B in smem, `RS` = A in registers + B in smem.

| Axis | Hopper-supported values | Notes |
|---|---|---|
| M | 64 | wgmma's M is fixed at 64 (= 4 warps × 16 rows) |
| N | 8, 16, 32, 64, 96, 128, 192, 256 | larger N → more FLOPs / issue, more accumulator regs |
| K | 8 (tf32) / 16 (bf16, fp16) / 32 (s8/u8/fp8) | dtype-dependent |
| Acc / A / B dtype | f32/tf32/tf32 · f32/bf16/bf16 · f32/fp16/fp16 · f16/fp16/fp16 · s32/s8/s8 · s32/u8/u8 · f32/fp8/fp8 | tf32 supports only `_TN`; the others support all four trans variants |
| Layout | `_SS_TN`, `_SS_NT`, `_SS_NN`, `_SS_TT` | bf16 / fp16 / int / fp8 atoms expose all four; tf32 only `_TN` |
| A-source | `_SS_*` (smem) / `_RS_*` (registers) | RS skips A's smem footprint at the cost of an explicit register pack |

## Cross-references

- Skill: `wiki/nvidia/hardware/wgmma/skill.md`
- Cutlass-free counterpart (raw PTX, same instructions): `wiki/nvidia/hardware/wgmma-ptx/skill.md`
- Pitfalls: `wiki/nvidia/hardware/wgmma/pitfalls.md`
- Canonical artifacts: `sources/experience/api-probes/gemm/artifacts/gemm_aligned.cu`
- Atom skeleton: `wgmma_skeleton.md`
- Surrounding WS mainloop: `../example48-hopper-warp-specialized-gemm/mainloop_skeleton.md`
- Atom-shape sweep: `sources/experience/api-probes/gemm/2026-04-28-wgmma-atom-shape-sweep.md`
- Counter measurements: `sources/experience/api-probes/gemm/2026-04-28-wgmma-counters.md`
