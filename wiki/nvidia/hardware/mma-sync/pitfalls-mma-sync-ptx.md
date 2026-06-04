---
id: pitfall-mma-sync-ptx
type: pitfall
vendor: nvidia
title: Pitfalls
architectures:
- sm90
- sm90a
languages:
- ptx
- cuda-cpp
hardware_features:
- wgmma
- ldmatrix
techniques:
- shared-memory-optimization
kernel_types:
- gemm
confidence: inferred
tags:
- wgmma
- ldmatrix
- shared-memory-optimization
- gemm
- ptx
- cuda-cpp
---
# mma.sync Pitfalls

## 1. Fragment layout mismatch

**Problem**: The per-thread register layout for mma.sync operands (A, B, C/D) follows a non-obvious mapping described in the PTX ISA. Packing fp16 values into uint32 registers in the wrong order produces silently incorrect results.

**Trigger**: Manually filling A/B register fragments without using `ldmatrix` or cutlass's `copy_atom` to handle the layout mapping.

**Mitigation**: Always use `ldmatrix` to load A/B from shared memory — it handles the smem-to-register transpose that mma.sync expects. If manual packing is unavoidable, consult the PTX ISA section "Matrix Fragments for mma.m16n8k16" and test with known-answer inputs (e.g., identity-matrix × vector).

## 2. Convergent warp required

**Problem**: `mma.sync` is a warp-synchronous instruction. If threads within the warp have diverged (e.g., via data-dependent `if` statements), the instruction will hang or produce undefined behavior.

**Trigger**: Calling mma.sync from within a branch that not all 32 threads in the warp take.

**Mitigation**: Ensure all 32 threads in the warp reach the mma.sync instruction. Use `__syncwarp()` if convergence is in doubt, or restructure the control flow so the mma.sync is on the converged path.

## 3. Alignment and dtype requirements

**Problem**: The A and B operands must be in the exact register types the instruction expects. For `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`, A must be 4 × uint32 (8 packed fp16), B must be 2 × uint32 (4 packed fp16). Passing the wrong number of registers causes a ptxas compile error; passing the right count but wrong dtype causes silent corruption.

**Trigger**: Mixing up the single-K (m16n8k8, 2 A-registers) and double-K (m16n8k16, 4 A-registers) variants, or using bf16 data with an f16 instruction variant.

**Mitigation**: Match the mnemonic suffix exactly: `.f16.f16` requires fp16 data in the registers, `.bf16.bf16` requires bf16. Use cutlass's `MMA_Traits` struct to derive the correct fragment counts: `FrgTypeA`, `FrgTypeB`, `FrgTypeC`, `FrgTypeD`.

## 4. Confusing mma.sync with wgmma

**Problem**: Writing a Hopper kernel that uses mma.sync instead of wgmma. mma.sync works on Hopper (sm_90a) but wastes the wgmma issue port — one mma.sync covers m16n8k16 per warp (32 threads), while one wgmma covers m64n128k16 per warpgroup (128 threads). Throughput can be 8-16x lower.

**Trigger**: Porting Ampere code to Hopper without migrating from mma.sync to wgmma.

**Mitigation**: On sm_90a, always prefer wgmma. Reserve mma.sync for backward-compatibility paths or operators that structurally cannot use warpgroup-level instructions.

## 5. Missing ldmatrix before mma.sync

**Problem**: mma.sync expects A and B fragments in a specific register layout that differs from a naive smem load. Using simple `__shfl_sync` or direct `smem[tid]` loads places data in the wrong register positions.

**Trigger**: Loading A/B from smem via `smem[thread_offset]` into registers and passing to mma.sync without the ldmatrix transpose.

**Mitigation**: Use `ldmatrix` (see `wiki/nvidia/hardware/ldmatrix-ptx/skill.md`) to load from smem — it performs the cross-lane shuffle needed by mma.sync in hardware.
