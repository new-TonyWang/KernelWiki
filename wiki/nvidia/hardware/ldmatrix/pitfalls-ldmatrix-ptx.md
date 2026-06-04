---
id: pitfall-ldmatrix-ptx
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
- tma
- ldmatrix
techniques:
- pipeline-stages
- shared-memory-optimization
kernel_types:
- gemm
confidence: inferred
tags:
- wgmma
- tma
- ldmatrix
- pipeline-stages
- shared-memory-optimization
- gemm
- ptx
- cuda-cpp
---
# ldmatrix Pitfalls

## 1. Smem alignment requirement

**Problem**: `ldmatrix.sync.aligned` requires the shared memory pointer to be 16-byte aligned. Misaligned addresses cause undefined behavior or a hardware trap (illegal memory access).

**Trigger**: Allocating smem with arbitrary byte offset, or computing a row pointer that is not 16-byte aligned due to non-power-of-2 row strides.

**Mitigation**: Ensure smem allocation is 16-byte aligned (the default for `__shared__` arrays is typically sufficient). When computing per-thread smem addresses, verify that `(address & 0xF) == 0`.

## 2. Transposed vs non-transposed confusion

**Problem**: Using `ldmatrix.sync.aligned.x4.m8n8` (non-transposed) to load the B operand for a `.row.col` mma.sync, when B needs to be in column-major (transposed) layout. The resulting register contents have the wrong element mapping, causing silent data corruption.

**Trigger**: Not matching the ldmatrix variant to the mma.sync operand layout convention. For `mma.sync...row.col`, A is row-major (use non-transposed ldmatrix) and B is col-major (use transposed ldmatrix).

**Mitigation**: Follow the cutlass convention: A uses `SM75_U32x4_LDSM_N` (non-transposed), B uses `SM75_U16x4_LDSM_T` (transposed) for the standard TN layout. Test with known-answer inputs.

## 3. Per-thread address must point to the thread's own row

**Problem**: All 32 threads in the warp execute ldmatrix simultaneously, and each thread provides its own smem address. The hardware gathers from all 32 addresses to fill the matrix fragment. If a thread provides a wrong address (e.g., all threads provide the same address), the fragment is filled with repeated data from one row.

**Trigger**: Using a broadcast address like `smem_base` instead of `smem_base + lane_id * row_stride`.

**Mitigation**: Compute per-thread addresses carefully. For an 8x8 tile of fp16 in row-major: `smem_ptr = smem_base + (lane_id % 8) * row_stride_bytes`. The first 8 lanes cover the first tile's 8 rows; lanes 8-15 cover the second tile, etc.

## 4. Incompatible with wgmma on Hopper

**Problem**: On Hopper, using ldmatrix to feed wgmma is unnecessary and inefficient. wgmma in SS mode reads operands directly from smem via descriptors. wgmma in RS mode expects A fragments in a different register layout than what ldmatrix produces.

**Trigger**: Porting Ampere kernel patterns (ldmatrix → mma.sync) to Hopper without switching to the TMA → wgmma pipeline.

**Mitigation**: On Hopper, use TMA to fill smem and wgmma.mma_async with smem descriptors (SS mode). Only fall back to ldmatrix if the kernel must use mma.sync for backward compatibility.

## 5. Register count mismatch between x1/x2/x4 and the mma variant

**Problem**: Loading with `ldmatrix.x1` (1 register per thread) but the mma.sync variant expects x4 (4 registers per thread for the A operand of m16n8k16). The mma.sync reads uninitialized register contents for the missing fragments.

**Trigger**: Mismatching the ldmatrix variant with the mma.sync K-dimension. m16n8k8 needs x2 for A; m16n8k16 needs x4 for A.

**Mitigation**: Always match: `mma.sync.m16n8k8` pairs with `ldmatrix.x2` for A and `ldmatrix.x1` for B; `mma.sync.m16n8k16` pairs with `ldmatrix.x4` for A and `ldmatrix.x2` for B. Consult the PTX ISA fragment tables.
