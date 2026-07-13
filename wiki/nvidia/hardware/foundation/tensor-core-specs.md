---
id: hw-foundation-tensor-core-specs
title: Tensor Core Specs
type: hardware
vendor: nvidia
architectures:
- sm90
- sm90a
tags:
- cuda-cpp
confidence: source-reported
related: []
sources: []
aliases: []
---
# Hopper (sm_90) Tensor Core Reference Card

Quick-reference for tensor core instruction selection and kernel optimization.
Covers 4th-generation Tensor Cores in H100/H200.

## Tensor Core Overview

- 4 Tensor Cores per SM, 528 total on H100/H200 SXM
- Each Tensor Core performs matrix multiply-accumulate (MMA): D = A * B + C
- 2x per-SM throughput vs A100 on all data types (clock-for-clock)
- 4x per-SM throughput vs A100 on FP8 (new in Hopper)
- Structured sparsity (2:4) doubles effective throughput

## Instruction Hierarchy: Three Abstraction Levels

### 1. WMMA (C++ API) -- Highest level, most portable

| Aspect | Details |
|---|---|
| API | `nvcuda::wmma` namespace (C++ templates) |
| Scope | Warp-level (32 threads) |
| Data handling | `load_matrix_sync` / `store_matrix_sync` manage layout implicitly |
| Fragment layout | Opaque, architecture-specific -- cannot assume mapping |
| Portability | Works across sm_70+ (Volta through Hopper) |
| Limitation | Fixed tile shapes, no explicit register control, no FP8 support via WMMA |
| Best for | Prototyping, portable code that does not need peak performance |

**WMMA Supported Shapes and Types:**

| Input A/B | Accumulator | Tile Shapes (m-n-k) |
|---|---|---|
| FP16 (__half) | FP32 or FP16 | 16x16x16, 32x8x16, 8x32x16 |
| BF16 (__nv_bfloat16) | FP32 | 16x16x16, 32x8x16, 8x32x16 |
| TF32 (precision::tf32) | FP32 | 16x16x8 |
| INT8 (u8/s8) | INT32 | 16x16x16, 32x8x16, 8x32x16 |
| FP64 (double) | FP64 | 8x8x4 |

### 2. MMA (PTX) -- Mid level, explicit register layout

| Aspect | Details |
|---|---|
| Instruction | `mma.sync.aligned` (PTX) |
| Scope | Warp-level (32 threads) |
| Data handling | Programmer distributes matrix elements across thread registers explicitly |
| Shapes | Asymmetric m16n8kX family (primary) plus legacy m8n8kX |
| Sparsity | Supports structured sparse matrix A (2:4 pattern) |
| Best for | Hand-tuned PTX kernels, libraries like CUTLASS |

**MMA (PTX) Supported Shapes for Hopper (sm_90):**

| Data Type | Dense Shapes | Sparse Shapes |
|---|---|---|
| FP64 | m8n8k4, m16n8k4, m16n8k8, m16n8k16 | -- |
| FP16 | m8n8k4, m16n8k8, m16n8k16 | m16n8k16, m16n8k32 |
| BF16 | m16n8k8, m16n8k16 | m16n8k16, m16n8k32 |
| TF32 | m16n8k4, m16n8k8 | m16n8k8, m16n8k16 |
| INT8 (u8/s8) | m8n8k16, m16n8k16, m16n8k32 | m16n8k32, m16n8k64 |
| FP8 (e4m3/e5m2) | m16n8k32, m16n8k16 | m16n8k64 |
| INT4 (u4/s4) | m8n8k32, m16n8k32, m16n8k64 | m16n8k64, m16n8k128 |

**MMA Data Type Combinations:**

| Input (A, B) | Accumulator (C, D) |
|---|---|
| .f16 | .f16 or .f32 |
| .bf16 | .f32 |
| .tf32 | .f32 |
| .f64 | .f64 |
| .e4m3 / .e5m2 | .f16 or .f32 |
| .u8 / .s8 | .s32 |
| .u4 / .s4 | .s32 |

### 3. WGMMA (PTX) -- Lowest level, Hopper-only, highest throughput

| Aspect | Details |
|---|---|
| Instruction | `wgmma.mma_async` (PTX) |
| Scope | **Warpgroup-level** (4 warps = 128 threads) |
| Execution | Asynchronous -- results via `wgmma.commit_group` / `wgmma.wait_group` |
| Matrix A source | Registers **or** shared memory (via descriptor) |
| Matrix B source | Shared memory only (via descriptor) |
| Requires | `wgmma.fence` before issuing; `fence.proxy.async` for visibility |
| Best for | Peak-throughput GEMM, CUTLASS 3.x, FlashAttention-style kernels |

**WGMMA Supported Shapes (M is always 64):**

| Data Type | K | N range | Shape Pattern |
|---|---|---|---|
| FP16 (dense) | 16 | 8 to 256, step 8 | m64n{8..256}k16 |
| BF16 (dense) | 16 | 8 to 256, step 8 | m64n{8..256}k16 |
| TF32 (dense) | 8 | 8 to 256, step 8 | m64n{8..256}k8 |
| TF32 (sparse) | 16 | 8 to 256, step 8 | m64n{8..256}k16 |
| FP8 e4m3/e5m2 (dense) | 32 | 8 to 256, step 8 | m64n{8..256}k32 |
| INT8 u8/s8 (dense) | 32 | 8 to 256, step 16 | m64n{8..256}k32 |
| FP16 (sparse) | 32 | 8 to 256, step 8 | m64n{8..256}k32 |
| BF16 (sparse) | 32 | 8 to 256, step 8 | m64n{8..256}k32 |
| FP8 (sparse) | 64 | 8 to 256, step 8 | m64n{8..256}k64 |
| INT8 (sparse) | 64 | 8 to 256, step 16 | m64n{8..256}k64 |

**WGMMA Data Type Combinations:**

| Input (A, B) | Accumulator (D) |
|---|---|
| .f16 | .f16 or .f32 |
| .bf16 | .f32 |
| .tf32 | .f32 |
| .e4m3 / .e5m2 | .f16 or .f32 |
| .u8 / .s8 | .s32 |
| .b1 | .s32 |

## Per-SM Throughput Per Cycle

Each SM has 4 Tensor Cores. At sm_90 boost clock (1830 MHz), throughput per SM per cycle:

| Precision | Dense (FLOPs/cycle/SM) | With Sparsity |
|---|---|---|
| FP64 TC | 128 | -- |
| TF32 | 512 | 1024 |
| FP16 / BF16 | 1024 | 2048 |
| FP8 (e4m3/e5m2) | 2048 | 4096 |
| INT8 | 2048 | 4096 |

Derivation: Peak TFLOPS / (num_SMs * boost_clock). Example: FP16 TC dense = 989.4e12 / (132 * 1.83e9) = 4096 ops/cycle/SM for the full chip, which is 1024 FMA-equivalent FLOPs/cycle/SM (each FMA = 2 ops).

Note: The values in the table above represent FMA operations per cycle per SM (each FMA counts as 2 floating-point operations). To get peak TFLOPS, use: TFLOPS = ops_per_cycle_per_SM * 2 * SM_count * clock_GHz.

## Data Layout Requirements

### MMA (mma.sync) Register Mapping

For the most commonly used `m16n8k16` FP16 shape:
- **32 threads** in a warp cooperatively hold the fragments
- **Matrix A (16x16):** Each thread holds 8 FP16 elements (4 registers of .b32, each packing 2xFP16)
- **Matrix B (16x8, transposed as 8x16):** Each thread holds 4 FP16 elements (2 registers of .b32)
- **Matrix C/D (16x8):** Each thread holds 4 FP32 elements (4 registers of .f32)
- Thread-to-element mapping is defined by the PTX ISA (grouping of 4 threads covers 8 rows)

### WGMMA Register/Shared Memory Layout

For `wgmma.mma_async.m64nNk16` (FP16/BF16):
- **Matrix A:** Can be in registers or shared memory
  - In registers: distributed across 128 threads (4 warps). Each thread holds elements from a 64xK tile.
  - In shared memory: described by a matrix descriptor (swizzle pattern configured via descriptor)
- **Matrix B:** Always in shared memory, accessed via descriptor
- **Matrix D (accumulator):** In registers, distributed across 128 threads of the warpgroup
- Shared memory layout for descriptors: 128-byte aligned, uses swizzle modes (128B, 64B, 32B, or none)

### Key Layout Constraints

- Shared memory pointers for WGMMA descriptors must be 16-byte aligned
- WMMA `load_matrix_sync` requires 256-bit (32-byte) aligned pointers
- Leading dimension (ldm) must be a multiple of 16 bytes
- Column-major A and row-major B is the natural layout for Tensor Cores
- WGMMA supports transpose on .f16/.bf16 inputs only (via imm-trans-a/imm-trans-b flags)

## Choosing the Right Abstraction

| Criterion | WMMA | MMA (PTX) | WGMMA (PTX) |
|---|---|---|---|
| Portability | sm_70+ | sm_75+ | sm_90 only |
| FP8 support | No | Yes | Yes |
| Sparsity support | No | Yes | Yes |
| Async execution | No | No | Yes |
| Tile size control | Fixed small tiles | Fixed small tiles | Large m64 tiles, flexible N |
| Integration with TMA | Manual | Manual | Native (descriptor-based) |
| Peak throughput | ~60-80% | ~80-95% | ~95-100% |
| Used by CUTLASS | 2.x (legacy) | 2.x | 3.x |

## Sources

- NVIDIA H100 Tensor Core GPU Architecture Whitepaper
- PTX ISA Reference (CUDA Toolkit 13.2), Sections 9.7.14-9.7.15
- CUDA C++ Programming Guide (CUDA Toolkit 13.2), Section 10.24
