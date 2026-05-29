---
id: hw-foundation-h200-specs
title: H200 Specs
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
blackwell_relevance: Foundation hardware concepts apply to both Hopper and Blackwell
---
# NVIDIA H200 SXM Hardware Reference Card

Quick-reference for roofline analysis and kernel optimization decisions.
The H200 uses the same GH100 Hopper die as the H100 SXM5, with upgraded HBM3e memory.

## GPU Compute Architecture

| Parameter | Value |
|---|---|
| Architecture / Compute Capability | Hopper / sm_90a |
| SMs | 132 (8 GPCs, 66 TPCs, 2 SMs/TPC) |
| FP32 CUDA Cores per SM | 128 |
| FP32 CUDA Cores total | 16,896 |
| Tensor Cores per SM | 4 (4th-generation) |
| Tensor Cores total | 528 |
| GPU Boost Clock | 1830 MHz |
| Max Warps per SM | 64 |
| Max Thread Blocks per SM | 32 |
| Warp Size | 32 threads |

## Memory Hierarchy

| Parameter | Value |
|---|---|
| HBM3e Capacity | 141 GB |
| HBM3e Bandwidth | 4.8 TB/s |
| L2 Cache | 50 MB |
| L1 / Shared Memory per SM (combined) | 256 KB |
| Max Shared Memory per SM | 228 KB |
| Max Shared Memory per Thread Block | 227 KB |
| Shared Memory Carveout Options | 0, 8, 16, 32, 64, 100, 132, 164, 196, 228 KB |
| Register File per SM | 64K 32-bit registers (256 KB) |
| Max Registers per Thread | 255 |

## Peak Compute Throughput

All values are for the H200 SXM (identical to H100 SXM5 compute). "Sparse" column uses 2:4 structured sparsity.

### CUDA Core (non-Tensor)

| Precision | Dense |
|---|---|
| FP64 | 33.5 TFLOPS |
| FP32 | 66.9 TFLOPS |
| FP16 | 133.8 TFLOPS |
| BF16 | 133.8 TFLOPS |

### Tensor Core

| Precision | Dense | Sparse (2:4) |
|---|---|---|
| FP64 | 66.9 TFLOPS | -- |
| TF32 | 494.7 TFLOPS | 989.4 TFLOPS |
| FP16 | 989.4 TFLOPS | 1,978.9 TFLOPS |
| BF16 | 989.4 TFLOPS | 1,978.9 TFLOPS |
| FP8 (E4M3/E5M2) | 1,978.9 TFLOPS | 3,957.8 TFLOPS |
| INT8 | 1,978.9 TOPS | 3,957.8 TOPS |

## Roofline Key Numbers

These are the numbers to plug directly into roofline calculations:

| Metric | Value |
|---|---|
| HBM Bandwidth (B_mem) | 4.8 TB/s = 4,800 GB/s |
| L2 Bandwidth (approx) | ~12 TB/s (estimated from H100) |
| Peak FP16 Tensor FLOPS (P) | 989.4 TFLOPS |
| Peak FP8 Tensor FLOPS | 1,978.9 TFLOPS |
| Arithmetic Intensity Crossover FP16 TC | 989.4 / 4.8 = ~206 FLOP/Byte |
| Arithmetic Intensity Crossover FP8 TC | 1,978.9 / 4.8 = ~412 FLOP/Byte |

## Differences from H100 SXM5

| Parameter | H100 SXM5 | H200 SXM |
|---|---|---|
| HBM Type | HBM3 | HBM3e |
| HBM Capacity | 80 GB | 141 GB |
| HBM Bandwidth | 3.35 TB/s | 4.8 TB/s |
| Compute (all precisions) | Same | Same |
| TDP | 700W | 700W |

## Interconnect

| Parameter | Value |
|---|---|
| NVLink (4th gen) | 18 links, 900 GB/s bidirectional |
| PCIe | Gen 5, 128 GB/s bidirectional |

## Sources

- NVIDIA H200 Tensor Core GPU Datasheet (SC24)
- NVIDIA H100 Tensor Core GPU Architecture Whitepaper
- CUDA Hopper Tuning Guide (CUDA Toolkit 13.2)
