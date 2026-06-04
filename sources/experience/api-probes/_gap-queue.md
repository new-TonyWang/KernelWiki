---
id: exp-_gap-queue
type: experience
vendor: nvidia
title: _Gap Queue
probe_slug: _gap-queue
evidence_level: measured
measured_on:
  device: H200
  sm: sm_90a
  cuda_runtime: '12.8'
  driver: '570'
architectures:
- sm90
- sm90a
techniques:
- vectorized-loads
- software-exp
confidence: experimental
tags:
- vectorized-loads
- software-exp
---
# API Probe Gap Queue

> Produced by `tools/probe_gap_scan.py` (M2). One row per pending probe.
> MVP rule: rows are appended at the tail; rows removed only when the probe
> record lands under `artifacts/experience/api-probes/<date>-<slug>.md` AND
> `lint_knowledge.py` passes on the result.

| priority | api | namespace | gap_reason | referenced_in_corpus | referenced_by | added_at | claimed_by | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

<!--
Empty. First rows are written during M5 after M4c produces the first set of
skill four-packs. Until then, this file serves as a place-holder so agents
can grep for the gap-queue without having to handle "file not found".
-->
| high | __ldg | runtime | no-end-to-end-example |  | wiki/nvidia/api-definitions/runtime/__ldg.md | 2026-04-16 | kb-gen-agent | done |
| high | __ballot_sync | runtime | no-end-to-end-example |  | wiki/nvidia/api-definitions/runtime/__ballot_sync.md | 2026-04-16 | kb-gen-agent | done |
| high | __shfl_xor_sync | runtime | no-end-to-end-example |  | wiki/nvidia/api-definitions/runtime/__shfl_xor_sync.md | 2026-04-16 | kb-gen-agent | done |
| high | __all_sync | runtime | no-end-to-end-example |  | wiki/nvidia/api-definitions/runtime/__all_sync.md | 2026-04-17 | kb-gen-agent | done |
| high | __any_sync | runtime | no-end-to-end-example |  | wiki/nvidia/api-definitions/runtime/__any_sync.md | 2026-04-17 | kb-gen-agent | done |
| medium | __expf | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/fast-math/apis.md | 2026-04-17 | seed-stubs | superseded |
| medium | __exp10f | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/fast-math/apis.md | 2026-04-17 | seed-stubs | superseded |
| medium | __logf | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/fast-math/apis.md | 2026-04-17 | seed-stubs | superseded |
| medium | __log2f | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/fast-math/apis.md | 2026-04-17 | seed-stubs | superseded |
| medium | __log10f | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/fast-math/apis.md | 2026-04-17 | seed-stubs | superseded |
| medium | __sinf | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/fast-math/apis.md | 2026-04-17 | seed-stubs | superseded |
| medium | __cosf | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/fast-math/apis.md | 2026-04-17 | seed-stubs | superseded |
| medium | __sincosf | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/fast-math/apis.md | 2026-04-17 | seed-stubs | superseded |
| medium | __tanf | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/fast-math/apis.md | 2026-04-17 | seed-stubs | superseded |
| medium | __tanhf | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/fast-math/apis.md | 2026-04-17 | seed-stubs | superseded |
| medium | __powf | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/fast-math/apis.md | 2026-04-17 |  | pending |
| medium | __fdividef | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/fast-math/apis.md | 2026-04-17 |  | pending |
| medium | __fadd_rn | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/fast-math/apis.md | 2026-04-17 |  | pending |
| medium | __fmul_rn | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/fast-math/apis.md | 2026-04-17 |  | pending |
| medium | __fmaf_rn | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/fast-math/apis.md | 2026-04-17 |  | pending |
| medium | __frcp_rn | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/fast-math/apis.md | 2026-04-17 |  | pending |
| medium | __fsqrt_rn | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/fast-math/apis.md | 2026-04-17 |  | pending |
| medium | __frsqrt_rn | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/fast-math/apis.md | 2026-04-17 |  | pending |
| medium | __expf | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/fast-math/pitfalls.md | 2026-04-17 | seed-stubs | superseded |
| medium | __fdividef | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/fast-math/pitfalls.md | 2026-04-17 |  | pending |
| medium | __sinf | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/fast-math/pitfalls.md | 2026-04-17 | seed-stubs | superseded |
| medium | __cosf | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/fast-math/pitfalls.md | 2026-04-17 | seed-stubs | superseded |
| medium | __log2f | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/fast-math/pitfalls.md | 2026-04-17 | seed-stubs | superseded |
| medium | __powf | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/fast-math/pitfalls.md | 2026-04-17 |  | pending |
| medium | __sinf | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/fast-math/skill.md | 2026-04-17 | seed-stubs | superseded |
| medium | __cosf | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/fast-math/skill.md | 2026-04-17 | seed-stubs | superseded |
| medium | __expf | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/fast-math/skill.md | 2026-04-17 | seed-stubs | superseded |
| medium | __logf | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/fast-math/skill.md | 2026-04-17 | seed-stubs | superseded |
| medium | __log2f | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/fast-math/skill.md | 2026-04-17 | seed-stubs | superseded |
| medium | __log10f | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/fast-math/skill.md | 2026-04-17 | seed-stubs | superseded |
| medium | __exp10f | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/fast-math/skill.md | 2026-04-17 | seed-stubs | superseded |
| medium | __powf | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/fast-math/skill.md | 2026-04-17 |  | pending |
| medium | __sincosf | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/fast-math/skill.md | 2026-04-17 | seed-stubs | superseded |
| medium | __tanf | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/fast-math/skill.md | 2026-04-17 | seed-stubs | superseded |
| medium | __tanhf | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/fast-math/skill.md | 2026-04-17 | seed-stubs | superseded |
| medium | __fdividef | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/fast-math/skill.md | 2026-04-17 |  | pending |
| medium | __fadd_rn | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/fast-math/skill.md | 2026-04-17 |  | pending |
| medium | __fmul_rn | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/fast-math/skill.md | 2026-04-17 |  | pending |
| medium | __fmaf_rn | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/fast-math/skill.md | 2026-04-17 |  | pending |
| medium | __fsub_rn | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/fast-math/skill.md | 2026-04-17 |  | pending |
| medium | __fdiv_rn | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/fast-math/skill.md | 2026-04-17 |  | pending |
| medium | __frcp_rn | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/fast-math/skill.md | 2026-04-17 |  | pending |
| medium | __fsqrt_rn | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/fast-math/skill.md | 2026-04-17 |  | pending |
| medium | __frsqrt_rn | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/fast-math/skill.md | 2026-04-17 |  | pending |
| medium | cudaFuncSetAttribute | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/compiler-hints/apis.md | 2026-04-17 | kb-gen-agent | done |
| medium | cudaFuncAttributeMaxDynamicSharedMemorySize | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/compiler-hints/apis.md | 2026-04-17 |  | pending |
| medium | cudaFuncAttributePreferredSharedMemoryCarveout | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/compiler-hints/apis.md | 2026-04-17 |  | pending |
| medium | cudaSharedmemCarveoutDefault | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/compiler-hints/apis.md | 2026-04-17 |  | pending |
| medium | cudaSharedmemCarveoutMaxL1 | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/compiler-hints/apis.md | 2026-04-17 |  | pending |
| medium | cudaSharedmemCarveoutMaxShared | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/compiler-hints/apis.md | 2026-04-17 |  | pending |
| medium | cudaFuncAttributeNonPortableClusterSizeAllowed | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/compiler-hints/apis.md | 2026-04-17 |  | pending |
| medium | cudaOccupancyMaxActiveBlocksPerMultiprocessor | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/compiler-hints/pitfalls.md | 2026-04-17 | kb-gen-agent | done |
| medium | cudaGetDeviceProperties | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/compiler-hints/pitfalls.md | 2026-04-17 |  | pending |
| medium | cudaFuncSetAttribute | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/compiler-hints/pitfalls.md | 2026-04-17 | seed-stubs | superseded |
| medium | cudaFuncSetAttribute | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/compiler-hints/skill.md | 2026-04-17 | seed-stubs | superseded |
| medium | cudaFuncAttributeMaxDynamicSharedMemorySize | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/compiler-hints/skill.md | 2026-04-17 |  | pending |
| medium | cudaFuncAttributePreferredSharedMemoryCarveout | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/compiler-hints/skill.md | 2026-04-17 |  | pending |
| medium | __shfl_sync | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/warp-primitives/apis.md | 2026-04-17 | kb-gen-agent | done |
| medium | __shfl_up_sync | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/warp-primitives/apis.md | 2026-04-17 | kb-gen-agent | done |
| medium | __shfl_down_sync | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/warp-primitives/apis.md | 2026-04-17 | kb-gen-agent | done |
| medium | __shfl_sync | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/warp-primitives/pitfalls.md | 2026-04-17 | seed-stubs | superseded |
| medium | __shfl_down_sync | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/warp-primitives/pitfalls.md | 2026-04-17 | seed-stubs | superseded |
| medium | __half | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/warp-primitives/skill.md | 2026-04-17 |  | pending |
| medium | __half2 | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/warp-primitives/skill.md | 2026-04-17 |  | pending |
| medium | __nv_bfloat16 | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/warp-primitives/skill.md | 2026-04-17 |  | pending |
| medium | __nv_bfloat162 | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/warp-primitives/skill.md | 2026-04-17 |  | pending |
| medium | __shfl_down_sync | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/warp-primitives/skill.md | 2026-04-17 | seed-stubs | superseded |
| medium | __shfl_up_sync | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/warp-primitives/skill.md | 2026-04-17 | seed-stubs | superseded |
| medium | cudaOccupancyMaxActiveBlocksPerMultiprocessor | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/occupancy-tuning/apis.md | 2026-04-17 | seed-stubs | superseded |
| medium | cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/occupancy-tuning/apis.md | 2026-04-17 |  | pending |
| medium | cudaOccupancyDefault | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/occupancy-tuning/apis.md | 2026-04-17 |  | pending |
| medium | cudaOccupancyDisableCachingOverride | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/occupancy-tuning/apis.md | 2026-04-17 |  | pending |
| medium | cudaOccupancyMaxPotentialBlockSize | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/occupancy-tuning/apis.md | 2026-04-17 |  | pending |
| medium | cudaOccupancyMaxPotentialBlockSizeVariableSMem | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/occupancy-tuning/apis.md | 2026-04-17 |  | pending |
| medium | cudaOccupancyMaxPotentialBlockSizeWithFlags | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/occupancy-tuning/apis.md | 2026-04-17 |  | pending |
| medium | cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/occupancy-tuning/apis.md | 2026-04-17 |  | pending |
| medium | cudaOccupancyAvailableDynamicSMemPerBlock | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/occupancy-tuning/apis.md | 2026-04-17 |  | pending |
| medium | cudaOccupancyMaxActiveClusters | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/occupancy-tuning/apis.md | 2026-04-17 |  | pending |
| medium | cudaOccupancyMaxPotentialClusterSize | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/occupancy-tuning/apis.md | 2026-04-17 |  | pending |
| medium | cudaOccupancyMaxPotentialBlockSize | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/occupancy-tuning/pitfalls.md | 2026-04-17 |  | pending |
| medium | cudaOccupancyMaxActiveBlocksPerMultiprocessor | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/occupancy-tuning/pitfalls.md | 2026-04-17 | seed-stubs | superseded |
| medium | cudaFuncSetAttribute | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/occupancy-tuning/pitfalls.md | 2026-04-17 | seed-stubs | superseded |
| medium | cudaFuncAttributePreferredSharedMemoryCarveout | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/occupancy-tuning/pitfalls.md | 2026-04-17 |  | pending |
| medium | cudaOccupancyAvailableDynamicSMemPerBlock | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/occupancy-tuning/pitfalls.md | 2026-04-17 |  | pending |
| medium | cudaOccupancyMaxPotentialBlockSize | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/occupancy-tuning/skill.md | 2026-04-17 |  | pending |
| medium | cudaOccupancyAvailableDynamicSMemPerBlock | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/occupancy-tuning/skill.md | 2026-04-17 |  | pending |
| medium | cudaOccupancyMaxActiveBlocksPerMultiprocessor | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/occupancy-tuning/skill.md | 2026-04-17 | seed-stubs | superseded |
| medium | __fmaf_rn | runtime | dangling-body-ref |  | wiki/nvidia/foundations/compute/ilp/pitfalls.md | 2026-04-17 |  | pending |
| medium | __NV_ATOMIC_SEQ_CST | runtime | dangling-body-ref |  | wiki/nvidia/foundations/sync/memory-ordering/apis.md | 2026-04-17 |  | pending |
| medium | __NV_ATOMIC_ACQUIRE | runtime | dangling-body-ref |  | wiki/nvidia/foundations/sync/memory-ordering/apis.md | 2026-04-17 |  | pending |
| medium | __NV_ATOMIC_RELEASE | runtime | dangling-body-ref |  | wiki/nvidia/foundations/sync/memory-ordering/apis.md | 2026-04-17 |  | pending |
| medium | __NV_ATOMIC_RELAXED | runtime | dangling-body-ref |  | wiki/nvidia/foundations/sync/memory-ordering/apis.md | 2026-04-17 |  | pending |
| medium | __NV_THREAD_SCOPE_CLUSTER | runtime | dangling-body-ref |  | wiki/nvidia/foundations/sync/memory-ordering/apis.md | 2026-04-17 |  | pending |
| medium | cudaMemcpyToSymbol | runtime | dangling-body-ref |  | wiki/nvidia/foundations/sync/memory-ordering/pitfalls.md | 2026-04-17 |  | pending |
| medium | atomicAdd | runtime | dangling-body-ref |  | wiki/nvidia/foundations/sync/memory-ordering/pitfalls.md | 2026-04-17 |  | pending |
| medium | cudaMalloc | runtime | dangling-body-ref |  | wiki/nvidia/foundations/sync/memory-ordering/pitfalls.md | 2026-04-17 |  | pending |
| medium | cudaMallocManaged | runtime | dangling-body-ref |  | wiki/nvidia/foundations/sync/memory-ordering/pitfalls.md | 2026-04-17 |  | pending |
| medium | cudaHostAlloc | runtime | dangling-body-ref |  | wiki/nvidia/foundations/sync/memory-ordering/pitfalls.md | 2026-04-17 |  | pending |
| medium | cudaHostAllocMapped | runtime | dangling-body-ref |  | wiki/nvidia/foundations/sync/memory-ordering/pitfalls.md | 2026-04-17 |  | pending |
| medium | atomicInc | runtime | dangling-body-ref |  | wiki/nvidia/foundations/sync/memory-ordering/skill.md | 2026-04-17 |  | pending |
| medium | cudaFuncAttributes | runtime | dangling-body-ref |  | wiki/nvidia/foundations/memory/register-pressure/apis.md | 2026-04-17 |  | pending |
| medium | cudaOccupancyMaxActiveBlocksPerMultiprocessor | runtime | dangling-body-ref |  | wiki/nvidia/foundations/memory/register-pressure/pitfalls.md | 2026-04-17 | seed-stubs | superseded |
| medium | cudaFuncGetAttributes | runtime | dangling-body-ref |  | wiki/nvidia/foundations/memory/register-pressure/pitfalls.md | 2026-04-17 |  | pending |
| medium | cudaOccupancyMaxActiveBlocksPerMultiprocessor | runtime | dangling-body-ref |  | wiki/nvidia/foundations/memory/register-pressure/skill.md | 2026-04-17 | seed-stubs | superseded |
| medium | __half | runtime | dangling-body-ref |  | wiki/nvidia/foundations/memory/register-pressure/skill.md | 2026-04-17 |  | pending |
| medium | __nv_bfloat16 | runtime | dangling-body-ref |  | wiki/nvidia/foundations/memory/register-pressure/skill.md | 2026-04-17 |  | pending |
| medium | cudaFuncGetAttributes | runtime | dangling-body-ref |  | wiki/nvidia/foundations/memory/register-pressure/skill.md | 2026-04-17 |  | pending |
| medium | __pipeline_memcpy_async | runtime | dangling-body-ref |  | wiki/nvidia/foundations/memory/async-copy/apis.md | 2026-04-17 | kb-gen-agent | done |
| medium | __pipeline_commit | runtime | dangling-body-ref |  | wiki/nvidia/foundations/memory/async-copy/apis.md | 2026-04-17 | kb-gen-agent | done |
| medium | __pipeline_wait_prior | runtime | dangling-body-ref |  | wiki/nvidia/foundations/memory/async-copy/apis.md | 2026-04-17 | kb-gen-agent | done |
| medium | thrust::transform | cccl/thrust | dangling-body-ref |  | wiki/nvidia/foundations/memory/async-copy/apis.md | 2026-04-17 |  | pending |
| medium | thrust::transform | cccl/thrust | dangling-body-ref |  | wiki/nvidia/foundations/memory/async-copy/skill.md | 2026-04-17 |  | pending |
| medium | __pipeline_memcpy_async | runtime | dangling-body-ref |  | wiki/nvidia/foundations/memory/async-copy/skill.md | 2026-04-17 | seed-stubs | superseded |
| medium | cudaMalloc | runtime | dangling-body-ref |  | wiki/nvidia/foundations/memory/vectorized-access/pitfalls.md | 2026-04-17 |  | pending |
| medium | __half | runtime | dangling-body-ref |  | wiki/nvidia/foundations/memory/vectorized-access/pitfalls.md | 2026-04-17 |  | pending |
| medium | __half2 | runtime | dangling-body-ref |  | wiki/nvidia/foundations/memory/vectorized-access/pitfalls.md | 2026-04-17 |  | pending |
| medium | cudaMalloc | runtime | dangling-body-ref |  | wiki/nvidia/foundations/memory/vectorized-access/skill.md | 2026-04-17 |  | pending |
| medium | __half | runtime | dangling-body-ref |  | wiki/nvidia/foundations/memory/coalescing/apis.md | 2026-04-17 |  | pending |
| medium | __half2 | runtime | dangling-body-ref |  | wiki/nvidia/foundations/memory/coalescing/apis.md | 2026-04-17 |  | pending |
| medium | __nv_bfloat16 | runtime | dangling-body-ref |  | wiki/nvidia/foundations/memory/coalescing/apis.md | 2026-04-17 |  | pending |
| medium | __nv_bfloat162 | runtime | dangling-body-ref |  | wiki/nvidia/foundations/memory/coalescing/apis.md | 2026-04-17 |  | pending |
| medium | __ldcg | runtime | dangling-body-ref |  | wiki/nvidia/foundations/memory/coalescing/apis.md | 2026-04-17 | kb-gen-agent | done |
| medium | __ldca | runtime | dangling-body-ref |  | wiki/nvidia/foundations/memory/coalescing/apis.md | 2026-04-17 | kb-gen-agent | done |
| medium | __ldcs | runtime | dangling-body-ref |  | wiki/nvidia/foundations/memory/coalescing/apis.md | 2026-04-17 | kb-gen-agent | done |
| medium | __ldlu | runtime | dangling-body-ref |  | wiki/nvidia/foundations/memory/coalescing/apis.md | 2026-04-17 | kb-gen-agent | done |
| medium | __ldcv | runtime | dangling-body-ref |  | wiki/nvidia/foundations/memory/coalescing/apis.md | 2026-04-17 | kb-gen-agent | done |
| medium | cudaMalloc | runtime | dangling-body-ref |  | wiki/nvidia/foundations/memory/coalescing/pitfalls.md | 2026-04-17 |  | pending |
| high | __cosf | runtime | no-end-to-end-example |  | wiki/nvidia/api-definitions/runtime/__cosf.md | 2026-04-17 |  | pending |
| high | __pipeline_memcpy_async | runtime | no-end-to-end-example |  | wiki/nvidia/api-definitions/runtime/__pipeline_memcpy_async.md | 2026-04-17 |  | pending |
| high | __ldca | runtime | no-end-to-end-example |  | wiki/nvidia/api-definitions/runtime/__ldca.md | 2026-04-17 |  | pending |
| high | __expf | runtime | no-end-to-end-example |  | wiki/nvidia/api-definitions/runtime/__expf.md | 2026-04-17 |  | pending |
| high | __shfl_sync | runtime | no-end-to-end-example |  | wiki/nvidia/api-definitions/runtime/__shfl_sync.md | 2026-04-17 |  | pending |
| high | __ldcg | runtime | no-end-to-end-example |  | wiki/nvidia/api-definitions/runtime/__ldcg.md | 2026-04-17 |  | pending |
| high | __log10f | runtime | no-end-to-end-example |  | wiki/nvidia/api-definitions/runtime/__log10f.md | 2026-04-17 |  | pending |
| high | __pipeline_wait_prior | runtime | no-end-to-end-example |  | wiki/nvidia/api-definitions/runtime/__pipeline_wait_prior.md | 2026-04-17 |  | pending |
| high | __shfl_down_sync | runtime | no-end-to-end-example |  | wiki/nvidia/api-definitions/runtime/__shfl_down_sync.md | 2026-04-17 |  | pending |
| high | cudaOccupancyMaxActiveBlocksPerMultiprocessor | runtime | no-end-to-end-example |  | wiki/nvidia/api-definitions/runtime/cudaOccupancyMaxActiveBlocksPerMultiprocessor.md | 2026-04-17 |  | pending |
| high | __exp10f | runtime | no-end-to-end-example |  | wiki/nvidia/api-definitions/runtime/__exp10f.md | 2026-04-17 |  | pending |
| high | __tanf | runtime | no-end-to-end-example |  | wiki/nvidia/api-definitions/runtime/__tanf.md | 2026-04-17 |  | pending |
| high | __log2f | runtime | no-end-to-end-example |  | wiki/nvidia/api-definitions/runtime/__log2f.md | 2026-04-17 |  | pending |
| high | __logf | runtime | no-end-to-end-example |  | wiki/nvidia/api-definitions/runtime/__logf.md | 2026-04-17 |  | pending |
| high | __pipeline_commit | runtime | no-end-to-end-example |  | wiki/nvidia/api-definitions/runtime/__pipeline_commit.md | 2026-04-17 |  | pending |
| high | __ldcs | runtime | no-end-to-end-example |  | wiki/nvidia/api-definitions/runtime/__ldcs.md | 2026-04-17 |  | pending |
| high | cudaFuncSetAttribute | runtime | no-end-to-end-example |  | wiki/nvidia/api-definitions/runtime/cudaFuncSetAttribute.md | 2026-04-17 |  | pending |
| high | __sincosf | runtime | no-end-to-end-example |  | wiki/nvidia/api-definitions/runtime/__sincosf.md | 2026-04-17 |  | pending |
| high | __shfl_up_sync | runtime | no-end-to-end-example |  | wiki/nvidia/api-definitions/runtime/__shfl_up_sync.md | 2026-04-17 |  | pending |
| high | __ldlu | runtime | no-end-to-end-example |  | wiki/nvidia/api-definitions/runtime/__ldlu.md | 2026-04-17 |  | pending |
| high | __sinf | runtime | no-end-to-end-example |  | wiki/nvidia/api-definitions/runtime/__sinf.md | 2026-04-17 |  | pending |
| high | __tanhf | runtime | no-end-to-end-example |  | wiki/nvidia/api-definitions/runtime/__tanhf.md | 2026-04-17 |  | pending |
| high | __ldcv | runtime | no-end-to-end-example |  | wiki/nvidia/api-definitions/runtime/__ldcv.md | 2026-04-17 |  | pending |
