---
api: __pipeline_wait_prior
namespace: runtime
probe_slug: runtime-pipeline-wait-prior
status: verified
kind: documented
trigger: init-sweep
evidence_level: measured
clock_policy: unknown
measured_on:
  device: NVIDIA H200
  sm: 9.0a
  gpu_uuid: ''
  cuda_runtime: '12.9'
  driver: 570.124.06
artifacts:
  code: artifacts/experience/api-probes/artifacts/__pipeline_memcpy_async_probe.cu
  build: nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo -o /tmp/pipeline_probe artifacts/experience/api-probes/artifacts/__pipeline_memcpy_async_probe.cu
  introspection: ''
  profile: ''
source:
- path: spec
  anchor: Reference
conclusions:
  max_abs_err: 0.0
  latency_ms_median: 0.009984
  latency_ms_p10: 0.00976
  latency_ms_p90: 0.010368
  baseline_name: cpu-pass-through-reference
  baseline_ms: null
  ratio: null
back_filled_into: []
open_questions:
- clock_policy is unknown -- clocks were not locked during measurement.
- Baseline is a CPU pass-through reference (correctness only), so no GPU timing ratio is reported.
- kp_introspect kernel-static not generated (cuda-python not installed).
- N must be a compile-time constant in the Ampere-era LDGSTS lowering; this probe uses integer literals (0, 1, 2). Behavior with a runtime-variable N was not measured.
id: exp-2026-04-17-runtime-pipeline-wait-prior
type: experience
vendor: nvidia
title: 2026 04 17 Runtime Pipeline Wait Prior
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L3927-L3930
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L985-L985
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L3905-L3945
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L11100-L11160
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L960-L1000
architectures:
- sm90
- sm90a
languages:
- cuda-cpp
- python
techniques:
- pipeline-stages
- double-buffering
- vectorized-loads
- shared-memory-optimization
confidence: experimental
tags:
- pipeline-stages
- double-buffering
- vectorized-loads
- shared-memory-optimization
- cuda-cpp
- python
artifact_dir: artifacts/experience/api-probes/artifacts
---
## Summary

End-to-end probe of `__pipeline_wait_prior(N)` on H200 (sm_90a, CUDA 12.9). The primitive blocks the calling thread until **at most N** of the thread's already-committed pipeline stages remain pending; all older stages are guaranteed complete and their destination shared memory visible to the calling thread on return. The probe validates two values of N: `N = 0` (drain everything) in a single-stage pass-through kernel, and `N = 2, 1, 0` in a FIFO drain of three pre-committed stages. Both paths passed: max_abs_err = 0 on 4,194,304 floats for the single-stage kernel, and the per-stage bit-exact checksums of the 3-stage kernel matched the CPU reference across all 3 stages.

## Minimal Kernel

```cuda
// Both uses of __pipeline_wait_prior appear in this probe.

// (a) Drain-all (N = 0):
__global__ void kernel_single_stage(const float4* in, float4* out, int n_vec) {
    __shared__ __align__(16) float4 smem[256];
    int tid = threadIdx.x;
    int g   = blockIdx.x * 256 + tid;
    if (g < n_vec) __pipeline_memcpy_async(&smem[tid], &in[g], sizeof(float4));
    __pipeline_commit();
    __pipeline_wait_prior(0);    // wait until 0 stages remain pending
    __syncthreads();
    if (g < n_vec) out[g] = smem[tid];
}

// (b) Partial drain (N = 2, 1, 0):
__global__ void kernel_multi_stage(const float* in, unsigned int* check, int n) {
    __shared__ float smem[3][256];
    int tid  = threadIdx.x;
    int base = blockIdx.x * 256;
    for (int s = 0; s < 3; ++s) {
        int g = base + s * 256 + tid;
        if (g < n) __pipeline_memcpy_async(&smem[s][tid], &in[g], sizeof(float));
        __pipeline_commit();
    }
    // At this point 3 stages are pending.
    __pipeline_wait_prior(2); __syncthreads();   // stage 0 now done
    atomicAdd(&check[0], __float_as_uint(smem[0][tid]));
    __pipeline_wait_prior(1); __syncthreads();   // stage 1 now done
    atomicAdd(&check[1], __float_as_uint(smem[1][tid]));
    __pipeline_wait_prior(0); __syncthreads();   // stage 2 now done
    atomicAdd(&check[2], __float_as_uint(smem[2][tid]));
}
```

Full source: `artifacts/experience/api-probes/artifacts/__pipeline_memcpy_async_probe.cu`.

## Build

```bash
nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo \
  -o /tmp/pipeline_probe \
  artifacts/experience/api-probes/artifacts/__pipeline_memcpy_async_probe.cu
```

## Measurement

Single-stage kernel configuration: N_VEC = 1,048,576 `float4` elements (16 MiB), grid = 4096, block = 256, 5 warmup + 20 measurement launches via CUDA events. Multi-stage kernel configuration: grid = 4, block = 256, 3 committed stages per block, per-stage checksum verified bit-exact against CPU reference.

| shape | dtype | latency_ms_median | latency_ms_p10 | latency_ms_p90 | baseline_name | baseline_ms | ratio | clock_policy | reproduce_cmd |
|---|---|---|---|---|---|---|---|---|---|
| N_VEC=1048576 | float4 | 0.009984 | 0.009760 | 0.010368 | cpu-pass-through-reference | N/A | N/A | unknown | `nvcc -arch=sm_90a -O3 -std=c++17 -o /tmp/p artifacts/experience/api-probes/artifacts/__pipeline_memcpy_async_probe.cu && /tmp/p` |

Per-stage multi-drain verification (bit-exact):

| wait call | observed stage completed | gpu checksum | cpu checksum | status |
|---|---|---|---|---|
| `__pipeline_wait_prior(2)` | stage 0 | 0x1100c000 | 0x1100c000 | ok |
| `__pipeline_wait_prior(1)` | stage 1 | 0xd3edc000 | 0xd3edc000 | ok |
| `__pipeline_wait_prior(0)` | stage 2 | 0xc29f4000 | 0xc29f4000 | ok |

## Introspection

No `kp_introspect kernel-static` bundle generated. Device-static info: NVIDIA H200, sm_9.0, CUDA runtime 12.9, driver CUDA 12.8 (system driver 570.124.06).

## Notes

- **Signature**: `void __pipeline_wait_prior(size_t N);` declared in `<cuda_pipeline.h>`. N is the **number of most-recent committed stages allowed to remain pending** after the wait returns; older stages are forced complete. Programming Guide §3.2.4.3 (L3927-L3930): "Waits for completion of asynchronous operations in all but the last N commits to the pipeline."
- **`N = 0` is a full drain** (Best Practices Guide §10.2.3.4, L985: "The __pipeline_wait_prior(0) will wait until all the instructions in the pipe object have been executed.").
- **Ordering model (verified by the multi-stage probe)**: commits form a per-thread FIFO. When `pending == 3` and the thread calls `wait_prior(2)`, exactly stage 0 is drained; a subsequent `wait_prior(1)` drains stage 1; `wait_prior(0)` drains stage 2. Data in `smem[s]` for the drained stage is visible **to the calling thread** on return.
- **Scope is thread-local**: on return, data is visible to the calling thread only. A `__syncthreads()` is still required if peer threads need to read what this thread's copy landed in shared memory. The Programming Guide §4.10.6 L11100-L11160 example puts `__syncthreads()` after each `__pipeline_wait_prior` for this reason.
- **Double-buffering pattern**: the canonical 2-stage prefetch (`NUM_STAGES = 2`, see `wiki/nvidia/foundations/memory/async-copy/skill.md`) calls `__pipeline_wait_prior(NUM_STAGES - 1)` each iteration so exactly one stage remains in flight (the freshly-committed prefetch), and the oldest stage becomes available for compute. This is the same primitive exercised here but with `N = NUM_STAGES - 1 = 1`.
- **`N > pending` is not useful** and not tested; behaviorally it is a no-op because the wait asks "at most N stages pending" and the condition is already satisfied.
- **`N` is effectively a compile-time constant**: the Ampere LDGSTS lowering emits a `CP.ASYNC.WAIT_GROUP` instruction whose immediate operand is N. GTC25-S72683 (see async-copy skill) explicitly notes that `NUM_STAGES` passed into `cuda::pipeline_consumer_wait_prior<N>` must be compile-time so the bookkeeping disappears; the same consideration applies here. The probe uses literal 0, 1, 2.
- **Does not block other threads**: like the rest of the primitives API, `__pipeline_wait_prior` is thread-scoped. It is cheaper than a `cuda::barrier::wait` when only one thread's staging needs to drain.
- **Relation to higher-level APIs**: `cuda::pipeline_consumer_wait_prior<N>(pipe)` and `pipe.consumer_wait()` on a `cuda::pipeline<thread_scope_thread>` both lower to this primitive.
