---
api: __pipeline_memcpy_async
namespace: runtime
probe_slug: runtime-pipeline-memcpy-async
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
  code: sources/experience/api-probes/artifacts/__pipeline_memcpy_async_probe.cu
  build: nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo -o /tmp/pipeline_probe sources/experience/api-probes/artifacts/__pipeline_memcpy_async_probe.cu
  introspection: ''
  profile: ''
referenced_in_corpus:
- path: corpus/nvidia/cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming
    Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  line_range: L3905-L3945
- path: corpus/nvidia/cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming
    Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  line_range: L11100-L11160
- path: corpus/nvidia/cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming
    Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  line_range: L960-L1000
source:
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L3919-L3923
  excerpt: __pipeline_memcpy_async — Request a memory copy from global to shared memory
    to be submitted for asynchronous evaluation.
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L985-L985
  excerpt: In the asynchronous version of the kernel, instructions to load from global
    memory and store directly into shared memory are issued as soon as __pipeline_memcpy_async()
    function is called.
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L985-L985
  excerpt: If individual CUDA threads are copying elements of 16 bytes, the L1 cache
    can be bypassed.
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
- Baseline is a CPU pass-through reference (correctness only), so no GPU timing ratio
  is reported. For bandwidth comparisons against a sync copy, see sources/experience/hw-probes/async-copy/2026-04-16-async-copy.md.
- kp_introspect kernel-static not generated (cuda-python not installed).
id: exp-2026-04-17-runtime-pipeline-memcpy-async
type: experience
vendor: nvidia
title: 2026 04 17 Runtime Pipeline Memcpy Async
---
## Summary

End-to-end probe of `__pipeline_memcpy_async` on H200 (sm_90a, CUDA 12.9). The primitive issues a hardware-accelerated asynchronous copy from global memory directly to shared memory (LDGSTS), bypassing the thread's register file. It only makes forward progress once grouped with `__pipeline_commit` and synchronised with `__pipeline_wait_prior`, so this probe exercises the full trio; this record focuses on `__pipeline_memcpy_async`'s role as the copy-issuing primitive. A single-stage 16-byte-per-thread `float4` copy (L1 BYPASS eligible) was run over 1 Mi float4 elements (16 MiB). Output matched the CPU reference exactly (max_abs_err = 0). Kernel median latency: 0.009984 ms over 20 runs.

## Minimal Kernel

```cuda
// Single-stage L1-bypass float4 copy.
__global__ void kernel_single_stage(const float4* __restrict__ in,
                                    float4* __restrict__ out,
                                    int n_vec) {
    __shared__ __align__(16) float4 smem[256];
    const int tid = threadIdx.x;
    const int g   = blockIdx.x * 256 + tid;

    if (g < n_vec) {
        // 16-byte async copy, L1 BYPASS path (dst & src both 16B aligned).
        __pipeline_memcpy_async(&smem[tid], &in[g], sizeof(float4));
    }
    __pipeline_commit();      // group the pending issues as one stage
    __pipeline_wait_prior(0); // wait until that stage completes
    __syncthreads();          // pipeline wait is thread-scoped

    if (g < n_vec) out[g] = smem[tid];
}
```

Full source: `sources/experience/api-probes/artifacts/__pipeline_memcpy_async_probe.cu`.

## Build

```bash
nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo \
  -o /tmp/pipeline_probe \
  sources/experience/api-probes/artifacts/__pipeline_memcpy_async_probe.cu
```

## Measurement

Configuration: N_VEC = 1,048,576 `float4` elements (16 MiB), grid = 4096 blocks, block = 256 threads. 5 warmup launches + 20 measurement launches via CUDA events.

| shape | dtype | latency_ms_median | latency_ms_p10 | latency_ms_p90 | baseline_name | baseline_ms | ratio | clock_policy | reproduce_cmd |
|---|---|---|---|---|---|---|---|---|---|
| N_VEC=1048576 | float4 | 0.009984 | 0.009760 | 0.010368 | cpu-pass-through-reference | N/A | N/A | unknown | `nvcc -arch=sm_90a -O3 -std=c++17 -o /tmp/p sources/experience/api-probes/artifacts/__pipeline_memcpy_async_probe.cu && /tmp/p` |

## Introspection

No `kp_introspect kernel-static` bundle generated (cuda-python not installed in this environment). Device-static info read from `cudaGetDeviceProperties`: NVIDIA H200, sm_9.0, CUDA runtime 12.9, driver CUDA 12.8 (system driver 570.124.06).

## Notes

- **Signature**: `void __pipeline_memcpy_async(void* dst, const void* src, size_t size);` Declared in `<cuda_pipeline.h>` (compiler builtin; not an ABI-stable entry point). Supported copy sizes are 4, 8, and 16 bytes per thread; `size = 16` combined with 16-byte aligned `dst`/`src` enables the L1 BYPASS path. Programming Guide §3.2.4.3 ("Pipelines") lists the primitive in the primitives-API table at L3919-L3923.
- **What the call does, on its own**: it only **requests** an async copy; the hardware does not begin scheduling the LDGSTS until the thread also calls `__pipeline_commit()`, and the result is not visible in shared memory until `__pipeline_wait_prior(N)` drains the stage. A lone `__pipeline_memcpy_async` followed by a plain SMEM read is undefined.
- **Scope**: the pipeline is thread-scoped. When a subset of threads does the copies (producer / consumer pattern with 16-byte-per-thread tiles), a `__syncthreads()` is still required before the consumers read smem, because the `__pipeline_wait_prior` only synchronises the calling thread's own pending pipeline (Programming Guide §4.10.6 example at L11100-L11160).
- **Alignment**: the L1 BYPASS path requires both source and destination 16-byte aligned. The probe uses `__align__(16) float4 smem[...]` and a `float4*` input pointer to guarantee alignment.
- **Upstream corroboration**: CUDA C++ Best Practices Guide §10.2.3.4 (L960-L1000) shows the canonical "issue in a loop, commit, wait_prior(0)" usage and confirms the L1-bypass rule at L985.
- **Relation to higher-level API**: the primitives API is the lowest level; `cuda::pipeline<thread_scope_thread>` and `cuda::memcpy_async` build on top of it. See `wiki/nvidia/foundations/memory/async-copy/skill.md` and `sources/experience/hw-probes/async-copy/2026-04-16-async-copy.md` for the skill-level discussion and a bandwidth comparison against a vanilla elementwise kernel.
- **No correctness issues observed**: max_abs_err = 0.0 across all 4,194,304 float elements in the pass-through test.
