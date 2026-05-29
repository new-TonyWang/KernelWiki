---
api: __pipeline_commit
namespace: runtime
probe_slug: runtime-pipeline-commit
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
- Baseline is a CPU pass-through reference (correctness only), so no GPU timing ratio
  is reported.
- kp_introspect kernel-static not generated (cuda-python not installed).
id: exp-2026-04-17-runtime-pipeline-commit
type: experience
vendor: nvidia
title: 2026 04 17 Runtime Pipeline Commit
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L3924-L3924
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L11100-L11160
---
## Summary

End-to-end probe of `__pipeline_commit` on H200 (sm_90a, CUDA 12.9). `__pipeline_commit` seals the sequence of `__pipeline_memcpy_async` calls issued by the calling thread since the previous commit into a single numbered pipeline stage, giving later `__pipeline_wait_prior(N)` a unit of accounting to drain. Internally the compiler lowers this to an `LDGDEPBAR`-like dependency barrier on the thread's LDGSTS pipeline. The probe exercises commit in two roles: (a) closing a single-stage copy before a `wait_prior(0)`, and (b) closing each of three consecutive stages to prove the in-order FIFO draining by `wait_prior(2)/(1)/(0)`. Both modes passed (max_abs_err = 0, and all three stage checksums matched the CPU reference exactly). The timing row is the single-stage commit+wait path which is the same kernel recorded under `2026-04-17-runtime-pipeline-memcpy-async.md` (shared probe).

## Minimal Kernel

```cuda
// Multi-stage ordering kernel: 3 commits, then drain FIFO-style.
__global__ void kernel_multi_stage(const float* __restrict__ in,
                                   unsigned int* __restrict__ stage_check,
                                   int n) {
    __shared__ float smem[3][256];
    const int tid  = threadIdx.x;
    const int base = blockIdx.x * 256;

    // Issue three commit-stages; each smem[s] holds one distinct slab.
    for (int s = 0; s < 3; ++s) {
        int g = base + s * 256 + tid;
        if (g < n) {
            __pipeline_memcpy_async(&smem[s][tid], &in[g], sizeof(float));
        }
        __pipeline_commit();   // << seal stage s
    }

    // Drain FIFO: wait_prior(N) leaves at most N newest stages pending.
    __pipeline_wait_prior(2); __syncthreads();
    atomicAdd(&stage_check[0], __float_as_uint(smem[0][tid]));

    __pipeline_wait_prior(1); __syncthreads();
    atomicAdd(&stage_check[1], __float_as_uint(smem[1][tid]));

    __pipeline_wait_prior(0); __syncthreads();
    atomicAdd(&stage_check[2], __float_as_uint(smem[2][tid]));
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

Configuration: single-stage pass-through kernel, N_VEC = 1,048,576 `float4` elements, grid = 4096, block = 256, 5 warmup + 20 measurements via CUDA events. Multi-stage correctness: 4 blocks x 3 stages x 256 threads; per-stage bit-exact checksum match vs. CPU reference.

| shape | dtype | latency_ms_median | latency_ms_p10 | latency_ms_p90 | baseline_name | baseline_ms | ratio | clock_policy | reproduce_cmd |
|---|---|---|---|---|---|---|---|---|---|
| N_VEC=1048576 | float4 | 0.009984 | 0.009760 | 0.010368 | cpu-pass-through-reference | N/A | N/A | unknown | `nvcc -arch=sm_90a -O3 -std=c++17 -o /tmp/p sources/experience/api-probes/artifacts/__pipeline_memcpy_async_probe.cu && /tmp/p` |

Multi-stage stage-checksum results (bit-exact):

| stage | gpu_checksum | cpu_checksum | status |
|---|---|---|---|
| 0 | 0x1100c000 | 0x1100c000 | ok |
| 1 | 0xd3edc000 | 0xd3edc000 | ok |
| 2 | 0xc29f4000 | 0xc29f4000 | ok |

## Introspection

No `kp_introspect kernel-static` bundle generated. Device-static info: NVIDIA H200, sm_9.0, CUDA runtime 12.9, driver CUDA 12.8 (system driver 570.124.06).

## Notes

- **Signature**: `void __pipeline_commit(void);` declared in `<cuda_pipeline.h>`. No arguments, no return. Acts on the **calling thread's** implicit pipeline only (thread scope).
- **Semantic (verified)**: every `__pipeline_memcpy_async` issued by this thread since the most recent `__pipeline_commit` (or since kernel entry) is bundled into one **commit stage**. Stages are numbered implicitly, in order. A subsequent `__pipeline_wait_prior(N)` will treat them as a FIFO queue and leave at most N **most-recent** stages pending; the older stages are guaranteed complete when the wait returns. The multi-stage probe confirmed this by placing stage-0 data into `smem[0]`, stage-1 into `smem[1]`, stage-2 into `smem[2]`, then reading the stages back in the `wait_prior(2) / (1) / (0)` order and matching a bit-exact CPU reference per stage.
- **Empty commit is legal**: if no `__pipeline_memcpy_async` calls were issued since the previous commit, `__pipeline_commit` still advances the stage counter (the Programming Guide §4.10.6 example at L11100-L11160 commits after a single copy without guarding for that case). The probe does not rely on this but observed no crash when an idle thread in the prologue skipped its `memcpy_async` and still hit the commit.
- **Compiler lowering**: the Programming Guide §3.2.4.3 table at L3924 summarises `__pipeline_commit` as "Commits the asynchronous operations issued before the call on the current stage of the pipeline." The instruction emitted for the primitives API is a dependency barrier on the thread's LDGSTS queue; see the async-copy skill for the LDGSTS / LDGDEPBAR correspondence.
- **Does NOT imply `__syncthreads`**: commit is thread-scoped. If the block uses a producer/consumer split across threads (only some threads issue `memcpy_async`), the consumer threads still need a `__syncthreads()` after their `__pipeline_wait_prior`, because neither commit nor wait_prior imposes inter-thread ordering.
- **Relation to `cuda::pipeline`**: the higher-level `cuda::pipeline<thread_scope_thread>` wraps exactly this mechanism; `pipe.producer_commit()` corresponds to `__pipeline_commit()` (Programming Guide §3.2.4.3, L3905-L3945 primitives-API table).
- **Pitfall**: calling `__pipeline_commit` without any `__pipeline_memcpy_async` beforehand but then relying on `__pipeline_wait_prior(0)` to "flush" is a no-op on correctness but wastes a stage slot; the ordering guarantee is about stages the calling thread itself committed.
