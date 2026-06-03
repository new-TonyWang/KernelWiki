---
api: __shfl_xor_sync
namespace: runtime
probe_slug: shfl-sync-bfly-warp-reduce
status: verified
kind: documented
trigger: skill-build
evidence_level: measured
clock_policy: unknown
measured_on:
  device: NVIDIA H200
  sm: 9.0a
  gpu_uuid: GPU-fbaa167f-4646-b8b9-c4f5-a4c4734ebc25
  cuda_runtime: '12.9'
  driver: 570.124.06
artifacts:
  code: artifacts/experience/hw-probes/shfl-sync-bfly/warp_reduce_probe.cu
  build: nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo -o warp_reduce_probe warp_reduce_probe.cu
  introspection: ''
  profile: ''
source:
- path: spec
  anchor: Reference
conclusions:
  max_abs_err: null
  latency_ms_median: null
  latency_ms_p10: null
  latency_ms_p90: null
  baseline_name: canonical-dependent-chain
  baseline_ms: null
  ratio: null
open_questions:
- clock_policy is unknown — clocks were not locked during measurement.
- The 29.00 cycles/shfl measured here exceeds the canonical 23.83 cycles because each
  step includes a dependent fadd between shfl_xor_sync calls.
id: exp-shfl-sync-bfly
type: experience
vendor: nvidia
title: 2026 04 16 Warp Primitives
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L23979
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/parallel-thread-execution/cuda_parallel-thread-execution_index.html.md
  anchor: L13435-L13520
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L23945-L24034
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1289-L1292
---
## Summary

This probe measures the latency of the **warp-reduce butterfly pattern** using `__shfl_xor_sync` on H200 (sm_90a, CUDA 12.9). Unlike the canonical `kp_introspect` baseline that measures a pure dependent `shfl.sync.bfly` chain (23.83 cycles), this probe exercises the real use-case: a 5-step butterfly reduction (`delta = 16, 8, 4, 2, 1`) with a dependent `fadd` between each shuffle. The measured per-shfl cost is **29.00 cycles** (including the interleaved fadd), and the full 5-step butterfly reduction costs **144.98 cycles**. All measurements are extremely stable (p10 = p90 = median).

## Minimal Kernel

```cuda
// Single butterfly reduction: 5 dependent shfl_xor_sync calls.
__device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_xor_sync(0xFFFFFFFF, val, 16);
    val += __shfl_xor_sync(0xFFFFFFFF, val, 8);
    val += __shfl_xor_sync(0xFFFFFFFF, val, 4);
    val += __shfl_xor_sync(0xFFFFFFFF, val, 2);
    val += __shfl_xor_sync(0xFFFFFFFF, val, 1);
    return val;
}

// Kernel: unrolled chain of 128 butterfly reductions.
__global__ void probe_kernel(uint64_t* __restrict__ out_cycles,
                             float*    __restrict__ out_val,
                             int                    num_trials) {
    int lane = threadIdx.x % 32;
    float val = static_cast<float>(lane + 1);

    for (int trial = 0; trial < num_trials; ++trial) {
        uint64_t start = clock64();
        #pragma unroll
        for (int i = 0; i < 128; ++i) {
            val = warp_reduce_sum(val);
        }
        uint64_t end = clock64();
        if (lane == 0) {
            out_cycles[trial] = end - start;
        }
    }
    if (lane == 0) {
        out_val[0] = val;
    }
}
```

## Build

```bash
nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo -o warp_reduce_probe warp_reduce_probe.cu
```

## Measurement

Configuration: 4096 trials, 128 butterfly reductions per trial, 5 `__shfl_xor_sync` calls per reduction (640 shfl total per trial). Single warp (1 block, 32 threads). Warmup: 5 launches discarded.

| metric | median | p10 | p90 | unit |
|---|---|---|---|---|
| per butterfly reduction (5 shfl + 5 fadd) | 144.98 | 144.98 | 144.98 | cycles |
| per shfl_xor_sync (within butterfly chain) | 29.00 | 29.00 | 29.00 | cycles |

Canonical baseline (pure dependent shfl.sync.bfly chain, no fadd): 23.83 cycles/shfl. The ~5-cycle difference accounts for the dependent fadd instruction interleaved between each shfl.

## Introspection

No `kp_introspect` bundle was generated for this probe (tool not available in this environment).

## Notes

- The probe uses `clock64()` for cycle measurement, consistent with `hardware-microbench.md` Step 4.
- The final `val` overflows to `inf` due to 128 chained reductions of a 32-lane sum; this is expected and does not affect timing. The store to `out_val` defeats dead-code elimination.
- The near-zero variance (p10 = p90 = median) confirms excellent measurement isolation: no memory stalls contaminate the signal.
- Throughput table from the best-practices guide (Table 5) gives `shfl.sync.idx.b32` throughput of 32 ops/clock/SM on sm_9.0. Since a warp is 32 threads, this means one shfl per clock per warp at peak throughput. Our 29-cycle latency reflects the **dependent-chain latency** (not throughput), which is dominated by the register-to-register shuffle pipeline depth plus the interleaved fadd.
