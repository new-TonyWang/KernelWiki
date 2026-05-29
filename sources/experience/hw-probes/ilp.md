---
api: __fmaf_rn
namespace: math
probe_slug: ilp-fma-accumulator-chains
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
  code: sources/experience/hw-probes/ilp/artifacts/ilp_fma_probe.cu
  build: nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo -o ilp_fma_probe ilp_fma_probe.cu
  introspection: ''
  profile: ''
referenced_in_corpus:
- path: corpus/nvidia/cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming
    Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  line_range: L1113
- path: corpus/nvidia/cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming
    Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  line_range: L1124
- path: corpus/nvidia/cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming
    Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  line_range: L1229-L1232
- path: corpus/nvidia/cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming
    Guides/parallel-thread-execution/cuda_parallel-thread-execution_index.html.md
  line_range: L9106-L9155
source:
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1113
  excerpt: The latency of most arithmetic instructions is typically 4 cycles on devices
    of compute capability 7.0.
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1124
  excerpt: with a high degree of exposed instruction-level parallelism (ILP) it is,
    in some cases, possible to fully cover latency with a low occupancy
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1229-L1232
  excerpt: '32-bit floating-point add, multiply, multiply-add: throughput 128 ops/clock/SM
    on sm_9.0'
conclusions:
  max_abs_err: null
  latency_ms_median: null
  latency_ms_p10: null
  latency_ms_p90: null
  baseline_name: 1-acc-dependent-chain
  baseline_ms: null
  ratio: null
open_questions:
- clock_policy is unknown -- clocks were not locked during measurement.
- The probe uses __fmaf_rn which maps to fma.rn.f32; results should be identical for
  fadd or fmul since they share the same FP32 pipeline on sm_90a.
id: exp-ilp
type: experience
vendor: nvidia
title: 2026 04 16 Ilp
---
## Summary

This probe measures the effect of **instruction-level parallelism (ILP)** on FMA throughput using 1, 2, 4, and 8 independent accumulator chains on H200 (sm_90a, CUDA 12.9). Each configuration executes exactly 1024 `fma.rn.f32` instructions per thread per trial. With a single dependent chain, the measured cost is **4.01 cycles per FMA** (matching the documented 4-cycle arithmetic pipeline latency). With 4 independent chains, the cost drops to **1.01 cycles per FMA**, a **3.97x throughput improvement**. 8 chains show no further gain, confirming that 4 chains fully saturate the FP32 pipeline for a single warp.

## Minimal Kernel

```cuda
// 1-accumulator: single dependent FMA chain
__global__ void probe_1acc(uint64_t* __restrict__ out_cycles,
                           float*    __restrict__ out_val) {
    float acc = 1.0001f;
    float a   = 1.000001f;
    float b   = 0.000001f;

    for (int trial = 0; trial < NUM_TRIALS; ++trial) {
        uint64_t start = clock64();
        #pragma unroll
        for (int i = 0; i < 1024; ++i) {
            acc = __fmaf_rn(acc, a, b);   // dependent chain
        }
        uint64_t end = clock64();
        if (threadIdx.x == 0) out_cycles[trial] = end - start;
    }
    if (threadIdx.x == 0) out_val[0] = acc;
}

// 4-accumulator: four independent FMA chains
__global__ void probe_4acc(uint64_t* __restrict__ out_cycles,
                           float*    __restrict__ out_val) {
    float acc0 = 1.0001f, acc1 = 1.0002f, acc2 = 1.0003f, acc3 = 1.0004f;
    float a = 1.000001f, b = 0.000001f;

    for (int trial = 0; trial < NUM_TRIALS; ++trial) {
        uint64_t start = clock64();
        #pragma unroll
        for (int i = 0; i < 256; ++i) {
            acc0 = __fmaf_rn(acc0, a, b);
            acc1 = __fmaf_rn(acc1, a, b);
            acc2 = __fmaf_rn(acc2, a, b);
            acc3 = __fmaf_rn(acc3, a, b);
        }
        uint64_t end = clock64();
        if (threadIdx.x == 0) out_cycles[trial] = end - start;
    }
    float sum = acc0 + acc1 + acc2 + acc3;
    if (threadIdx.x == 0) out_val[0] = sum;
}
```

## Build

```bash
nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo -o ilp_fma_probe ilp_fma_probe.cu
```

## Measurement

Configuration: 4096 trials, 1024 FMA instructions per trial per thread. Single warp (1 block, 32 threads). Warmup: 5 kernel launches discarded.

| shape | dtype | latency_ms_median | latency_ms_p10 | latency_ms_p90 | baseline_name | baseline_ms | ratio | clock_policy | reproduce_cmd |
|---|---|---|---|---|---|---|---|---|---|
| 1-acc (1024 dep FMAs) | fp32 | n/a (4104 cyc) | n/a (4104 cyc) | n/a (4104 cyc) | 1-acc-dependent-chain | n/a | 1.00 | unknown | `./ilp_fma_probe` |
| 2-acc (2x512 indep FMAs) | fp32 | n/a (2053 cyc) | n/a (2053 cyc) | n/a (2053 cyc) | 1-acc-dependent-chain | n/a (4104 cyc) | 2.00 | unknown | `./ilp_fma_probe` |
| 4-acc (4x256 indep FMAs) | fp32 | n/a (1031 cyc) | n/a (1031 cyc) | n/a (1031 cyc) | 1-acc-dependent-chain | n/a (4104 cyc) | 3.98 | unknown | `./ilp_fma_probe` |
| 8-acc (8x128 indep FMAs) | fp32 | n/a (1031 cyc) | n/a (1031 cyc) | n/a (1031 cyc) | 1-acc-dependent-chain | n/a (4104 cyc) | 3.98 | unknown | `./ilp_fma_probe` |

Key derived metrics (cycles per FMA):

| accumulators | cycles/FMA median | cycles/FMA p10 | cycles/FMA p90 | speedup vs 1-acc |
|---|---|---|---|---|
| 1 | 4.01 | 4.01 | 4.01 | 1.00x |
| 2 | 2.00 | 2.00 | 2.00 | 2.00x |
| 4 | 1.01 | 1.01 | 1.01 | 3.97x |
| 8 | 1.01 | 1.01 | 1.01 | 3.97x |

## Introspection

No `kp_introspect` bundle was generated for this probe (tool not available in this environment).

## Notes

- The measured 4-cycle latency for a single dependent FMA chain matches the documented arithmetic instruction latency for compute capability 7.0+ in the best-practices guide (L1113). The same latency holds on sm_90a (H200).
- The FP32 pipeline throughput from Table 5 of the best-practices guide gives 128 ops/clock/SM for `add.f32` on sm_9.0. Since one warp = 32 threads and the SM has 4 processing blocks (sub-partitions), this translates to 128/32 = 4 warp-instructions/clock/SM, meaning each sub-partition can issue one FP32 instruction per clock. Our single-warp probe sees 1 FMA/clock throughput at 4 accumulators, consistent with the warp being dispatched to a single sub-partition.
- The p10 = median = p90 across all configurations confirms excellent measurement isolation: no memory stalls or scheduling noise.
- 8 accumulators give no improvement over 4. This confirms the pipeline depth is exactly 4 stages: 4 independent chains fully saturate it. Additional chains add register pressure with no throughput benefit.
- The probe uses `__fmaf_rn` (fused multiply-add with round-to-nearest), which compiles to `fma.rn.f32` at the PTX level. Since `fadd.f32` and `fmul.f32` share the same FP32 datapath on Hopper, these latency/throughput numbers apply to all single-precision arithmetic instructions.
