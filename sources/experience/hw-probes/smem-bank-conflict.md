---
api: ld.shared
namespace: ptx
probe_slug: smem-bank-conflict-stride-sweep
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
  code: sources/experience/hw-probes/smem-bank-conflict/artifacts/smem_bank_conflict_probe.cu
  build: nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo -o smem_bank_conflict_probe smem_bank_conflict_probe.cu
  introspection: ''
  profile: ''
referenced_in_corpus:
- path: corpus/nvidia/cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming
    Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  line_range: L1450-L1482
- path: corpus/nvidia/cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming
    Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  line_range: L1604-L1655
- path: corpus/nvidia/cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming
    Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  line_range: L720-L726
- path: corpus/nvidia/cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming
    Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  line_range: L888-L900
- path: corpus/nvidia/cuda-official/cuda-toolkit-documentation-13.2/Miscellaneous/cupti/cupti_index.html.md
  line_range: L4506-L4510
source:
- path: spec
  anchor: Reference
conclusions:
  max_abs_err: null
  latency_ms_median: null
  latency_ms_p10: null
  latency_ms_p90: null
  baseline_name: conflict-free (stride-1)
  baseline_ms: null
  ratio: null
open_questions:
- clock_policy is unknown -- clocks were not locked during measurement.
- 2-way bank conflict shows near-zero overhead (31.01 vs 30.01 cycles); this may indicate
  that H200 LSU handles low-degree conflicts nearly for free, or the measurement granularity
  (1 cycle) may not capture sub-cycle penalties.
id: exp-smem-bank-conflict
type: experience
vendor: nvidia
title: 2026 04 16 Bank Conflict
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L1452-L1453
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L1640-L1647
---
## Summary

This probe measures **shared-memory load latency** on H200 (sm_90a, CUDA 12.9) under five bank-conflict patterns, using a dependent pointer-chasing chain in shared memory (4096 hops per trial, 1280 total samples per pattern).  The conflict-free baseline is **30.01 cycles per load**.  A 32-way bank conflict (stride-32, all 32 threads hit bank 0 with different addresses) costs **91.00 cycles** (3.03x baseline).  The classic +1 padding trick completely eliminates the conflict, restoring performance to **30.01 cycles** (1.00x baseline). Broadcast (all threads reading the same address) also incurs zero penalty (30.01 cycles), confirming hardware multicast.  A 2-way conflict shows negligible overhead (31.01 cycles, 1.03x).

## Minimal Kernel

The probe uses self-loop pointer chasing: `smem[idx] = idx`, so `idx = smem[idx]` is a dependent load that returns the same index, maintaining the access pattern across all 4096 hops.  No arithmetic is interleaved between loads.

```cuda
// Pattern (a): conflict-free (stride-1)
// Thread i reads smem[i] which contains i (self-loop).
// Bank(i) = i % 32 -- all 32 threads hit different banks.
__global__ void probe_conflict_free(uint64_t* __restrict__ out_cycles,
                                    uint32_t* __restrict__ out_val) {
    __shared__ int smem[1024];
    int tid = threadIdx.x;
    for (int i = tid; i < 1024; i += 32)
        smem[i] = i;
    __syncwarp();

    int idx = tid;
    for (int t = 0; t < NUM_TRIALS; ++t) {
        uint64_t s = clock64();
        #pragma unroll 1
        for (int h = 0; h < CHAIN_LEN; ++h)
            idx = smem[idx];
        uint64_t e = clock64();
        if (tid == 0) out_cycles[t] = e - s;
    }
    if (tid == 0) out_val[0] = (uint32_t)idx;
}

// Pattern (d): 32-way bank conflict (stride-32)
// Thread i reads smem[i*32].  All 32 threads hit bank 0.
__global__ void probe_32way(uint64_t* __restrict__ out_cycles,
                            uint32_t* __restrict__ out_val) {
    __shared__ int smem[1024];
    int tid = threadIdx.x;
    for (int i = tid; i < 1024; i += 32)
        smem[i] = i;
    __syncwarp();

    int idx = tid * 32;
    for (int t = 0; t < NUM_TRIALS; ++t) {
        uint64_t s = clock64();
        #pragma unroll 1
        for (int h = 0; h < CHAIN_LEN; ++h)
            idx = smem[idx];
        uint64_t e = clock64();
        if (tid == 0) out_cycles[t] = e - s;
    }
    if (tid == 0) out_val[0] = (uint32_t)idx;
}
```

## Build

```bash
nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo -o smem_bank_conflict_probe smem_bank_conflict_probe.cu
```

## Measurement

Configuration: 4096 hops per trial, 64 trials per launch, 20 launches, 1280 total samples per pattern.  Single warp (1 block, 32 threads). Warmup: 5 launches discarded per pattern.

| pattern | conflict_factor | median_cycles_per_load | p10 | p90 | ratio_vs_baseline |
|---|---|---|---|---|---|
| (a) conflict-free (stride-1) | 0 | 30.01 | 30.01 | 30.01 | 1.00x |
| (b) 2-way conflict (stride-2) | 2 | 31.01 | 31.01 | 31.01 | 1.03x |
| (c) broadcast (all smem[0]) | 0 (multicast) | 30.01 | 30.01 | 30.01 | 1.00x |
| (d) 32-way conflict (stride-32) | 32 | 91.00 | 91.00 | 91.00 | 3.03x |
| (e) stride-32 padded (+1 fix) | 0 | 30.01 | 30.01 | 30.01 | 1.00x |

All measurements show zero variance (p10 = median = p90), indicating excellent isolation and no memory-level interference.

The benchmark-protocol table (latency in milliseconds, with baseline) is not directly applicable here because this probe measures cycle-level per-load latency rather than kernel-level wall-clock time.  The relevant baseline is the conflict-free smem load latency (pattern a).

| shape | dtype | latency_ms_median | latency_ms_p10 | latency_ms_p90 | baseline_name | baseline_ms | ratio | clock_policy | reproduce_cmd |
|---|---|---|---|---|---|---|---|---|---|
| 4096 hops x 32 threads | int32 | n/a (cycle-level) | n/a | n/a | conflict-free stride-1 | n/a | 3.03 (32-way vs free) | unknown | `./smem_bank_conflict_probe` |

## Introspection

No `kp_introspect` bundle was generated for this probe (tool not available in this environment).

## Notes

- The conflict-free baseline of 30.01 cycles per dependent smem load matches the canonical smem pointer-chasing latency of ~28-30 cycles for H200, consistent with the warp-primitives probe which measured 28.83 cycles for a canonical `ld.shared` (before accounting for minor differences in loop overhead).

- The 2-way bank conflict overhead is negligible (~1 cycle).  This may indicate that the H200 shared memory unit handles low-degree conflicts with very little serialization, or that the measurement granularity (clock64() reads in integer cycles) does not capture sub-cycle penalties.

- The 32-way conflict at 91.00 cycles (3.03x baseline) shows significant serialization.  However, the theoretical worst case (32x) is not observed, suggesting that the hardware shares memory bandwidth more efficiently than a naive 32-round serial model.  The ~3x penalty is consistent with the shared memory controller serving multiple conflicting requests per cycle when possible.

- Broadcast (pattern c) confirms the hardware multicast documented in the programming guide: when all threads in a warp read the SAME address in the same bank, the word is broadcast and there is no conflict.

- The +1 padding trick (pattern e) completely eliminates the 32-way conflict, confirming the programming guide's recommendation.  Stride-33 mod 32 = 1, so consecutive threads access consecutive banks.

- NCU metrics for diagnosing bank conflicts in production kernels: `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` (load conflicts) and `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum` (store conflicts), as documented in CUPTI metric mapping (L4506-L4510).
