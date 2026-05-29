---
api: atomicAdd
namespace: runtime
probe_slug: atomic-reduction-contention
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
  code: sources/experience/hw-probes/atomic-reduction-contention/artifacts/atomic_reduction_probe.cu
  build: sources/experience/hw-probes/atomic-reduction-contention/artifacts/build.sh
  introspection: sources/experience/hw-probes/atomic-reduction-contention/artifacts/device.json
  profile: ''
referenced_in_corpus:
- path: corpus/nvidia/cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming
    Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  line_range: L3435-L3436
- path: corpus/nvidia/cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming
    Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  line_range: L3641-L3645
- path: corpus/nvidia/cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming
    Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  line_range: L23252-L23295
source:
- path: spec
  anchor: Reference
conclusions:
  max_abs_err: 0.05029
  latency_ms_median: 0.0393
  latency_ms_p10: 0.039
  latency_ms_p90: 0.04
  baseline_name: naive_atomic (every-thread global atomicAdd to 1 address)
  baseline_ms: 58.8769
  ratio: 1498.29
open_questions:
- clock_policy is unknown — H200 GPU clocks were not explicitly locked during this
  run. Speedup ratios are robust (>100x), but the absolute GB/s numbers should be
  retaken with `nvidia-smi -lgc` lock-and-report for the measured-env contract.
- naive_atomic produced rel_err = 5.03e-02 vs the double-precision reference. The
  error is NOT a bug in the kernel — it is the well-known FP32 accumulation loss when
  millions of small positive values are summed into a single scalar whose magnitude
  grows into the thousands. This means the naive pattern is both contention-pathological
  AND numerically wrong; `hierarchical_s1` cures both because warp/block-local sums
  stay small before the final atomic. This finding has been back-filled into `skill.md`
  as a new sub-pitfall note.
- Scope-latency measurement (cta vs gpu vs sys) from the S2 technique is not yet probed;
  follow-up probe under `sources/experience/hw-probes/atomic-reduction-scope-latency/`
  is open.
- 'Histogram variant (S4: shared-memory atomics vs global-only) is not yet probed;
  follow-up under `sources/experience/hw-probes/atomic-reduction-histogram/` is open.'
id: exp-atomic-reduction-contention
type: experience
vendor: nvidia
title: 2026 04 20 Atomic Reduction
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L3435-L3436
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L3641-L3645
---
## Summary

This probe validates the **S1 hierarchical reduction** technique from `wiki/nvidia/foundations/sync/atomic-reduction/skill.md` on H200 (sm_90a, CUDA 12.9). Three kernels sum `N = 33,554,432` floats (128 MB) to a single scalar:

- **`naive_atomic`** — every thread issues one `atomicAdd(&out, x[i])` to a single global address. Pathological contention.
- **`hierarchical_s1`** — warp shuffle -> shared-memory per-warp partials -> first warp reduces across warps -> **one atomic per block**. The textbook S1 pattern.
- **`grid_stride_s1`** — S1 plus a grid-stride loop so each block folds many elements (persistent-style grid of 16 blocks per SM = 2,112 blocks instead of 131,072). Extends S1 with block-launch reduction.

Results on the first H200 (GPU-fbaa167f..., 132 SMs):

| Kernel            | Median ms | p10    | p90    | Eff. BW GB/s | DRAM SOL | Warp Cyc/Issue |
|-------------------|-----------|--------|--------|--------------|----------|----------------|
| `naive_atomic`    | 58.8769   | 58.876 | 58.878 | 2.28         | 0.05%    | **39,148**     |
| `hierarchical_s1` | 0.2376    | 0.237  | 0.238  | 564.89       | 11.74%   | 65.6           |
| `grid_stride_s1`  | **0.0393**| 0.039  | 0.040  | **3415.56**  | **75.58%** | 67.4         |

**Headline**: S1 alone is 247.8x faster than the naive pattern; S1 + grid-stride is **1498x** faster and approaches HBM peak (~75.6% DRAM SoL, ~3.4 TB/s out of H200's ~4.8 TB/s).

## Minimal kernels (excerpts from the probe .cu)

```cuda
// A. Naive -- one global atomicAdd per thread. DO NOT USE.
__global__ void naive_atomic(const float* in, float* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) atomicAdd(out, in[tid]);
}

// B. S1 hierarchical -- warp shuffle, shmem fan-in, ONE atomic per block.
__global__ void hierarchical_s1(const float* in, float* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (tid < n) ? in[tid] : 0.0f;
    for (int o = 16; o > 0; o >>= 1)                    // warp reduce
        val += __shfl_down_sync(0xFFFFFFFFu, val, o);

    __shared__ float warpSums[32];
    int lane = threadIdx.x & 31, warpId = threadIdx.x >> 5;
    if (lane == 0) warpSums[warpId] = val;
    __syncthreads();

    if (warpId == 0) {
        int nWarps = (blockDim.x + 31) >> 5;
        val = (lane < nWarps) ? warpSums[lane] : 0.0f;
        for (int o = 16; o > 0; o >>= 1)
            val += __shfl_down_sync(0xFFFFFFFFu, val, o);
        if (lane == 0) atomicAdd(out, val);             // one per block
    }
}

// C. S1 + grid-stride loop -- each block folds many elements first.
__global__ void grid_stride_s1(const float* in, float* out, int n) {
    float val = 0.0f;
    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        val += in[i];
    /* same warp + block fan-in as B, then single atomicAdd */
}
```

Launch: `gridFull = ceil(N/256)` blocks for A and B (131,072 blocks); `gridPersist = 132 * 16 = 2,112` blocks for C.

## Protocol

- Problem size: N = 33,554,432 floats (128 MB input + 4 bytes output).
- Block size: 256 threads.
- Warmup: 5 launches; measured: 20 launches per kernel.
- Timer: `cudaEvent_t` pairs with `cudaEventElapsedTime`; median / p10 / p90 over the 20 measured runs.
- Correctness reference: host-side `double`-precision accumulation of the same host buffer; pass threshold `rel_err < 1e-3`.
- NCU section pass: separate run with `--launch-skip 1 --launch-count 1` per kernel, sections `SpeedOfLight` + `WarpStateStats`.
- Clock policy: **unknown** (not locked) — listed as open question.

## NCU section highlights

| Section | naive_atomic | hierarchical_s1 | grid_stride_s1 |
|---|---|---|---|
| Memory Throughput | 0.86% | 20.95% | 75.58% |
| DRAM Throughput   | 0.05% | 11.74% | 75.58% |
| L2 Cache Throughput | 0.86% | 15.93% | 78.93% |
| Compute (SM) SoL    | 0.28% | 23.41% | 20.09% |
| Warp cycles / issued | **39,148** | 65.6 | 67.4 |
| SM active cycles    | 89,755,222 | 346,520 | 49,832 |

The `naive_atomic` line is the smoking gun: DRAM is 0.05% utilized, yet every warp waits 39,148 cycles between issued instructions. The kernel is not memory-bound — it is **atomic-contention-bound at L2**. NCU's top-1 rule on the naive kernel also flagged "52.1% of stall cycles are waiting after EXIT for outstanding memory ops to drain," which is the expected signature of warps posting their atomic then stalling for L2 acknowledgement.

By contrast `grid_stride_s1` lands at 75.58% DRAM SoL — within striking distance of an `ld.global`-bound copy kernel — confirming that once the atomic is no longer the bottleneck, a reduction is simply a memory-bound streaming read.

## New empirical findings (to back-fill into skill.md)

1. **Naive global-atomic reduction is both contention-pathological AND numerically wrong**. In our run, `naive_atomic` produced a result 2,531 units off the double-precision reference (rel_err 5.03e-02, correctness FAIL). The reason is that with 33M individual atomic adds into a single FP32 scalar, the running accumulator grows to ~50,000 early on; thereafter each added sample (~1e-4 to 3e-3) is quantized out by float's 24-bit mantissa. Hierarchical S1 cures both problems simultaneously: warp-lane partials stay in O(32) units, block partials stay in O(8192), and only the final 131K block partials accumulate into the global scalar — each of which is itself a normalized-magnitude sum. **This is a new pitfall** ("naive atomic is numerically wrong, not just slow") worth adding to `pitfalls.md`.
2. **Grid-stride + S1 beats plain S1 by a further ~6x** (under profiling 3.4x) on this problem size. Reason: plain S1 launches one block per 256 elements = 131,072 blocks, each doing ~256 loads + 1 atomic. Block-launch overhead + L2-traffic-per-block dominate once per-block work is so small. Grid-stride S1 uses 2,112 blocks (16 per SM), each folding 15,880 elements before its single atomic; this is essentially a streaming `ld.global` workload with 2,112 atomics at the tail.
3. **NCU's "Memory Throughput" metric is misleading for atomic-bound kernels**. On `naive_atomic` the Memory SoL reads 0.86% — which is not wrong, but if read as "the kernel is compute-bound," it misses the actual story. The signal is in *Warp Cycles Per Issued Instruction* (39,148 vs the typical <100) and the NCU top-rule "waiting after EXIT for outstanding memory ops." Recorded into `pitfalls.md` P6 as a concrete NCU-reading caveat.

## Reproduction

```bash
# On H200 (sm_90a, CUDA 12.9):
cd sources/experience/hw-probes/atomic-reduction-contention/artifacts
bash build.sh
CUDA_VISIBLE_DEVICES=0 ./atomic_reduction_probe
# Expected: naive ~58.9 ms FAIL, hierarchical ~0.24 ms OK, grid-stride ~0.04 ms OK.

# NCU section pass (per kernel):
ncu --launch-skip 1 --launch-count 1 --kernel-name regex:naive_atomic \
    --section SpeedOfLight --section WarpStateStats ./atomic_reduction_probe
```
