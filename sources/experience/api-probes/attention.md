---
api: MVP minimal flash-attention — thread-level reference (secondary; TMA+wgmma primary
  is pending H200 validation)
namespace: attention
probe_slug: mvp-attention
status: verified
kind: api-probe
trigger: validate correctness and kernel-parameter tuning of hand-built flash-attention
  on H200
evidence_level: measured
clock_policy: as-launched (H200 boost-clock unlocked)
measured_on: H200-SXM | sm_90a | cuda 12.9.86 | driver 570.124.06
source:
- path: sources/experience/api-probes/attention/artifacts/flash_attn_minimal.cu
  anchor: flash_attn_kernel<BM,BN> — templatized thread-level online-softmax attention
artifacts:
  code: artifacts/experience/api-probes/attention/artifacts/flash_attn_minimal.cu
  build: artifacts/experience/api-probes/attention/artifacts/build.sh
  run: artifacts/experience/api-probes/attention/artifacts/run.sh
  introspection: artifacts/experience/api-probes/attention/artifacts/device.json
  profile: artifacts/experience/api-probes/attention/artifacts/profiles/attention-sweep.csv
conclusions:
  workload: Scaled dot-product attention O = softmax(Q@K^T / sqrt(d)) @ V via FlashAttention-2
    online softmax. Thread-level math (no wgmma). fp16 inputs, f32 accumulator, fp16
    output. HEAD_DIM=64, NTHREADS=128. Kernel templatized on BLOCK_M and BLOCK_N.
  correctness: max_abs_err = 0.000031 at B=1 H=2 S=128 D=64 BLOCK_M=64 BLOCK_N=64
    seed=42 against CPU reference. Well within 1e-2 tolerance. All 6 tuning configs
    also pass correctness.
  tuning_sweep: 'Fixed-workload kernel-parameter sweep at B=1 H=2 S=256 D=64 over
    6 BLOCK_M x BLOCK_N configs. Best: BLOCK_M=32 BLOCK_N=64 at 0.151 ms (2.0x faster
    than default 64x64). BLOCK_M is the dominant knob.'
id: exp-attention
type: experience
vendor: nvidia
title: 2026 05 08 Mvp Attention
---
# MVP Minimal Flash-Attention — H200 Measured Record

## Correctness gate

```
Flash-attention MVP correctness: B=1 H=2 S=128 D=64 seed=42 dtype=fp16 tol=1e-2
Config: BLOCK_M=64 BLOCK_N=64 HEAD_DIM=64 NTHREADS=128
max_abs_err: 0.000031
mismatches (>1e-2): 0 / 16384
Disposition: Passed
```

Build: `nvcc -std=c++17 -O3 -gencode=arch=compute_90a,code=sm_90a -lineinfo flash_attn_minimal.cu -o flash_attn_minimal`

Run: `./flash_attn_minimal`

## Kernel-parameter tuning sweep

**Fixed workload**: B=1, H=2, S=256, D=64. Swept BLOCK_M x BLOCK_N (compile-time kernel parameters) across 6 configurations. 1 warmup + 5 timed iterations via CUDA events.

Run: `./flash_attn_minimal --tune`

| BLOCK_M | BLOCK_N | kernel_ms | smem_KB | Q tiles | KV iters | max_abs_err | Disposition |
|---|---|---|---|---|---|---|---|
| 32 | 32 | 0.166 | 20.4 | 8 | 8 | 0.000031 | Passed |
| 32 | 64 | **0.151** | 36.4 | 8 | 4 | 0.000015 | Passed |
| 64 | 32 | 0.341 | 32.8 | 4 | 8 | 0.000031 | Passed |
| 64 | 64 | 0.298 | 56.8 | 4 | 4 | 0.000015 | Passed |
| 64 | 128 | 0.310 | 104.8 | 4 | 2 | 0.000015 | Passed |
| 128 | 64 | 0.502 | 97.5 | 2 | 4 | 0.000015 | Passed |

### Analysis

**Best config**: BLOCK_M=32, BLOCK_N=64 at 0.151 ms.

**BLOCK_M is the dominant knob**: At fixed BLOCK_N=64, doubling BLOCK_M from 32→64 costs 1.97x kernel time (0.151→0.298 ms). This is because larger BLOCK_M increases per-thread work (EPT = BLOCK_M × HEAD_DIM / NTHREADS scales linearly with BLOCK_M), per-tile S=Q@K^T compute (O(BLOCK_M × BLOCK_N × HEAD_DIM)), and per-tile P@V accumulation. The number of Q-tiles halves, but the per-tile cost more than doubles.

**BLOCK_N effect**: At BLOCK_M=32, increasing BLOCK_N from 32→64 gives a 9% speedup (0.166→0.151 ms) because KV iterations halve (8→4), reducing tile-load and sync overhead. At BLOCK_M=64, the BLOCK_N=64 config (0.298 ms) is faster than both BLOCK_N=32 (0.341 ms) and BLOCK_N=128 (0.310 ms), making 64 the sweet spot for this workload size.

**Recommended default**: BLOCK_M=32, BLOCK_N=64 for this thread-level MVP. For wgmma-based attention (where M must be ≥64), BLOCK_M=64, BLOCK_N=64 is the practical floor.

### Asymptotic note

Full self-attention work per batch-head is O(S² × D): the kernel launches ceil(S/BLOCK_M) Q-tiles, each iterating over ceil(S/BLOCK_N) KV-tiles. Total tiles = ceil(S/BLOCK_M) × ceil(S/BLOCK_N), so work scales quadratically with S.

## Device

H200-SXM, sm_90a, 143771 MiB HBM, driver 570.124.06, CUDA 12.9 (nvcc build cuda_12.9.r12.9/compiler.36037853_0).

## Caveats

- This is a **correctness-focused MVP**, not a performance-optimized kernel. Thread-level matmul replaces wgmma; global loads replace TMA. Throughput is orders of magnitude below H200 peak.
- The 128-thread block underutilizes the H200's 132 SMs at small B×H products.
- For production attention throughput, see `wiki/nvidia/code-walkthroughs/cutlass-cute/attention-fmha-example/` (cutlass) or `wiki/nvidia/code-walkthroughs/flash-attention-v3/` (FAv3).
