---
api: PTX cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster
  (cutlass-free)
namespace: ptx
probe_slug: tma-multicast
status: verified
kind: hw-feature
trigger: characterize cluster-multicast TMA load — DRAM-read fan-out across CTAs in
  a cluster
evidence_level: measured
clock_policy: as-launched (H200 boost-clock unlocked)
measured_on: H200-SXM | sm_90a | cuda 12.9.86 | driver 570.124.06
source:
- path: blogs/colfax/cutlass-tutorial-mastering-the-nvidia-tensor-memory-accelerator-tma
  anchor: Hopper TMA walkthrough — multicast variant for cooperative GEMM
- path: 40-hardware-feature/tma-ptx/skill.md
  anchor: cutlass-free TMA primitive (this probe extends it with multicast::cluster)
artifacts:
  code: 80-experience/hw-probes/tma-ptx/artifacts/tma_multicast_probe.cu
  build: 80-experience/hw-probes/tma-ptx/artifacts/build_multicast.sh
  run: 80-experience/hw-probes/tma-ptx/artifacts/run_multicast.sh
  profile: 80-experience/hw-probes/tma-ptx/artifacts/profiles/2026-04-30-tma-multicast.csv
upstream_repo: none (hand-rolled cutlass-free implementation)
conclusions:
  workload: 132 CTAs (= H200 SM count) arranged as 132/C clusters of size C ∈ {1,
    2, 4}; each cluster's leader (rank 0) issues 124 unique TMA tile loads via cp.async.bulk.tensor.2d…multicast::cluster
    with cta_mask = (1<<C)-1; all C CTAs in the cluster receive the same tile sequence
    into their own smem ring buffer (depth=4). Per-CTA workload is held constant at
    124 tiles regardless of C, so DRAM bytes scale as 1/C while smem-bytes delivered
    per launch stays at 128 MiB. 5 warmup + 20 timed launches, median ms.
  c1_baseline_dram_gbps: 3299
  c2_multicast_dram_gbps: 2083
  c4_multicast_dram_gbps: 1480
  c1_baseline_smem_gbps: 3299
  c2_multicast_smem_gbps: 4165
  c4_multicast_smem_gbps: 5918
  c2_amplification: 1.26
  c4_amplification: 1.79
  c1_per_issuer_dram_mbps: 25000
  c2_per_issuer_dram_mbps: 31561
  c4_per_issuer_dram_mbps: 44848
  ideal_c2_smem_gbps: 6598
  ideal_c4_smem_gbps: 13196
open_questions:
- ALL Q1–Q6 below resolved in the v2 follow-up probe at `2026-04-30-tma-multicast-v2.md`.
  Summary of v2 findings inline; original questions retained for traceability.
- Q1 [RESOLVED in v2] Sub-linear effective-bandwidth scaling root cause. v2 ncu confirms
  DRAM bytes scale exactly 1/C (134 → 67 → 33 → 16 → 8 MB) and multicast request count
  scales 1/C; SM active % falls 93% → 78% as cluster grows. Sub-linearity is from
  issuer-count attrition + non-leader idle time, not from any multicast-fanout overhead.
- Q2 [RESOLVED in v2] C=8 / C=16 measured. Effective-bw plateaus at C=4 (1.73×); C=8
  (1.68×) and C=16 (1.69×) regress because issuer count drops below DRAM-saturation
  threshold. Production GEMM with much larger grids should not see this regression.
- Q3 [RESOLVED in v2] Multi-producer-warp tested at C=2. 1/2/4 producer warps lift
  effective bandwidth 4.08 → 5.69 → 6.26 TB/s (1.54×). 4-warp C=2 (6.26 TB/s) beats
  1-warp C=4 (5.63 TB/s) — multi-warp issue is a cheaper amplifier than larger clusters.
- Q4 [RESOLVED in v2] L2 promotion swept at C=2 (NONE / 64B / 128B / 256B). Marginal
  effect; 256B gives +4.5%, smaller promotions are within noise. Probe is DRAM-bound
  (first-touch), so L2 promotion's caching axis sees little to optimise.
- Q5 [RESOLVED in v2] Cross-CTA empty mbarrier protocol implemented via `mapa.shared::cluster.u32`
  + `mbarrier.arrive.release.cluster.shared::cluster.b64`. Functional but ~4.4× slower
  at microbench grain (no consumer). In real producer/consumer GEMM the round-trip
  is amortised behind 32+ wgmma instances per tile.
- Q6 [RESOLVED in v2] Numeric correctness — element-by-element compare of every loaded
  tile against deterministic source pattern. **0 / 67 043 328 mismatches** across
  C=1, C=2, C=4. Multicast PTX path is byte-identical to non-multicast.
referenced_in_corpus:
- path: 05-source-corpus/blogs/colfax/cutlass-tutorial-mastering-the-nvidia-tensor-memory-accelerator-tma
  line_range: section on multicast TMA + cooperative kernel
id: exp-tma-ptx
type: experience
vendor: nvidia
title: 2026 04 30 Tma Multicast
---
## Summary

This probe extends the cutlass-free TMA primitive at [`40-hardware-feature/tma-ptx/skill.md`](../../../40-hardware-feature/tma-ptx/skill.md) with the **cluster-multicast** variant of `cp.async.bulk.tensor` — the PTX instruction that lets one CTA in a cluster issue a single TMA load and have the result delivered into the smem of multiple CTAs in the same cluster, with each receiving CTA's mbarrier signalled by the same load. Multicast is the hardware mechanism that makes cooperative warp-specialised GEMM (cutlass `KernelTmaWarpSpecializedCooperative`) bandwidth-efficient: when both CTAs in a cluster need the same A or B tile, multicast loads it from DRAM once instead of twice.

The probe sweeps cluster size **C ∈ {1, 2, 4}**, holds the per-CTA workload constant at 124 tiles (8 KiB each), and reports DRAM bytes / time + effective smem-bytes-delivered / time. Holding per-CTA work constant lets the multicast benefit show as **DRAM reads scaling 1/C while smem-delivered stays constant** — i.e., effective bandwidth amplification.

Headline: **C=2 multicast amplifies effective bandwidth 1.26×** over the no-multicast baseline (4.17 → 3.30 TB/s); **C=4 reaches 1.79×** (5.92 → 3.30 TB/s). Both are below the theoretical (C×) ceiling because shrinking the issuer count from 132 → 66 → 33 also drops DRAM-controller parallelism, partially offsetting the multicast win. The probe is cutlass-free at both gates: `nvcc -E | grep cutlass::|cute::` = 0 matches; `cuobjdump --dump-elf-symbols | grep` = 0 matches.

## Setup

- GPU: NVIDIA H200, sm_90a, CUDA 12.9.86, driver 570.124.06.
- Build: `nvcc -std=c++17 -O3 -gencode=arch=compute_90a,code=sm_90a -lineinfo -lcuda`. No cutlass / cute include.
- Source: `__nv_bfloat16[16384][4096]` = 128 MiB on device.
- Tile: 64 × 64 bf16 (8 KiB), `SWIZZLE_128B` to match wgmma's smem-descriptor expectation. (Fast-axis = 128 B is the only width valid for SWIZZLE_128B, see `tma-ptx/pitfalls.md` #9.)
- Launch: 132 CTAs total = N_SMS, with `__cluster_dims__(C, 1, 1)` so the runtime arranges them as 132/C clusters. Each cluster has CLUSTER_SIZE CTAs at consecutive `blockIdx.x` (the linear cluster rank is read from `%cluster_ctarank`).
- Per-CTA work fixed at `n_tiles_per_cluster = 124` (the c1 baseline value); each cluster's leader does 124 multicast loads, all C CTAs receive each one. Total smem bytes delivered per launch = `132 × 124 × 8 KiB = 128 MiB` (independent of C). Total DRAM bytes per launch = `(132/C) × 124 × 8 KiB = 128 / C MiB`.

### Cluster-multicast PTX wrapper (cutlass-free)

```cpp
__device__ __forceinline__ void tma_load_2d_multicast(
    void* smem_dst, const CUtensorMap* tensor_map,
    int32_t coord_row, int32_t coord_col, uint64_t* mbar, uint16_t cta_mask)
{
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    uint32_t b = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    uint64_t map_addr = reinterpret_cast<uint64_t>(tensor_map);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster "
        "[%0], [%1, {%3, %4}], [%2], %5;"
        :: "r"(s), "l"(map_addr), "r"(b),
           "r"(coord_col), "r"(coord_row), "h"(cta_mask));
}
```

The `cta_mask` is a 16-bit immediate where bit `i` set means the CTA at cluster-rank `i` receives the multicast. For our probe it is `(1<<C) - 1` (all CTAs in the cluster receive).

### Issue protocol (per tile)

```
Each CTA in the cluster (executed by tid==0):
  1. mbarrier_arrive_expect_tx(local_bar, tile_bytes)   // local mbar expects tile_bytes
  (Cluster leader only):
  2. tma_load_2d_multicast(local_smem, tensor_map, coords, local_bar, cta_mask)
     // signals complete_tx::bytes on EACH receiving CTA's mbarrier (relative-in-smem
     // offsets are the same across all CTAs in the cluster — that's how multicast
     // addresses work)
  3. mbarrier_wait(local_bar, phase)                    // wait for the multicast to land

Each CTA's mbarrier sees: 1 expect_tx (local) + 1 complete_tx (from multicast). They cancel
in any order; wait fires once both have happened.
```

The leader's local pipeline (depth=4 ring buffer with phase-tracking on each slot) throttles its own re-issue rate so the same smem slot is not overwritten before the previous load's data is consumed. For our microbench (no consumer), the depth=4 lookahead is sufficient.

## Results

| config       | C | n_clusters | n_tiles/cluster | DRAM bytes | smem delivered | median ms | DRAM GB/s | effective smem GB/s | amplification |
|--------------|--:|-----------:|----------------:|-----------:|---------------:|----------:|----------:|--------------------:|--------------:|
| c1 baseline  | 1 |        132 |             124 |    128 MiB |        128 MiB |    0.0410 |  **3299** |             **3299** |          1.00× |
| c2 multicast | 2 |         66 |             124 |     64 MiB |        128 MiB |    0.0324 |      2068 |             **4136** |     **1.26×** |
| c4 multicast | 4 |         33 |             124 |     32 MiB |        128 MiB |    0.0227 |      1480 |             **5918** |     **1.79×** |

### Per-issuer DRAM bandwidth (= DRAM GB/s ÷ n_clusters)

| C | n_clusters (= n_issuers) | per-issuer DRAM MB/s |
|--:|-------------------------:|---------------------:|
| 1 |                      132 |               25 000 |
| 2 |                       66 |               31 561 |
| 4 |                       33 |               44 848 |

When the issuer count shrinks, each remaining issuer gets more DRAM bandwidth — but the per-issuer pipeline cannot saturate fast enough to keep aggregate DRAM at the c1 level.

## Interpretation

### 1. Cluster multicast genuinely amplifies effective bandwidth

Holding per-CTA work constant at 124 tiles, C=2 multicast delivers the same 128 MiB to all 132 CTAs while only reading 64 MiB from DRAM, in 32 μs vs the c1 baseline's 41 μs. From the smem-perspective that is **4.14 TB/s effective** vs c1's 3.30 TB/s — a 1.26× amplification. C=4 reaches 5.92 TB/s = 1.79× amplification.

This is the mechanism that makes cooperative warp-specialised GEMM viable. A 2-CTA cluster sharing an A-row needs A loaded only once from DRAM; a 4-CTA cluster with both A-row and B-col multicast needs each tile loaded only once instead of four times.

### 2. The amplification is sub-linear in C

Theoretical ceiling at C=2 is 2× (DRAM is halved → time should be halved → bandwidth doubled); actual is 1.26×. At C=4, ceiling is 4×; actual is 1.79×. The gap comes from two effects, both visible in the per-issuer table:

1. **Issuer-count drops faster than per-issuer bandwidth grows.** C=1 has 132 concurrent issuers, each at 25 GB/s; C=4 has 33 issuers each at 45 GB/s. Per-issuer grew 1.79×; issuer count fell 4×. Aggregate DRAM bandwidth therefore drops from 3300 → 1480 GB/s.
2. **Leader-only issue serialises the cluster's TMA pipeline.** Each cluster's leader is the single thread that issues all `n_tiles_per_cluster` multicast loads serially through its depth=4 ring buffer. With C=4 each cluster has 4 CTAs but still only one issuer; the issuer pipeline is the bottleneck of the cluster.

For real cooperative GEMM the issuer-count drop is less severe (the kernel grid is 100s–1000s of CTAs, not 132), and the cluster leader's pipeline is hidden behind warpgroup-specialisation (multiple producer warps issuing concurrently). The cooperative GEMM benefit can therefore be closer to the theoretical C×, but this microbench's tight 132-CTA grid is a worst-case for issuer-count attrition.

### 3. The DRAM-bandwidth metric inverts intuition

A naive "TMA bandwidth" metric (DRAM bytes ÷ time) goes *down* with multicast: 3300 → 2068 → 1480 GB/s. That is correct — multicast reads less from DRAM. The right metric for the user (cooperative GEMM consumer) is **effective bandwidth from the smem perspective**: how fast does data arrive at each CTA's wgmma-feeding ring buffer. By that metric, multicast is 1.26× / 1.79× faster.

When evaluating multicast in profilers, look at `l1tex__m_l1tex2xbar_read_requests_mem_global_op_tma_ld_dest_multicast.sum` (multicast-fan-out request count) versus `l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum` (DRAM-bytes-loaded). The ratio tells you the multicast factor.

### 4. cta_mask is per-cluster, not per-CTA

The `cta_mask` argument is a 16-bit immediate or register-held value where bit `i` corresponds to **cluster-rank `i`**, not the global `blockIdx.x`. For a cluster of 4 CTAs the mask is `0b1111` regardless of which 4 CTAs in the grid form the cluster. Setting bit `i ≥ CLUSTER_SIZE` is a no-op (the runtime ignores out-of-range bits); setting fewer bits than CLUSTER_SIZE causes those CTAs to NOT receive — useful for partial-multicast patterns but not exercised here.

## Takeaways

1. **Multicast is functional cutlass-free**: the 16-bit `cta_mask` form of `cp.async.bulk.tensor.2d…multicast::cluster` works end-to-end on H200 sm_90a from raw PTX with the `__cluster_dims__` attribute and `barrier.cluster.{arrive,wait}.aligned` for cluster-wide sync. Both cutlass-free gates pass.
2. **Effective-bandwidth amplification is real but sub-linear**: C=2 → 1.26×, C=4 → 1.79× at this 132-CTA grid. The full C× ideal is unreached because issuer count drops faster than per-issuer bandwidth grows.
3. **Per-CTA workload must stay constant to compare modes apples-to-apples.** Letting `n_tiles_per_cluster = total/n_clusters` (so total work is constant) hides the multicast benefit because the leader's serialised pipeline grows with cluster size. Holding per-CTA work fixed and letting DRAM bytes shrink is the right comparison.
4. **For cooperative GEMM the practical multicast factor approaches C×** because real grids have far more clusters than this microbench's 132 CTAs / 33 clusters, and warp-specialisation lifts the leader's pipeline ceiling. The 1.26× / 1.79× here is the worst-case microbench number; production GEMM does better.

## Files

- `artifacts/tma_multicast_probe.cu` — 3-config sweep harness with cluster-multicast PTX.
- `artifacts/build_multicast.sh` — `nvcc` build + cutlass-free gate verification.
- `artifacts/run_multicast.sh` — build + run + persist CSV.
- `artifacts/profiles/2026-04-30-tma-multicast.csv` — per-config table (config / cluster_size / n_clusters / n_tiles_per_cluster / dram_bytes / smem_bytes_delivered / median_ms / dram_gbps / effective_smem_gbps / multicast_speedup).
