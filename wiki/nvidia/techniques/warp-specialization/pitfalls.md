---
id: algo-warp-specialization-pitfalls
type: algorithm
vendor: nvidia
title: Pitfalls
tags:
- cuda-cpp
evidence_level: spec
source:
- path: spec
  anchor: Algorithm reference
---
# Warp-specialized GEMM mainloop — pitfalls

## 1. Stage count is a smem-budget × TMA-latency-coverage trade-off

Each smem stage holds one tile's A + B in shared memory plus its mbarrier pair. More stages hide longer TMA latency but cost smem (typically 8–32 KiB per stage at production tile sizes). Two stages is the minimum that overlaps producer with consumer at all; 3–4 is the typical sweet spot on H200; beyond 4 is rarely justified — you're hiding latency you don't have. If you change tile size, dtype, or epilogue smem footprint, re-budget the stage count.

## 2. mbarrier `expected_tx` must equal the *sum* of A and B tile bytes

The producer issues two TMA loads per stage (A tile + B tile) but only signals one `mbarrier.arrive.expect_tx`. The expected-tx count must be `(BOX_M*BOX_K + BOX_K*BOX_N) * sizeof(elem)`, not just one of the two. Get this wrong and the consumer wakes either too early (one load not done — corrupt result) or never (over-counted — kernel hangs).

## 3. Phase tracking on each mbarrier alternates 0,1,0,1,... per cycle through stages

For S smem stages, barrier `bar_full[s]` flips its parity once per use of stage s. The consumer wait must use phase `(k / S) & 1` for iteration k; the producer must wait on `bar_empty[s]` with `((k - S) / S) & 1`. A constant phase causes back-to-back iterations to wake on the same flip — silent correctness bug since the data is from the *previous* iteration of stage s.

## 4. Pingpong assumes the K-loop is uniform across tiles

The pingpong variant routes alternating tiles to consumer-A and consumer-B. For this to balance, every tile must have the same K-loop count (= same N_TILES_K). If different tiles have different K (e.g. masked-out rows in attention), pingpong loses balance and one consumer stalls. Use plain WS or cooperative for non-uniform per-tile work.

## 5. Cooperative needs cluster-multicast TMA

The cooperative variant keeps two consumers on the same tile, halving per-consumer wgmma work. Without cluster-multicast TMA (`cp.async.bulk.tensor.shared::cluster.global.tile.bulk_group.cta_group::2`), the cluster's two CTAs each issue their own load — twice the DRAM bandwidth for the same data. Always pair cooperative with cluster ≥ `<2, 1, 1>` and the multicast variant of the TMA instruction. Otherwise cooperative has higher coordination cost than plain WS for no win.

## 6. Plain WS at small clusters does not always beat cooperative

Measured on H200 at 2048³: plain WS (`<1, 1, 1>` cluster) reaches 195.4 TFLOPS; cooperative with multicast (`<4, 2, 1>` cluster) reaches 186.9 TFLOPS — plain wins. Cooperative + multicast pays a fixed coordination cost per cluster that does not amortize until the problem is large enough to fill many cluster grids. For problems ≤ a few clusters' worth of CTAs in M × N, plain WS is the better default. Don't reflexively reach for cooperative because it's the most-recent variant.

## 7. Producer must `wgmma.wait_group 0` before re-issuing TMA into a stage

This is implicit if the producer and consumer are different warps and the mbarrier protocol is correct, but if you collapse them onto the same warp (e.g. for a degenerate single-stage debug build), you must explicitly fence the wgmma completion before the next TMA into the same smem slot — otherwise the new load races with the still-running wgmma's smem reads.

## 8. Cutlass implementation specifics (KernelTmaWarpSpecialized*)

These are pitfalls of cutlass's *implementation* of the algorithm, not of the algorithm itself. Listed here for reference because cutlass's variant naming is widely cited:

- The cutlass `(Cluster, Schedule)` tuple is co-constrained — `KernelTmaWarpSpecializedPingpong` requires cluster `<2, 1, 1>`; `Cooperative` requires `<4, 2, 1>`. Mismatched tuples either fail `gemm.can_implement` or run at degraded throughput.
- There is no flag to disable warp-spec on a `KernelTmaWarpSpecialized*` policy; the on/off A/B requires building two binaries (`KernelTma` for off, `KernelTmaWarpSpecialized` for on).
- `EpilogueSchedule` must match the mainloop's persistence model — pingpong/cooperative epilogues are persistent-aware (`TmaWarpSpecialized*`); plain WS uses `NoSmemWarpSpecialized`. `KernelScheduleAuto` enforces this; explicit-schedule callers must enforce it themselves.
- Pipeline depth is encoded as the `Stages` template parameter; `StageCountAutoCarveout` computes the maximum stages that fit given the epilogue smem footprint. Recompute when you change epilogue / dtype.
- `KernelScheduleAuto` is shape-dependent — two callers with the same shape but different SM targets get different schedules. For reproducibility, set the schedule explicitly.

A cutlass-free implementation (composing `wiki/nvidia/hardware/{tma-ptx, wgmma-ptx}/`) skips all of the above by making each of these knobs an explicit kernel parameter or template argument.
