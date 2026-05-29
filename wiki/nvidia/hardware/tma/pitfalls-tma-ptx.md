---
id: pitfall-tma-ptx
type: pitfall
vendor: nvidia
title: Pitfalls
---
# Hopper TMA via raw PTX — pitfalls

## 1. `-arch=sm_90a` is insufficient for TMA inline PTX (same as wgmma)

`cp.async.bulk.tensor.*` and `mbarrier.try_wait.parity.*` are sm_90a-only. nvcc 12.9 with `-arch=sm_90a` generates an intermediate `compute_90` (no `a`) PTX that ptxas rejects. Use `-gencode=arch=compute_90a,code=sm_90a` instead. See `wgmma-ptx/pitfalls.md` Pitfall #1 for the same issue.

## 2. cuTensorMapEncodeTiled's stride array has rank-1 entries

For a rank-2 tensor (M × N), `global_stride` is a rank-1 (= 1-entry) array — you only specify the stride between rows. The fastest-moving dim (cols, in 2D) has implicit stride 1 element. This is easy to get wrong: passing a 2-element stride array silently produces undefined behavior.

## 3. Dimension order in the TMA map is fastest-moving FIRST

`global_dim[0]` is the fastest-moving dimension (cols for 2D row-major), `global_dim[1]` is the next-slowest. This is opposite to how PyTorch / numpy / many CUDA libraries describe shapes (slowest-first). Get this wrong and the TMA loads transposed data.

## 4. mbarrier expected-tx must equal the TMA's transaction byte count

`mbarrier.arrive.expect_tx [bar], <bytes>` tells the barrier how many bytes to wait for before allowing the consumer to wake. If you under-count, the barrier wakes early and the consumer reads partially-loaded data; if you over-count, the barrier never wakes and the kernel hangs. The bytes must equal `tile_rows × tile_cols × sizeof(elem)`.

## 5. mbarrier wait must use parity tracking

The `mbarrier.try_wait.parity.shared::cta.b64` variant tracks barrier-flip parity. The phase parameter starts at 0 and flips with each completion. For multi-iteration TMA (e.g. K-loop pipelining), the consumer must alternate `phase=0, 1, 0, 1, ...` per iteration; using a constant phase causes consecutive iterations to wake on the same flip.

## 6. `__cvta_generic_to_shared` is the right way to convert smem pointers for inline PTX

CUDA shared-memory pointers are in the generic address space by default (in C++); inline PTX requires the smem-specific 32-bit address. `__cvta_generic_to_shared(ptr)` is the CUDA-runtime intrinsic that does this conversion. Don't try to bit-cast the pointer to `uint32_t` — the address spaces aren't trivially convertible.

## 7. The CUtensorMap struct must be device-accessible

The CUtensorMap is built on host but consumed on device. cutlass does this via a `__grid_constant__ const CUtensorMap` kernel parameter (which puts the 128-byte struct in constant memory); this skill's hello-world uses an explicit device allocation for simplicity. Both work; passing the CUtensorMap by-value as a regular kernel argument **does not** because the struct exceeds the by-value parameter size limit on some toolchains. `__grid_constant__` is the production-canonical pattern.

## 8. cuInit(0) is required before cuTensorMapEncodeTiled

The driver API requires `cuInit(0)` before any `cu*` call. If your program previously only used the runtime API (`cudaMalloc`, etc.), the driver context is implicitly initialized; but `cuTensorMapEncodeTiled` is a driver-API call that may be invoked before any runtime call has triggered the lazy driver init. Calling `cuInit(0)` once at startup is the safe pattern.

## 9. Swizzle mode dictates fast-axis tile bytes — they are not independent

`CU_TENSOR_MAP_SWIZZLE_NB` requires the fast-axis tile to be exactly N bytes; passing any other fast-axis fails `cuTensorMapEncodeTiled` with `CUDA_ERROR_INVALID_VALUE`. Concretely:

| Swizzle | Required fast-axis bytes |
|---|---|
| `NONE` | any 16-byte-aligned width |
| `32B`  | 32 (= 16 bf16 = 8 fp32) |
| `64B`  | 64 (= 32 bf16 = 16 fp32) |
| `128B` | 128 (= 64 bf16 = 32 fp32) |

A "swizzle sweep" at fixed tile-bytes therefore is impossible by spec — what you actually scan is *fast-axis bytes*, not the swizzle layout. Throughput differences across swizzle modes in such a sweep are **dominated by tile-bytes scaling**, not by the swizzle layout itself. To compare swizzle layouts apples-to-apples, hold fast-axis = 128 bytes and compare `SWIZZLE_NONE` vs `SWIZZLE_128B` (both legal at that width). The probe at `80-experience/hw-probes/tma-ptx/2026-04-29-tma-throughput.md` shows they differ by < 0.5 % — swizzle layout itself is bandwidth-neutral; its role is downstream wgmma smem-descriptor compatibility, not raw load bandwidth.

## 10. Tile size, not swizzle, is the throughput knob

Per-tile setup cost (one `mbarrier.arrive.expect_tx` + one `cp.async.bulk.tensor` issue + one `mbarrier.try_wait.parity`) is roughly fixed regardless of tile bytes. Throughput therefore scales with tile bytes until DRAM saturates: at fixed swizzle / depth, growing `box_rows` 8 → 128 lifts bandwidth 0.85 → 3.72 TB/s on H200 (4.4×). Below 8 KiB tiles you are leaving issue overhead on the floor; above 16 KiB tiles you saturate DRAM and further growth is wasted smem.

## 11. Pipeline depth ≥ 4 is required to approach saturation

Single-issue (`depth=1`) caps at ~1.44 TB/s on H200 (30 % of HBM3e peak); `depth=2` at 2.38 TB/s (50 %); `depth=4` at 3.32 TB/s (69 %). Depth > 4 returns less per smem cost and hits the per-warp TMA issue rate ceiling — invest in tile size or multi-warp producers from there.

## 12. Multicast `cta_mask` is a cluster-rank bitmap, not a `blockIdx.x` bitmap

The 16-bit `cta_mask` operand on `cp.async.bulk.tensor.<n>d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster` indexes by **cluster-rank** (`%cluster_ctarank` register, 0..CLUSTER_SIZE-1), not by global `blockIdx.x`. For a cluster of 4 CTAs the mask is `0b1111` regardless of which 4 CTAs in the grid form that cluster. Bits beyond `CLUSTER_SIZE` are ignored; setting fewer bits than `CLUSTER_SIZE` causes those CTAs to NOT receive (useful for partial-multicast patterns).

## 13. Each CTA in the cluster must call `mbarrier.arrive.expect_tx` LOCALLY

The multicast TMA only does the `complete_tx::bytes` half of the contract — it signals each receiving CTA's mbarrier with `tile_bytes` of completed transaction. The `arrive.expect_tx` half (declaring the expectation) must be done **by each receiving CTA on its own local mbarrier** before the wait. The mbarrier counters `expected_tx` and `received_tx` are LOCAL state; the multicast only updates `received_tx` remotely. Skip the local arrive_expect_tx and the wait either spuriously succeeds (received with no expected) or hangs depending on phase parity.

## 14. Cluster sync is required between mbarrier init and first multicast issue

`barrier.cluster.arrive.aligned;` + `barrier.cluster.wait.aligned;` ensures all CTAs in the cluster have called `mbarrier.init.shared.b64` on their local mbarriers before the leader issues a multicast that signals them. Without this fence, the leader's first multicast can fire `complete_tx::bytes` against a non-leader's mbarrier that has not been initialised yet — the non-leader's later wait then sees an inconsistent counter and hangs or returns wrong data. Subsequent iterations are throttled by the leader's own pipeline depth; only the first issue needs the explicit cluster fence.

## 15. Multicast amplifies smem-bandwidth, not DRAM-bandwidth

The DRAM-bandwidth metric (`l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum / time`) goes *down* with multicast — fewer DRAM reads serve the same delivered bytes. The user-facing metric is **effective smem-bandwidth** = (smem bytes delivered to all receiving CTAs) / time. By that measure a cluster of size C delivers up to C× the baseline. Profilers separately report `l1tex__m_l1tex2xbar_read_requests_mem_global_op_tma_ld_dest_multicast.sum` — that counter goes *up* with multicast and confirms the fan-out path is active.
