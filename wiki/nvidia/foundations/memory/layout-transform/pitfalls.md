---
title: Layout Transform — Pitfalls
status: draft
id: pitfall-layout-transform
type: pitfall
vendor: nvidia
---
# Layout Transform — Pitfalls

## P1. Bank conflicts on the transpose tile

**Symptom**: Smem-tiled transpose delivers ~70 % of expected bandwidth; NCU shows bank conflicts on the smem column read.

**Root cause**: `__shared__ float tile[TILE][TILE]` with `TILE = 32` puts every column's 32 elements on the same bank. The `tile[threadIdx.x][threadIdx.y]` read hits 32-way bank conflicts.

**Fix**: Pad the inner dimension: `__shared__ float tile[TILE][TILE + 1]`. Measured on H200 (sibling skill `shared-memory-cache` probe): unpadded → 16.3 M bank conflicts, padded → 33 K (488× drop; 1.71× speedup).

**Source**: Best Practices Guide §10.2.3.1 (L722-L727); cross-ref sibling skill `shared-memory-cache` pitfall P8 (measured on H200).

## P2. Edge-tile out-of-bounds for non-TILE-multiple dimensions

**Symptom**: Correct on 4096² but corrupts memory on 4097². `compute-sanitizer` reports out-of-bounds on both the load and the store.

**Root cause**: Tile-based transpose indexing `blockIdx.x * TILE + threadIdx.x` can exceed the matrix dim on the rightmost / bottom blocks.

**Fix**: Predicate both load and store with the actual dim: `if (x < width && y < height) ...`. Both the "load to smem" and the "store from smem" branches must be guarded.

**Source**: CUDA C++ Programming Guide §2.2.4.2.1 (L1484-L1540).

## P3. Transform overhead exceeds the amortized benefit

**Symptom**: Total pipeline runtime went UP after introducing AoS→SoA. Profile shows the conversion kernel consumes most of what the downstream kernels saved.

**Root cause**: The data buffer is touched by too few downstream kernels (often just 1). The one-pass conversion kernel reads + writes the full buffer; if the downstream benefit is less than that cost, the transform is a loss.

**Fix**: Only transform when ≥ 3 downstream kernels will use the new layout (or one hot-path kernel that runs many times per second in a server loop). For single-use data, stay in the ingress layout. See skill §"When NOT to use".

**Source**: Best Practices Guide §10.2.1.4 (L606-L634).

## P4. Pitch mismatch between host and device buffers

**Symptom**: After `cudaMemcpy2D` the device buffer contains garbage or shifted rows.

**Root cause**: `cudaMallocPitch` on the device returns an architecture-dependent pitch (often 512 B), but the host buffer is tightly packed (`width * sizeof(elem)`). Using `cudaMemcpy` or passing the wrong pitch to `cudaMemcpy2D` scrambles the copy.

**Fix**: Always use `cudaMemcpy2D(dst, devicePitch, src, hostPitch, widthBytes, height, kind)`. For round-trips, keep the two pitches explicit — do not derive the host pitch from the device pitch or vice versa.

**Source**: CUDA C++ Programming Guide §2.2.4.1 (L1379-L1411).

## P5. Smem-transpose `__syncthreads` cost dominates small kernels (legacy sandbox finding)

**Symptom**: After adding shared-memory staging to an elementwise-copy-style kernel, barrier stalls rose from 0 % to ~20 %; roofline efficiency dropped from 82.6 % to 65.0 % even though wall-clock was 85 % faster.

**Root cause**: The baseline's "high roofline %" was misleading — it counted inefficient strided transactions as "utilized bandwidth". The smem-staged kernel moves less HBM traffic but pays the barrier cost.

**Fix**: Interpret roofline with care after layout transforms. Compare wall-clock latency against effective HBM traffic, not against peak-bandwidth percentages. If the baseline reported high roofline % from wasted transactions, the roofline comparison is not apples-to-apples.

**Source**: KernelPilot legacy L3 sandbox run (2026-04-04). Retained as anecdotal pending H200 re-measurement.

## P6. `prmt.b32` can shift bottleneck without improving latency

**Symptom**: Inline `prmt.b32` replaced a couple of shifts + ANDs but latency barely moved; NCU reports rising `long_scoreboard` stalls.

**Root cause**: In a memory-bound kernel, the critical path is the load. Swapping an ALU sequence for a single `prmt.b32` does not reduce the load-wait, it just shifts the reported bottleneck from "compute" to "memory latency", making the profile *look* worse without actually regressing.

**Fix**: Only pursue `prmt.b32` in compute-bound sub-word-format kernels. For memory-bound kernels, the layout transform itself should target the memory access pattern (S1 / S2 / S3), not the register-level permute.

**Source**: KernelPilot legacy L3 sandbox run (2026-04-04). Retained as anecdotal.

## P7. Occupancy regression from transpose tile smem footprint

**Symptom**: Smem-tiled transpose kernel smem footprint rose to 5248 B per block; occupancy dropped from 8 blocks/SM to 3 blocks/SM; small-matrix latency regressed.

**Root cause**: `[TILE][TILE + 1]` padding + multi-tile buffers for batched transpose can cross the per-block smem threshold at which occupancy collapses. On small matrices (< 512² fp32), the reduced occupancy hurts more than the coalescing helps.

**Fix**: Pick tile size per shape regime. For large matrices (> 2048² fp32), `[32][33]` is fine. For small matrices, try `[16][17]` (625 B per single-buffer tile, fits 8+ blocks/SM) or drop the smem stage entirely and eat the coalescing cost.

**Source**: KernelPilot legacy L3 sandbox run (2026-04-05). H200 re-measurement scheduled in pattern T2 / T4 seed tasks (see `20-pattern/cuda-core/transpose/TASK-PACKET.md`).

## P8. `long_scoreboard` stall after transform — now latency-bound

**Symptom**: After the transform, NCU's `long_scoreboard` stall rises from 45 % to 80 %; further kernel tuning shows diminishing returns.

**Root cause**: The layout transform successfully eliminated bandwidth waste, so the kernel is no longer bandwidth-limited — it is now **memory-latency-bound**. Further speedup requires increasing issued-warps-in-flight (raising occupancy or ILP), not reducing bytes.

**Fix**: Pair the layout transform with an occupancy or ILP pass. See `30-skill/compute/ilp/` and `30-skill/compute/occupancy-tuning/`. Watch register-per-thread count: legacy sandbox observed regs/thread 16 → 18 after vectorization, dropping the occupancy ceiling from 16 to 10 warps — the next bottleneck.

**Source**: KernelPilot legacy L3 sandbox run (2026-04-05). Retained as anecdotal; the latency regime is real but the specific numbers need H200 re-measurement.

## P9. Pitched allocation hurts L2 hit rate

**Symptom**: Pitched buffer used by a kernel whose rows are short (e.g. 32 elements) pushes L2 hit rate from 62.7 % down to 50.1 %; net throughput regresses despite better coalescing.

**Root cause**: The pitch padding expands the buffer's footprint; if rows were already short enough that the unpadded buffer fit comfortably in L2, the padded version no longer does, and the cache-miss tax exceeds the coalescing win.

**Fix**: Only use `cudaMallocPitch` when (a) row width in bytes is not already a multiple of 128 AND (b) the kernel's bandwidth-to-L2-hit-rate sensitivity favors the coalescing fix. For already-aligned rows, stay with plain `cudaMalloc`.

**Source**: KernelPilot legacy L3 sandbox run (2026-04-07). Retained as anecdotal; revisit after H200 probe if pitched use-cases land in a downstream skill.
