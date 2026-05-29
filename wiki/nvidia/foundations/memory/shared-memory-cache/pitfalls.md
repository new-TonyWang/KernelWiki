---
title: Shared Memory Cache — Pitfalls
status: draft
id: pitfall-shared-memory-cache
type: pitfall
vendor: nvidia
---
# Shared Memory Cache — Pitfalls

## P1. Missing `__syncthreads()` between smem write and cross-warp read

**Symptom**: Intermittently incorrect results. `compute-sanitizer --tool racecheck` reports a hazard.

**Root cause**: One warp writes into smem, another warp reads from that smem location without a block-level barrier. Warp schedulers run warps in arbitrary order; the reader may observe stale data.

**Fix**: Place `__syncthreads()` after the write phase and before the read phase. Use `__syncwarp(mask)` only when the producer and consumer lanes are in the same warp (same-warp intra-tile shuffles).

**Source**: Best Practices Guide §10.2.3.2 (L728-L732); CUDA C++ Programming Guide §2.2.3.2 (L1130-L1170).

## P2. Excessive smem footprint collapses occupancy

**Symptom**: Nsight Compute occupancy section shows shared memory as the limiting resource; only 1 block/SM reached; kernel is latency-bound rather than bandwidth-bound.

**Root cause**: The chosen tile size claims more smem per block than the architecture's budget allows at the desired blocks-per-SM count. The unified L1/smem cache has a hard per-SM cap (228 KB on H100/H200 for opt-in).

**Fix**: Shrink the tile, split into multiple passes, or process multiple output elements per thread with a smaller tile. Verify with `cudaOccupancyAvailableDynamicSMemPerBlock`. Profile with `-Xptxas=-v` and the NCU "Launch Statistics" section before and after.

**Source**: Best Practices Guide §11.4 (L1132-L1150) — "Effects of Shared Memory".

## P3. Dynamic smem alignment errors when partitioning one buffer into multiple arrays

**Symptom**: Wrong results or CUDA illegal-memory-access when a dynamic-smem buffer is sliced into e.g. a `short` prefix and a `float` suffix.

**Root cause**: The second pointer is computed as `(float*)&s_short[short_count]`. If `short_count` is odd, the pointer is not 4-byte aligned, and PTX `ld.shared.f32` will trap or silently load wrong data.

**Fix**: Round partitions up to each type's alignment manually. Cast through `char*` to reason about bytes rather than elements:

```cuda
extern __shared__ char smem_raw[];
short* s_short  = (short*)smem_raw;
float* s_float  = (float*)(smem_raw + ROUND_UP(short_count * sizeof(short), 16));
```

**Source**: CUDA C++ Programming Guide §2.2.3.2.2 (dynamic smem allocation, L1173+).

## P4. Forgetting the >48 KB opt-in for dynamic smem

**Symptom**: `cudaLaunchKernel` returns `cudaErrorLaunchOutOfResources`; the same configuration worked at 48 KB but fails at 64 KB.

**Root cause**: Per PG §3.2.6 (L4094-L4130), allocations above 48 KB per block require an explicit opt-in via `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)`. The default cap is 48 KB for launch-safety across older architectures.

**Fix**: Call `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_bytes)` once after module load, before the first launch. Query the per-architecture ceiling with `cudaDeviceGetAttribute(&maxopt, cudaDevAttrMaxSharedMemoryPerBlockOptin, device)`.

**Source**: CUDA C++ Programming Guide §3.2.6 (L4094-L4130), §18194.

## P5. Staging smem for single-use data

**Symptom**: Kernel latency rises after "optimizing" it by adding a smem tile. NCU shows no reduction in DRAM bytes; `smsp__warps_issue_stalled_barrier_per_issue_active.pct` rises.

**Root cause**: Each element is loaded to smem and read exactly once, with no reuse, no reorder, and no cross-warp share. The `__syncthreads()` and smem round-trip are pure overhead; hardware L1 would have cached the load for free.

**Fix**: Remove the smem stage. Load directly from global into a register; let L1 absorb the single-shot read. Reserve smem for the three justifying roles (reuse / reorder / cross-warp comm).

**Source**: Best Practices Guide §10.2.3 (L719-L727) — "Effects of Shared Memory".

## P6. Unconditional smem writes when only a subset of threads have real data

**Symptom**: Silent correctness bugs if the "unused" slots contain garbage read by the reduction tail; wasted bandwidth in any case; potential bank conflicts if the unused writes alias banks.

**Root cause**: Block-level load code writes `smem[tid] = in[blockIdx.x * blockDim.x + tid]` without guarding against `tid >= valid_count`.

**Fix**: Guard the write with the validity predicate and pre-zero the unused region before the barrier:

```cuda
smem[tid] = (tid < valid_count) ? in[...] : 0.0f;   // zero-extend, not UB
__syncthreads();
```

Or use `cuda::memcpy_async` with a size argument to do the tail-guarded load cooperatively.

**Source**: CUDA C++ Programming Guide §4.11.1.1 (batching loads in conditional code).

## P7. Occupancy crash from tripled smem in a reuse-thin kernel (legacy sandbox finding)

**Symptom**: Smem footprint rose from 1 KB to 3 KB per block; the smem-limited block count fell from 32 to 21 blocks/SM; `smsp__warps_issue_stalled_barrier_per_issue_active.pct` rose from 0 % to 14 %.

**Root cause**: The kernel's reuse factor was too low to amortize the additional `__syncthreads()` and the occupancy loss.

**Fix**: Confirm a reuse factor ≥ 4× (each smem word read ≥ 4 times) before accepting the footprint increase. If reuse < 4×, either stay in registers or reduce tile size.

**Source**: KernelPilot legacy L3 sandbox run (2026-04-05). Retained here as anecdotal; to be re-measured on H200 in a follow-up probe.

## P8. Unpadded `[TILE][TILE]` smem causes 32-way bank conflicts on column access (measured on H200)

**Symptom**: NCU `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` spikes; `short_scoreboard` stalls rise to 30 %+ when column-indexing the smem tile.

**Root cause**: 32 successive 32-bit words of a row map to 32 distinct banks, but 32 successive row-strided accesses (`smem[0..31][threadIdx.x]`) all land on the same bank.

**Fix**: Declare the tile as `__shared__ float smem[TILE][TILE + 1];`. The extra column shifts each row by one bank, making column reads conflict-free. This is the default idiom for any 2-D smem tile intended to be read along both axes.

**Measured on H200** (4096² fp32 transpose, sm_90a, CUDA 12.9, probe `80-experience/hw-probes/smem-tile-reuse/`):

| Kernel                         | Bank conflicts (ld.sum) | Warp cyc/issued | Median ms |
|--------------------------------|------------------------:|----------------:|----------:|
| `smem_tiled` (unpadded 32×32)  |              16,326,828 |           85.54 |    0.1363 |
| `smem_tiled` (padded 32×33)    |                  33,449 |           42.09 |    0.0796 |

Padding cuts bank-conflict count **488.4×** and delivers another **1.71×** speedup on top of the smem staging itself.

**Source**: Best Practices Guide §10.2.3.1 (L722-L727); H200 probe `2026-04-21-smem-tile-reuse.md` (measured, supersedes earlier KernelPilot L3 sandbox anecdote).

## P9. Dynamic smem is not free for compile-time-known tile sizes

**Symptom**: Replacing `__shared__ float smem[128];` with `extern __shared__ float smem[];` added ~5 % kernel time in a small-tile kernel.

**Root cause**: The compiler can constant-propagate static smem addresses through subsequent `ld.shared` / `st.shared` instructions, whereas dynamic smem forces address computation via base + offset. For small compile-time-known tiles the optimizer gap becomes measurable.

**Fix**: Prefer static `__shared__` whenever the tile size is known at compile time. Only reach for `extern __shared__` when the size must be launch-time-selected, or when you intend to partition a single buffer into multiple typed arrays.

**Source**: KernelPilot legacy L3 sandbox run (2026-04-05). Retained as anecdotal; warrants re-measurement on H200 once sm_90a compiler revs have settled.

## P10. Carveout success without occupancy gain (register-limited kernel)

**Symptom**: `cudaFuncSetAttribute(..., cudaFuncAttributePreferredSharedMemoryCarveout, MaxShared)` did change `launch__occupancy_limit_shared_mem` (e.g. from 3 to 25) but achieved occupancy did not budge because the kernel was register-limited (limit = 2).

**Root cause**: Occupancy is the minimum of the per-resource limits. Lifting the smem ceiling is invisible unless smem was the binding constraint.

**Fix**: Before tuning carveout, identify the binding resource. NCU "Occupancy" section lists `launch__occupancy_limit_{registers, shared_mem, blocks}`; carveout only helps when `shared_mem` is the tightest. Use `__launch_bounds__` or `-maxrregcount` to address a register-limited case instead.

**Source**: KernelPilot legacy L3 sandbox run (2026-04-05). Same re-measurement caveat as above.

## P11. Over-allocating dynamic smem zeros out L1 hit rate

**Symptom**: L1 hit rate drops to near zero; DRAM read traffic rises; smem is only partially used.

**Root cause**: Large dynamic smem claims (close to the 228 KB opt-in ceiling on H100/H200) leave no room for L1 in the unified data cache. If only part of the claim is actually written, the kernel pays the L1 cost for no benefit.

**Fix**: Claim only what you will use. `cudaFuncAttributePreferredSharedMemoryCarveout` should reflect the *actually written* smem footprint, not the dynamic-smem *ceiling*.

**Source**: KernelPilot legacy L3 sandbox run (2026-04-05). Same re-measurement caveat.
