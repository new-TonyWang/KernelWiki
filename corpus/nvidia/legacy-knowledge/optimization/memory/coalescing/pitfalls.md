# Coalescing -- Pitfalls

## P1: Column-Major Access in Row-Major Array
**Symptom**: Bandwidth drops to 1/N of peak where N is the matrix width. Nsight Compute shows high transaction count per warp.
**Detection**: Check `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` vs expected. If consecutive threads access addresses N*4 bytes apart, access is fully uncoalesced.
**Fix**: Use shared memory to stage a tile: load row-major into shared memory, then read column-major from shared memory. Alternatively, transpose the data layout.
**Source**: Best Practices Guide, Section 10.2.1.4 (Strided Accesses)

## P2: Array-of-Structures Causing Strided Access
**Symptom**: Each field access by consecutive threads hits different cache lines. Effective bandwidth is 1/sizeof(struct) of peak.
**Detection**: Profile shows high bytes transferred per element. Check data layout for interleaved fields.
**Fix**: Convert to Structure-of-Arrays (SoA) layout. Each field becomes a separate contiguous array.
**Source**: Best Practices Guide, Section 10.2.1.4 (Strided Accesses)

## P3: Block Size Not Multiple of Warp Size
**Symptom**: Under-populated warps waste execution resources; partial warps may access non-contiguous memory.
**Detection**: Check launch configuration. If blockDim.x is not a multiple of 32, warps are under-populated.
**Fix**: Set blockDim.x to a multiple of 32 (128 or 256 recommended).
**Source**: Best Practices Guide, Section 11.3 (Thread and Block Heuristics)

## P4: Misalignment from Sub-Array Offsets
**Symptom**: ~20% bandwidth loss compared to aligned access. Extra 32-byte transactions needed per warp.
**Detection**: Check if data pointer + offset is aligned to 32 bytes. Nsight Compute shows 5 transactions per warp instead of 4.
**Fix**: Align starting offsets to 32-byte boundaries. Process unaligned head elements separately, then switch to aligned bulk processing.
**Source**: Best Practices Guide, Section 10.2.1.2 (Sequential but Misaligned Access)

## P5: Indirect (Gather/Scatter) Access Patterns
**Symptom**: Random access pattern causes 32 separate transactions per warp (worst case). Bandwidth drops to ~3% of peak.
**Detection**: Check if index array is used for indirection (data[index[tid]]). Profile shows very high sector count.
**Fix**: Sort indices to improve locality, use shared memory as a staging buffer, or restructure the algorithm to avoid indirect access. If unavoidable, GPUs handle random access better than CPUs due to latency hiding.
**Source**: Best Practices Guide, Section 10.2.1 (Coalesced Access to Global Memory)

## P6: Coalesced access dropped L1 hit rate from 64% to 0% and L2 from 92% to 50% — this is expected and healthy (uncoalesced patterns artificially inflate cache hit rates by redundantly fetching the same lines), but could be misread as a regression if only cache metrics are checked (discovered in verification)

**Symptom**: Coalesced access dropped L1 hit rate from 64% to 0% and L2 from 92% to 50% — this is expected and healthy (uncoalesced patterns artificially inflate cache hit rates by redundantly fetching the same lines), but could be misread as a regression if only cache metrics are checked.
**Source**: Level 3 sandbox verification (2026-04-04)

## P7: cudaMallocPitch can hurt performance when row widths are already reasonably aligned or when the padding significantly increases working set size beyond L2 capacity — the coalescing metric improves but wall-clock time gets worse (discovered in verification)

**Symptom**: cudaMallocPitch can hurt performance when row widths are already reasonably aligned or when the padding significantly increases working set size beyond L2 capacity — the coalescing metric improves but wall-clock time gets worse.
**Source**: Level 3 sandbox verification (2026-04-05)

## P8: Shared memory allocation (5248B/block) dropped the occupancy limit from 8 to 3 blocks/SM, and __syncthreads introduced 19 (discovered in verification)

**Symptom**: Shared memory allocation (5248B/block) dropped the occupancy limit from 8 to 3 blocks/SM, and __syncthreads introduced 19.9% barrier stalls — for smaller problem sizes or higher-occupancy kernels, this tradeoff may not pay off.
**Source**: Level 3 sandbox verification (2026-04-05)

## P9: L1 cache hit rate dropped from 22 (discovered in verification)

**Symptom**: L1 cache hit rate dropped from 22.43% to 0.0% in the optimized version — perfect coalescing can eliminate partial-line reuse that previously inflated L1 hits, but this is benign since DRAM throughput doubled and overall performance improved significantly.
**Source**: Level 3 sandbox verification (2026-04-05)

## P10: Shared-memory AoS→SoA transposition can be counterproductive for small/moderate problem sizes where the instruction and synchronization overhead exceeds the coalescing savings; the L1 cache was effectively serving the role of coalescing in the AoS baseline (60% hit rate) (discovered in verification)

**Symptom**: Shared-memory AoS→SoA transposition can be counterproductive for small/moderate problem sizes where the instruction and synchronization overhead exceeds the coalescing savings; the L1 cache was effectively serving the role of coalescing in the AoS baseline (60% hit rate).
**Source**: Level 3 sandbox verification (2026-04-05)

## P11: After fixing coalescing the kernel became so fast (3 (discovered in verification)

**Symptom**: After fixing coalescing the kernel became so fast (3.9us) that it shifted from memory-bound (73% SOL) to underutilized (30.8% SOL), meaning launch overhead now dominates — further optimization requires kernel fusion or larger problem sizes, not more memory tuning.
**Source**: Level 3 sandbox verification (2026-04-05)
