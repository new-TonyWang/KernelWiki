# Distributed Shared Memory (DSMEM)

## Overview

Distributed Shared Memory is a Hopper (SM90+) feature that allows thread blocks within a **Thread Block Cluster** to directly access each other's shared memory using load, store, and atomic operations. The shared memory of all blocks in a cluster is logically unified into a single distributed address space, eliminating the need to route inter-block data through global memory.

## Hardware Implementation

- A **dedicated SM-to-SM network** connects SMs within a GPC (GPU Processing Cluster).
- All blocks in a cluster are co-scheduled on SMs in the same GPC, guaranteeing physical proximity.
- The SM-to-SM interconnect provides fast, low-latency access to remote shared memory segments.
- Each SM still has its own physical shared memory (up to **228 KB** on H100, from a combined 256 KB L1/SMEM budget). DSMEM is the logical union of these per-SM segments across the cluster.

## Access Method

### Address Mapping via `map_shared_rank`

At the CUDA level, all DSMEM segments from all blocks in the cluster are mapped into the **generic address space** of each thread. The Cooperative Groups API provides the mapping function:

```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

cg::cluster_group cluster = cg::this_cluster();

// Get a pointer to another block's shared memory
int target_rank = 3;  // block rank within the cluster
int *remote_smem = cluster.map_shared_rank(local_smem_ptr, target_rank);

// Now use remote_smem like a regular pointer
atomicAdd(remote_smem + offset, 1);
int val = *remote_smem;  // load from remote block's shared memory
```

**How it works at the PTX level:** The shared memory of all CTAs in a cluster is treated as a single unified address space (PTX "shared state space"). The CTA ID within the cluster occupies high bits of the shared memory address. The `map_shared_rank` function translates a local shared memory address to the corresponding address in another block's SMEM segment by adjusting these high-order bits.

### Additional API

| Method | Description |
|--------|-------------|
| `cluster.map_shared_rank(addr, rank)` | Returns pointer to the variable at `addr` in block `rank`'s shared memory |
| `cluster.query_shared_rank(addr)` | Returns the block rank that owns the given shared memory address |

## Latency and Bandwidth Characteristics

| Access Type | Relative Latency | Notes |
|-------------|-------------------|-------|
| Local shared memory (same SM) | Lowest | ~20-30 cycles, same as traditional SMEM |
| Remote DSMEM (within cluster) | Medium | Via dedicated SM-to-SM network in GPC |
| Global memory (HBM) | Highest | ~200-400 cycles to HBM |

**Key performance numbers (H100 whitepaper):**
- DSMEM data exchange between thread blocks is approximately **7x faster** than routing through global memory.
- DSMEM can also be accessed via **asynchronous copy operations** synchronized with shared memory-based barriers (Asynchronous Transaction Barriers), enabling overlap of data movement and computation.

## Typical Use Cases

### 1. Sharing A/B Matrices in GEMM (with TMA Multicast)

The most impactful use case. In a tiled GEMM:

- CTAs in the same cluster row share the same A operand tile.
- CTAs in the same cluster column share the same B operand tile.
- **TMA Multicast** loads a single tile from global memory and places it into the SMEM of all participating CTAs simultaneously.
- This reduces global memory traffic by the multicast factor (e.g., a 4x4 cluster reduces A-tile loads by 4x along the N dimension and B-tile loads by 4x along the M dimension).

```
Cluster shape (2,2,1) example:
- 4 CTAs handle a 2x2 block of output tiles
- Without multicast: 8 tile loads (each CTA loads its own A and B)
- With multicast: 4 tile loads (each unique tile loaded once, multicast to sharing CTAs)
```

### 2. Distributed Histogram

When histogram bins exceed one block's shared memory capacity, distribute bins across cluster blocks:

```cpp
// Each block owns bins_per_block histogram bins in shared memory
// Total distributed bins = cluster_size * bins_per_block

int dst_block_rank = binid / bins_per_block;
int dst_offset = binid % bins_per_block;
int *dst_smem = cluster.map_shared_rank(smem, dst_block_rank);
atomicAdd(dst_smem + dst_offset, 1);
```

This provides an intermediate tier between per-block shared memory atomics and global memory atomics.

### 3. Halo Exchange / Stencil Computations

Neighboring blocks can directly read boundary elements from adjacent blocks' shared memory without a global memory round-trip.

## Integration with TMA Multicast

TMA Multicast is the primary mechanism for efficiently populating DSMEM:

1. A **single TMA instruction** can load a tensor tile from global memory and place it into the shared memory of **multiple CTAs** specified by a bitmask (`ctaMask`).
2. The bitmask is up to 16 bits (max cluster size on Blackwell), with each bit indicating a participating CTA.
3. Each CTA loads a **portion** of the shared tile; TMA multicast distributes the data to all participants. With N participating CTAs, each loads 1/N of the data.
4. Synchronization uses **Asynchronous Transaction Barriers** (`mbarrier`): the TMA arrives at each participating CTA's barrier, and CTAs wait until the expected transaction byte count is met.

```
PTX instruction format:
cp.async.bulk.tensor.dim.shared::cluster.global.tile
    .mbarrier::complete_tx::bytes.multicast::cluster
    [dstMem], [tensorMap, tensorCoords], [mbar], ctaMask
```

**Synchronization pattern:**
```cpp
// Set expected transaction bytes on the barrier
cute::set_barrier_transaction_bytes(mbar, total_tile_bytes);

// Each CTA issues its TMA multicast with a bitmask
copy(tma_atom.with(mbar, multicast_mask), src_tile, dst_smem);

// Wait for all participating TMAs to complete
cute::wait_barrier(mbar, phase_bit);
```

## Constraints and Limitations

1. **Requires Thread Block Clusters:** DSMEM is only available when blocks are launched as part of a cluster (Compute Capability >= 9.0).

2. **All blocks must be alive:** DSMEM access requires all thread blocks in the cluster to exist. Use `cluster.sync()` before first DSMEM access to guarantee all blocks have started.

3. **Lifetime management:** All DSMEM operations targeting a block's shared memory must complete before that block exits. If block A reads block B's SMEM, block B must not exit until the read is complete. Use `cluster.sync()` before exit.

4. **SMEM size is still per-block:** Dynamic and static shared memory size specifications apply per block. The total DSMEM is `cluster_size * per_block_smem`. There is no way to allocate "cluster-wide" shared memory directly.

5. **Cluster size limits occupancy:** Larger clusters require more SMs to be co-scheduled, reducing scheduling flexibility. Use `cudaOccupancyMaxActiveClusters()` to evaluate the trade-off.

6. **Grid dimension alignment:** Grid dimensions must be exact multiples of cluster dimensions.

7. **Remote access is slower than local:** While DSMEM is much faster than global memory (~7x), accessing remote SMEM through the SM-to-SM network is still slower than accessing the local SM's shared memory. Minimize remote accesses where possible; prefer TMA multicast to populate local SMEM copies.

8. **Atomics on remote DSMEM:** Supported but may have higher latency than local shared memory atomics. Use when the alternative is global memory atomics.
