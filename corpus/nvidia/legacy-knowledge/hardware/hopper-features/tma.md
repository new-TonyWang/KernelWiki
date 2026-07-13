# Tensor Memory Accelerator (TMA)

Hardware DMA engine introduced in Hopper (sm_90) for asynchronous data transfer between global memory (GMEM) and shared memory (SMEM). TMA offloads address computation, stride handling, and out-of-bounds predication from threads to dedicated hardware.

## Working Principle

- **Hardware DMA engine**: TMA operates entirely in the async proxy -- once a single thread issues the TMA instruction, the hardware DMA unit performs the copy without any further thread involvement. Other threads/warps are free to do compute.
- **Single-thread issue**: Only one thread per CTA needs to issue the TMA operation (typically thread 0 or an elected leader). This is fundamentally different from cooperative copy patterns where all threads participate.
- **Descriptor-based**: TMA uses a tensor map descriptor (`CUtensorMap` / `cuTensorMap`) created on the host that encodes the tensor layout (shape, strides, element type, swizzle mode). This descriptor is passed to the kernel as `__grid_constant__ const`.
- **Automatic predication**: Out-of-bounds accesses are automatically handled -- remainder tiles are zero-filled without any thread-side predication logic.

## Supported Operations

| Operation | PTX Instruction | Direction | Completion Mechanism |
|-----------|----------------|-----------|---------------------|
| TMA Load | `cp.async.bulk.tensor.Nd.shared::cluster.global` | GMEM -> SMEM | mbarrier (`complete_tx::bytes`) |
| TMA Store | `cp.async.bulk.tensor.Nd.global.shared::cta` | SMEM -> GMEM | bulk async-group (`bulk_group`) |
| TMA Reduce | `cp.reduce.async.bulk.tensor` | SMEM -> GMEM (with reduction) | bulk async-group |
| TMA Load Multicast | `cp.async.bulk.tensor` with `.multicast::cluster` | GMEM -> multiple SMEMs | mbarrier |

**Dimensionality**: 1D, 2D, 3D, 4D, 5D tile copies supported. The `.dim` modifier in the PTX instruction specifies the tensor rank.

**Load modes**: `.tile` (preserves layout), `.tile::gather4` / `.tile::scatter4` (4-row gather/scatter), `.im2col` (unrolls spatial dimensions for convolution).

## TMA Descriptor (Tensor Map)

Created on host via `cuTensorMapEncode` API (or CUTLASS `make_tma_copy`). Encodes:

- **Global tensor**: base pointer, shape, strides, element size
- **Tile (box) dimensions**: the SMEM tile shape to copy per operation
- **Swizzle mode**: optional data shuffling for bank-conflict-free SMEM access
- **Out-of-bounds fill**: zero-fill for elements outside tensor bounds
- **Interleave mode**: layout permutation between GMEM and SMEM

The descriptor is passed to the kernel as `__grid_constant__ const` parameter. Each tensor being copied requires its own descriptor.

## Synchronization

### TMA Load (GMEM -> SMEM): mbarrier-based

1. Initialize mbarrier: `mbarrier.init` with arrival count (typically 1 for single-thread issue)
2. Set expected transaction bytes: `mbarrier.arrive.expect_tx` with total bytes to be copied
3. Issue TMA: `cp.async.bulk.tensor` with the mbarrier -- hardware performs `complete_tx` on mbarrier upon completion
4. Wait: all threads call `mbarrier.try_wait.parity` on the mbarrier phase

After wait completes, SMEM writes from TMA are visible to all threads that participated in the wait.

### TMA Store (SMEM -> GMEM): fence + async-group

1. All threads write data to SMEM
2. **Fence**: `fence.proxy.async.shared::cta` -- ensures SMEM writes by threads are visible to the TMA engine (async proxy)
3. Issue TMA store (single thread)
4. Optionally: `cp.async.bulk.commit_group` + `cp.async.bulk.wait_group` to wait for store completion

Key difference: TMA Load syncs **after** the operation (mbarrier wait); TMA Store syncs **before** the operation (proxy fence).

## Multicast

Single TMA load broadcasts the same GMEM tile to shared memory of multiple CTAs within a threadblock cluster.

- **Requirement**: participating CTAs must be in the same cluster
- **Mechanism**: a `uint16` bitmask (`ctaMask`) specifies which CTAs receive the data; each bit position corresponds to a CTA rank within the cluster
- **mbarrier signal is also multicast**: the completion signal arrives at all participating CTAs' mbarriers
- **Slice partitioning**: each CTA uses its `block_rank_in_cluster` to determine which portion of the tile to load via `get_slice(ctaid)`, so that the full tile is assembled across all CTAs
- **L2 cache benefit**: multicast guarantees L2 cache hits -- the data is fetched from GMEM once and distributed to multiple SMs
- **Max cluster size**: up to 16 CTAs (non-portable), 8 CTAs (portable)
- **Optimized for sm_90a**: may have significantly reduced performance on other targets

Typical use case: GEMM where an input matrix column tile is shared across multiple row-tile CTAs (or vice versa).

## Key Constraints and Limitations

| Constraint | Detail |
|-----------|--------|
| **16-byte stride alignment** | All non-contiguous strides must be multiples of 16 bytes. E.g., for FP32 row-major `(M, N)` with stride `(N, 1)`, requires `N % 4 == 0`. |
| **128-byte SMEM alignment** | Shared memory destination must be 128-byte aligned. |
| **Descriptor immutability** | Tensor map is `__grid_constant__ const` -- cannot be modified in device code. |
| **Cluster requirement for multicast** | Multicast requires non-trivial cluster dimensions; cluster dims must evenly divide grid dims. |
| **Max 5 dimensions** | Tensor rank limited to 5D. |
| **Async proxy** | TMA executes in the async proxy; cross-proxy fences (`fence.proxy.async`) are needed to synchronize between generic and async proxy. |
| **Single-thread issue** | Only one thread should issue TMA per operation -- multiple threads issuing the same TMA will multiply the transfer. |

## Optimization Relevance

- **Register efficiency**: TMA offloads address computation to hardware. Producer warps using TMA need very few registers (as low as 24-40), enabling aggressive register reallocation to consumer warps via `setmaxnreg`.
- **Warp specialization enabler**: because TMA is single-thread-issued and fully async, it pairs naturally with warp-specialization where dedicated producer warps issue TMA while consumer warps run WGMMA.
- **Eliminates predication overhead**: automatic out-of-bounds handling removes the need for per-thread boundary checks.
- **Swizzle for bank-conflict-free access**: TMA can apply swizzle patterns during transfer, avoiding shared memory bank conflicts for subsequent WGMMA or MMA consumption.
