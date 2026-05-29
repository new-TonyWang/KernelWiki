---
title: Shared Memory as User-Managed Cache
status: verified
evidence_level: measured
cuda_version_tested: '12.9'
driver_version_tested: 570.124.06
toolchain: nvcc 12.9.86 + ptxas 12.9
measured_on: H200-SXM
applies_to_pattern_class:
- cuda-core
applies_to_ops:
- reduction
- normalization
- pooling
- scan
- transpose
- elementwise
requires_sm: '>=7.0'
requires_features:
- static-shared-memory
- dynamic-shared-memory
- smem-l1-carveout
single_kernel_useful: true
source:
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L719-L727
  excerpt: Because it is on-chip, shared memory has much higher bandwidth and lower
    latency than local and global memory - provided there are no bank conflicts between
    the threads. On devices of compute capability 5.x or newer, each bank has a bandwidth
    of 32 bits every clock cycle, and successive 32-bit words are assigned to successive
    banks.
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L728-L732
  excerpt: Shared memory enables cooperation between threads in a block. When multiple
    threads in a block use the same data from global memory, shared memory can be
    used to access the data from global memory only once. Shared memory can also be
    used to avoid uncoalesced memory accesses by loading and storing data in a coalesced
    pattern from global memory and then reordering it in shared memory.
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1132-L1150
  excerpt: 'Effects of Shared Memory: shared memory can be helpful in several situations,
    such as helping to coalesce or eliminate redundant access to global memory. However,
    it can also act as a constraint on occupancy. In many cases, the amount of shared
    memory required by a kernel is related to the block size chosen.'
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L1130-L1170
  excerpt: Shared memory is expected to be much faster than global memory. Any opportunity
    to replace global memory accesses by shared memory accesses should therefore be
    exploited. Static and dynamic shared memory allocations are distinguished by __shared__
    arrays with compile-time-known sizes versus extern __shared__ with size passed
    at launch time.
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L1484-L1540
  excerpt: 'Matrix Transpose Example Using Shared Memory: shared memory will be treated
    as a user-managed cache to stage loads and stores from global memory, resulting
    in coalesced global memory access of both reads and writes.'
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L4094-L4130
  excerpt: 'Configuring L1/Shared Memory Balance: the L1 and shared memory on an SM
    use the same physical resource, known as the unified data cache. An application
    can set the carveout, or preferred shared memory capacity, with the cudaFuncSetAttribute
    function. Kernels relying on shared memory allocations over 48 KB per block must
    use dynamic shared memory and require an explicit opt-in.'
artifacts:
  code: 80-experience/hw-probes/smem-tile-reuse/artifacts/smem_tile_reuse_probe.cu
  build: 80-experience/hw-probes/smem-tile-reuse/artifacts/build.sh
  introspection: 80-experience/hw-probes/smem-tile-reuse/artifacts/device.json
  profile: ''
related_apis:
- __syncthreads
- cudaFuncSetAttribute
- cudaFuncAttributeMaxDynamicSharedMemorySize
- cudaFuncAttributePreferredSharedMemoryCarveout
- cudaOccupancyAvailableDynamicSMemPerBlock
related_skills:
- bank-conflict
- coalescing
- async-copy
- occupancy-tuning
- layout-transform
id: skill-shared-memory-cache
type: skill
vendor: nvidia
tags:
- cuda-cpp
applies_to:
- general
---
## What

**Shared memory** is an on-SM SRAM partition of the unified data cache, addressable only from inside a thread block. A kernel uses it as a **user-managed cache**: the programmer decides which global data to stage into shared memory, when to synchronize, and when to discard the staged copy. Per BP §10.2.3 (L719-L727) "shared memory has much higher bandwidth and lower latency than local and global memory — provided there are no bank conflicts."

Three distinct roles justify the cost of the `__syncthreads()` barrier that almost always accompanies shared memory:

1. **Temporal reuse** — the same global value is read by multiple threads in the block, or by the same thread multiple times (BP §10.2.3.2, L728-L732: "shared memory can be used to access the data from global memory only once").
2. **Coalescing transform** — a required access pattern would be non-coalesced in global memory. Load coalesced into smem, then read in the desired pattern at no bandwidth penalty (PG §2.2.4.2.1, L1484-L1540, matrix transpose).
3. **Inter-warp communication** — one warp computes, another warp consumes. Shared memory is the natural staging area because warp shuffles do not cross warps.

If none of these three roles apply, the extra `__syncthreads()` and shared-memory footprint are pure overhead (see pitfall P5).

## Why

Shared memory is the only CUDA storage that simultaneously satisfies: **low latency**, **high bandwidth**, **indexable by lane**, **sized in tens of KB per block**, and **survives across `__syncthreads()` within a kernel**. Warp shuffles beat it on latency but are warp-scoped; global memory beats it on capacity but is an order of magnitude slower on SM-local latency. For reduction, stencil, GEMM, normalization, and any axis-switching operation, the tile-in-smem pattern is the default building block.

The programming guide's own matrix-transpose worked example (PG §2.2.4.2.1, L1484-L1540) makes the point explicit: the naive `c[col*m + row] = a[row*m + col]` pattern writes non-coalesced; staging the tile through `__shared__ float smem[32][32]` restores coalesced reads *and* coalesced writes. The skill below is a generalization of that single example to four sub-patterns.

## When to use

### S1. Tile reuse — cache a tile of global memory for in-block reuse

When multiple threads of the block read the same tile of global data multiple times (e.g. a row of matrix A that participates in every column of the output tile):

```cuda
#define TILE 32
__global__ void tiledMultiply(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float*       __restrict__ C,
                              int N) {
    __shared__ float aTile[TILE][TILE];
    __shared__ float bTile[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int k = 0; k < N; k += TILE) {
        aTile[threadIdx.y][threadIdx.x] = A[row * N + (k + threadIdx.x)];
        bTile[threadIdx.y][threadIdx.x] = B[(k + threadIdx.y) * N + col];
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < TILE; i++)
            sum += aTile[threadIdx.y][i] * bTile[i][threadIdx.x];
        __syncthreads();
    }
    C[row * N + col] = sum;
}
```

Each A element is read `TILE` times from smem instead of global. BP §10.2.3.2 (L728-L732) documents a direct reduction from `N` global reads per element to `1` global read per element.

### S2. Coalescing transform — coalesce on load, then reorder in smem

When the natural access pattern would be non-coalesced in global (the canonical case is matrix transpose), load coalesced into smem, sync, then read in the transposed order. Pad the second dimension to `TILE+1` to avoid 32-way bank conflicts on column access.

```cuda
#define TILE 32
__global__ void transpose(const float* __restrict__ in,
                          float*       __restrict__ out,
                          int width, int height) {
    __shared__ float smem[TILE][TILE + 1];        // +1 breaks column-bank conflict
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < width && y < height)
        smem[threadIdx.y][threadIdx.x] = in[y * width + x];   // coalesced read
    __syncthreads();
    x = blockIdx.y * TILE + threadIdx.x;          // swap the block coords
    y = blockIdx.x * TILE + threadIdx.y;
    if (x < height && y < width)
        out[y * height + x] = smem[threadIdx.x][threadIdx.y]; // coalesced write
}
```

Both sides (global read and global write) become coalesced. This is the worked example in PG §2.2.4.2.1 (L1484-L1540).

### S3. Dynamic shared memory — tile size chosen at launch

When the tile size depends on runtime arguments, declare `extern __shared__` and pass the byte count as the kernel's third launch parameter:

```cuda
extern __shared__ float smem[];
__global__ void dyn_kernel(const float* in, float* out, int tile_w) {
    int tid = threadIdx.x;
    if (tid < tile_w) smem[tid] = in[blockIdx.x * tile_w + tid];
    __syncthreads();
    if (tid < tile_w) out[blockIdx.x * tile_w + tid] = smem[tid] * 2.f;
}

// host:
int tile_w = runtime_chosen();
dyn_kernel<<<grid, block, tile_w * sizeof(float)>>>(in, out, tile_w);
```

For >48 KB per block, also call `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)` before launch (PG §3.2.6, L4094-L4130, "must use dynamic shared memory and require an explicit opt-in"). When partitioning one buffer into multiple typed arrays, align each partition manually (see P3).

### S4. L1/Shared carveout — give the kernel the split it actually needs

On Volta+, L1 cache and shared memory share one physical SRAM. A kernel that uses little or no shared memory should let the hardware use the area as L1; a kernel that lives and dies by its smem tile should claim most of it for shared. Control per-kernel with:

```cuda
// Prefer maximum shared memory (typical for GEMM/normalization tiling)
cudaFuncSetAttribute(my_kernel,
    cudaFuncAttributePreferredSharedMemoryCarveout,
    cudaSharedmemCarveoutMaxShared);

// Or integer percentage (PG §3.2.6, L4094-L4130):
cudaFuncSetAttribute(my_kernel,
    cudaFuncAttributePreferredSharedMemoryCarveout, 50);
```

The driver rounds up to the next supported capacity. PG §3.2.6 notes the setter is a **hint** — the driver may override if the kernel cannot otherwise launch.

## When NOT to use

- **No reuse, no coalescing problem, no cross-warp communication.** If each smem element is read exactly once per block and the global access pattern is already coalesced, the smem stage adds a `__syncthreads()` barrier and a register round-trip for zero benefit. Let the hardware L1 handle the one-shot read (BP §10.2.3 L719-L727 — "effects of shared memory" cautions against this exact case, and pitfall P5 describes the symptom).
- **Tile size forces occupancy below the bandwidth knee.** When the per-block smem claim drops occupancy to 1 block/SM, memory-latency hiding collapses. Budget the tile to keep ≥2 blocks/SM on the target device (BP §11.4, L1132-L1150). For the H200-specific breakpoint, see the measured section below.
- **`cp.async` would do the job better.** For a fresh tile used once, `cuda::memcpy_async` + `cuda::pipeline` overlaps the load with compute and avoids blocking the warp on the barrier (see the `async-copy` skill). Shared memory is still the destination, but the path is different.
- **Distributed smem fits the shape better.** Histograms / reductions that need >one-block smem can use sm_90 distributed shared memory via thread-block clusters (`cluster.map_shared_rank`). That is not covered by this skill — see the `thread-block-cluster` entry under `40-hardware-feature/` once bootstrapped.

## Measured Characteristics

Measured on H200-SXM (sm_90a, CUDA 12.9, driver 570.124.06) using `80-experience/hw-probes/smem-tile-reuse/` — S2 coalescing-transform variant on matrix transpose, N = 4096×4096 fp32, default L1/smem carveout, unlocked clock logged at 1980 MHz. Full record: [80-experience/hw-probes/smem-tile-reuse/2026-04-21-smem-tile-reuse.md](../../../80-experience/hw-probes/smem-tile-reuse/2026-04-21-smem-tile-reuse.md).

| Kernel                          | Median ms | Eff. BW GB/s | DRAM SoL | Warp cyc/issue | Bank conflicts (ld) |
| ------------------------------- | --------: | -----------: | -------: | -------------: | ------------------: |
| `naive_transpose`               |    0.2544 |       527.65 |    9.64% |         163.48 |                   0 |
| `smem_tiled` (unpadded 32×32)   |    0.1363 |       984.81 |   14.96% |          85.54 |      **16,326,828** |
| `smem_tiled` (padded 32×33)     | **0.0796**|  **1685.14** | **29.10%**|     **42.09** |              33,449 |

- **S2 alone gives 1.87× over the naive non-coalesced-store baseline** (coalescing transform via smem), even before bank-conflict fixes.
- **`[TILE][TILE+1]` padding contributes another 1.71×** on top of S2 for **3.19× total**; bank conflicts drop **488.4×** (16.3 M → 33.4 K).
- **Warp cycles / issued instruction** is the cleanest NCU signature: 163 → 86 → 42 tracks the two optimizations exactly.
- DRAM SOL rises from 9.6 % (naive, L2-replay-limited) to 29.1 % (padded, memory-balanced). The padded kernel is now DRAM-bound on this 64 MB working set, not smem-pipe-limited.
- Pitfall P8 ("unpadded `[TILE][TILE]` causes 32-way bank conflicts") is upgraded to measured on the strength of this probe.
- Sub-skills **S1 (temporal reuse)**, **S3 (dynamic vs static smem)**, **S4 (carveout sweep)** are **not yet probed**; they remain `inferred` against PG / BP Guide until follow-up probes land.

## Principles

1. **Pay `__syncthreads()` only for reuse, reorder, or cross-warp communication.** Absent one of the three, stay in registers or let L1 do its job.
2. **Padding is how smem layouts stop being bank-pathological.** A `[TILE][TILE+1]` declaration turns a column access from 32-way conflict into conflict-free. Treat this as the default for any 2-D smem tile you intend to read by column.
3. **Smem footprint is an occupancy knob, not just a correctness choice.** Per BP §11.4 (L1132-L1150), tile size decisions must be paired with a check on "how many blocks/SM does this buy me."
4. **Carveout API is a hint, not a contract.** Ship with a verified default (often `MaxShared` for GEMM/LN kernels, default for elementwise) and measure. Do not write a runtime heuristic that flips carveout per launch unless profiling demands it.

## Open questions

- Q1. On H200, what is the smem-carveout break-even point where L1 pressure costs more than the reuse gain from extra shared memory? Planned follow-up probe: `smem-tile-reuse/carveout-sweep/`.
- Q2. Does `[TILE][TILE+1]` padding still strictly dominate on sm_90a for 16-bit tile elements (where 4 bf16 share one 32-bit bank)? Planned follow-up: bf16 variant inside the same probe family.
- Q3. Break-even between static and dynamic smem for tiles under 16 KB — is there a measurable compiler-optimization penalty as old KB pitfall P9 claims (see `pitfalls.md`)?

## Legacy references

- `legacy_sandbox_path`: `KernelPilot/knowledge/optimization/memory/shared-memory-cache/skill.md`. Original source had six sub-skills; this MVP port keeps S1–S4 and routes Skill 5 (large dynamic smem >48 KB) into S3 as the opt-in footnote, and Skill 6 (distributed shared memory) to the `40-hardware-feature/thread-block-cluster/` skill (pending bucket F bootstrap).
- Legacy Level-3 sandbox findings are retained in `pitfalls.md` as P7 through P11; they were run on small inputs and do not constitute measured evidence until re-run on the H200 probe.
