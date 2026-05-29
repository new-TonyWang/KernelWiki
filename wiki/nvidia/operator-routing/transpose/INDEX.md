---
title: Transpose Pattern -- Decision Tree
pattern_class: cuda-core
op: transpose
covers:
- matrix_transpose_2d
- axis_permute_nd
- aos_soa_conversion
status: draft
hardware:
  device: H200
  sm: 9.0a
source:
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L1413-L1483
  excerpt: 'Matrix Transpose Example Using Global Memory: a naive implementation of
    matrix transpose is functionally correct but not optimized because the write of
    the c matrix is not coalesced.'
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L1484-L1540
  excerpt: 'Matrix Transpose Example Using Shared Memory: shared memory will be treated
    as a user-managed cache to stage loads and stores from global memory, resulting
    in coalesced global memory access of both reads and writes.'
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L606-L634
  excerpt: 'Strided Accesses: stride-2 gives 50% load/store efficiency; as the stride
    increases, bandwidth decreases. Non-unit-stride accesses should be avoided whenever
    possible.'
id: routing-transpose-INDEX
type: operator-routing
vendor: nvidia
operator: transpose
---
# Transpose Pattern -- Decision Tree

This document guides the kernel-writing agent through a transpose task from initial problem statement to a working, optimized kernel. The transpose family covers 2-D matrix transpose, general N-D axis permutation, and AoS ↔ SoA layout conversion.

**Key characteristic**: a transpose is *not* a compute operation — it is a pure layout rearrangement. The central optimization challenge is that the naive implementation forces one side (load or store) to be non-coalesced. Staging through shared memory lets both sides be coalesced at the cost of one `__syncthreads()` and a smem tile.

---

## Step 0 -- Try the library first

```
Q0. Is the caller's environment PyTorch-based?
    YES --> Can one of the following handle the shape + dtype?
            - tensor.transpose(dim0, dim1)      # swap two dims
            - tensor.permute(dims)               # general N-D permute
            - tensor.T                            # 2-D convenience
            - tensor.contiguous()                 # force materialization
            YES --> Use PyTorch op. DONE.
            NO  --> Continue to Q1.
    NO  --> Continue to Q1.

Q1. Is this a 2-D matrix transpose with dim0↔dim1 swap, standard dtype
    (fp32/fp16/bf16), and the matrix fits in device memory?
    YES --> Is the data going to feed cuBLAS / cuBLASLt / CUTLASS next?
            YES --> Use cublasLtMatrixTransform. DONE.
            NO  --> Proceed to Q2.
    NO  --> Continue to Q3.

Q2. For library-GEMM downstream: does cublasLtMatrix{Transform,Layout}
    accept the needed (order, stride, dtype) tuple?
    YES --> Use cublasLtMatrixTransform. See library-fallback.md. DONE.
    NO  --> Continue to Q3.

Q3. Is this an AoS → SoA split (struct-of-fields → array-per-field)?
    YES --> This is a layout transform, not a transpose. Follow
            wiki/nvidia/foundations/memory/layout-transform/skill.md S1; the kernel
            is a one-pass conversion, not a smem-tiled transpose.
    NO  --> Continue to Q4.

Q4. Does library performance fall >10% below a bandwidth-bound
    theoretical limit for your shape, OR does the transpose need to
    fuse with a neighboring op to avoid an extra gmem round-trip?
    YES --> Proceed to Step 1 (custom kernel).
    NO  --> Stay with the library path. Most transposes are
            well-served by torch.permute + cublasLtMatrixTransform.
```

**When to skip the library**: custom is justified when (a) the transpose fuses with surrounding compute (e.g. transpose + elementwise in one kernel), (b) the shape regime is too small for the library to amortize its own launch overhead, or (c) an unusual dtype / layout (packed int8 / packed fp8 / strided views) is not supported by the library.

---

## Step 1 -- Choose the custom kernel strategy

### 2-D matrix transpose (the canonical case)

Thread `(y, x)` reads `in[y*N + x]` and writes `out[x*N + y]`. The read is coalesced; the write is stride-N and therefore non-coalesced. Without smem, this is bounded by ~1/32 of peak on a warp's store side.

```
Q5a. Is M == N (square) and a multiple of 32?
     YES --> Use the standard [TILE][TILE+1] padded smem tile (see
             wiki/nvidia/foundations/memory/shared-memory-cache/ skill S2 and its
             probe sources/experience/hw-probes/smem-tile-reuse/ — 3.19x
             faster than naive, 1685 GB/s on H200 at 4096x4096 fp32).
     NO   --> Same kernel works; add boundary predicates on the load
             AND the store (sibling skill pitfall P2). Expect slightly
             reduced peak BW due to partial edge tiles.
```

```cuda
#define TILE 32
__global__ void transpose_smem(const float* __restrict__ in,
                               float*       __restrict__ out,
                               int width, int height) {
    __shared__ float tile[TILE][TILE + 1];    // +1 breaks bank conflicts
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < width && y < height)
        tile[threadIdx.y][threadIdx.x] = in[y * width + x];   // coalesced read
    __syncthreads();
    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;
    if (x < height && y < width)
        out[y * height + x] = tile[threadIdx.x][threadIdx.y]; // coalesced write
}
```

Measured on H200 (fp32, 4096×4096): unpadded `[32][32]` gives 985 GB/s; padded `[32][33]` gives 1685 GB/s (3.19× over naive). See `wiki/nvidia/foundations/memory/shared-memory-cache/` and its probe record.

### High-dim permute (N-D)

```
Q5b. Does the permutation swap the innermost axis (last dim) with a
     non-innermost axis?
     YES --> This is a "real" transpose: one axis becomes non-
             contiguous. Collapse to a 2-D transpose where the
             two affected dims form the tile, and the remaining dims
             form a batch over which you loop with gridDim.z / a
             block-per-batch strategy. Each 2-D slice uses the
             transpose_smem kernel above.
     NO   --> The permutation only swaps among non-innermost axes.
             Innermost dim stays stride-1 in both layouts. This is
             just an index remap; a plain elementwise kernel with a
             rewritten output-index formula works with full coalescing.
             Prefer that over smem staging.
```

### AoS → SoA (and SoA → AoS)

This is technically a layout transform, not a transpose, but agents routinely confuse the two.

```
Q5c. Is the input a struct-of-fields array and the kernel wants
     stride-1 access to one field?
     YES --> Follow the layout-transform skill S1. Kernel is a one-
             pass conversion that reads AoS coalesced (as a bulk
             struct read) and writes N separate SoA arrays coalesced.
             Measured on H200 (24-B Particle struct, N=16M): convert
             cost 0.2017 ms; amortizes after 3.65 downstream calls.
```

### Strided view (in-place logical transpose)

```
Q5d. Can you get away with NOT materializing the transpose, by
     changing downstream kernels' index formula to access strided?
     YES --> Skip the transpose kernel entirely. The downstream
             kernel pays the stride penalty (sibling skill
             layout-transform: ~2x on H200 for stride-24 vs stride-1),
             but you save the 0.2017 ms transpose cost.
             Amortize: if only 1 downstream kernel reads the
             transposed view, prefer the strided-view path; if >= 4,
             materialize.
     NO   --> Materialize via transpose_smem (Q5a) or layout-transform
             S1 (Q5c).
```

---

## Step 2 -- Optimization via ROUTING.md skills

After the basic custom kernel is working and correct, apply optimization skills from `ROUTING.md` in priority order:

1. **Shared memory cache** (`wiki/nvidia/foundations/memory/shared-memory-cache/`) — the central mechanism of the smem-tiled transpose. Applies to Q5a.
2. **Bank-conflict avoidance** (`wiki/nvidia/foundations/memory/bank-conflict/`) — the `[TILE][TILE+1]` padding rule. Measured on H200: 488× bank- conflict reduction, 1.71× speedup.
3. **Layout transform** (`wiki/nvidia/foundations/memory/layout-transform/`) — skill S1 covers AoS↔SoA (Q5c); S2 covers pitched allocation when rows are not naturally aligned.
4. **Vectorized access** (`wiki/nvidia/foundations/memory/vectorized-access/`) — when the tile element is float or fp16, wider loads (float4 / bfloat162) reduce instruction count. Applies most to Q5a's large-shape regime.
5. **Coalescing** (`wiki/nvidia/foundations/memory/coalescing/`) — sanity-check that both the smem-load and smem-store sides of the transpose are stride-1 on global memory.

After each skill application, re-benchmark against the baseline (`torch.permute` / `cublasLtMatrixTransform`) and follow the bottleneck-triage procedure in `reasoning/bottleneck-triage.md`.

---

## Cross-references

- **Library fallback details**: `library-fallback.md`
- **Skill whitelist for this pattern**: `ROUTING.md`
- **Task packet template**: `TASK-PACKET.md`
- **Central mechanism**: `wiki/nvidia/foundations/memory/shared-memory-cache/`
- **Layout decision (AoS/SoA/pitched)**: `wiki/nvidia/foundations/memory/layout-transform/`
- **Bottleneck triage after benchmarking**: `reasoning/bottleneck-triage.md`
