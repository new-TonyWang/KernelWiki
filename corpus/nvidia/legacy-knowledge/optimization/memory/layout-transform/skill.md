---
status: unverified
verified_date: null
verified_on: null
performance_data:
  before: null
  after: null
source:
  - Best Practices Guide, Section 10.2.1.4 (Strided Accesses)
  - Programming Guide, Section 2.2.4.1.1 (Matrix Transpose Example Using Global Memory)
  - Programming Guide, Section 2.2.4.2.1 (Matrix Transpose Using Shared Memory)
cross_ref:
  - PTX ISA, cp.async.bulk.tensor (layout transformation via tensor descriptor)
  - PTX ISA, prmt.b32
related_apis:
  - cp.async.bulk.tensor, prmt.b32, cublasLtMatrixTransform, cudaMallocPitch
related_experience: []
unlocks:
  - "memory/coalescing: layout transform enables coalesced access for all subsequent kernels"
  - "memory/shared-memory-cache: shared-memory-based transpose is the primary layout transform technique"
conflicts_with:
  - "memory/host-device-transfer: layout transform adds a kernel launch or transfer overhead"
---

# Layout Transform

## Skill 1: Shared-Memory-Based Matrix Transpose for Coalesced Access

### When to Use
- Need to transpose a matrix to enable coalesced access in subsequent kernels
- Out-of-place transpose: source and destination are separate buffers

### How to Apply
1. Each thread block handles a TILE x TILE tile
2. Coalesced read from source into shared memory tile
3. Synchronize
4. Coalesced write from shared memory to destination (transposed indices)
5. Pad shared memory by +1 to avoid bank conflicts on column writes

### Code Template
```cuda
#define TILE 32
__global__ void transpose(const float* __restrict__ in,
                          float* __restrict__ out, int M, int N) {
    __shared__ float tile[TILE][TILE + 1];  // +1 padding for bank conflicts

    int xIdx = blockIdx.x * TILE + threadIdx.x;
    int yIdx = blockIdx.y * TILE + threadIdx.y;

    if (xIdx < N && yIdx < M)
        tile[threadIdx.y][threadIdx.x] = in[yIdx * N + xIdx];
    __syncthreads();

    xIdx = blockIdx.y * TILE + threadIdx.x;
    yIdx = blockIdx.x * TILE + threadIdx.y;
    if (xIdx < M && yIdx < N)
        out[yIdx * M + xIdx] = tile[threadIdx.x][threadIdx.y];
}
```

### Source
Programming Guide, Section 2.2.4.2.1 (Matrix Transpose Using Shared Memory)

## Skill 2: AoS-to-SoA Conversion Kernel

### When to Use
- Input data is in Array-of-Structures format
- Kernel accesses individual fields, causing strided global memory access
- Want to convert to Structure-of-Arrays for coalesced access

### How to Apply
1. Launch a conversion kernel that reads AoS data coalescedly
2. Stage through shared memory with appropriate reordering
3. Write fields to separate SoA arrays coalescedly

### Code Template
```cuda
struct AoS {
    float x, y, z, w;
};

__global__ void aos_to_soa(const AoS* __restrict__ in,
                           float* __restrict__ out_x,
                           float* __restrict__ out_y,
                           float* __restrict__ out_z,
                           float* __restrict__ out_w,
                           int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Each thread reads one AoS element (16 bytes, naturally coalesced)
        AoS val = in[idx];
        out_x[idx] = val.x;
        out_y[idx] = val.y;
        out_z[idx] = val.z;
        out_w[idx] = val.w;
    }
}
```

### Source
Best Practices Guide, Section 10.2.1.4 (Strided Accesses)

## Skill 3: In-Place Row-to-Column Reorder Using Shared Memory

### When to Use
- Need to transform data layout within a single buffer (in-place)
- Working with small tiles that fit in shared memory

### How to Apply
1. Load a tile from global memory into shared memory (coalesced, row order)
2. Synchronize
3. Write back to global memory from shared memory in column order
4. Handle edge tiles with bounds checking

### Code Template
```cuda
#define TILE_X 32
#define TILE_Y 8
__global__ void reorder_tile(float* data, int width, int height) {
    __shared__ float smem[TILE_Y][TILE_X + 1];

    int x = blockIdx.x * TILE_X + threadIdx.x;
    int y = blockIdx.y * TILE_Y + threadIdx.y;

    if (x < width && y < height)
        smem[threadIdx.y][threadIdx.x] = data[y * width + x];
    __syncthreads();

    // Write back in transformed order
    if (x < width && y < height)
        data[y * width + x] = smem[threadIdx.y][threadIdx.x];
}
```

### Source
Best Practices Guide, Section 10.2.1.4 (Strided Accesses)

## Skill 4: Use cudaMallocPitch for Automatic Row Padding

### When to Use
- 2D array where rows are not naturally aligned
- Want the runtime to pad rows for optimal coalescing

### How to Apply
1. Call `cudaMallocPitch(&ptr, &pitch, width_bytes, height)`
2. Use returned pitch for row addressing: `row_ptr = (char*)ptr + row * pitch`
3. When copying to/from host, use `cudaMemcpy2D` with proper pitch

### Code Template
```cuda
float* d_data;
size_t pitch;
cudaMallocPitch(&d_data, &pitch, cols * sizeof(float), rows);

// Copy from host (assumes host data is tightly packed)
cudaMemcpy2D(d_data, pitch, h_data, cols * sizeof(float),
             cols * sizeof(float), rows, cudaMemcpyHostToDevice);
```

### Source
Programming Guide, Section 2.2.4.1 (Coalesced Global Memory Access)

## Skill 5: Byte Permutation with prmt.b32 for Sub-Word Layout Changes

### When to Use
- Need to rearrange bytes within 32-bit words (e.g., RGBA to BGRA)
- Sub-word data packing/unpacking for mixed-precision formats

### How to Apply
1. Use PTX inline assembly for `prmt.b32`
2. Specify a permutation selector to choose which bytes go where
3. Operates on two 32-bit source registers

### Code Template
```cuda
__device__ unsigned int byte_swap(unsigned int a, unsigned int b) {
    unsigned int result;
    // Selector 0x00010203 reverses the bytes of register a
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(result) : "r"(a), "r"(b), "r"(0x00010203));
    return result;
}
```

### Source
PTX ISA, prmt.b32

## Cascading Opportunities (unlocks)
After layout transform:
1. Check memory/coalescing -- the transformed layout should now enable perfect coalescing
2. Check memory/shared-memory-cache -- verify no bank conflicts in the transpose tile
3. Check memory/vectorized-access -- transformed SoA layout enables float4 loads per field

## Conflicts
- memory/host-device-transfer: layout transform adds a kernel or memcpy overhead; amortize over many kernel invocations

## Principles
- **P1**: Strided access with stride S wastes (S-1)/S of memory bandwidth. Layout transforms convert strided access to unit-stride.
- **P2**: The shared-memory transpose technique handles the read-write asymmetry: read coalesced, write coalesced, use shared memory as the pivot.
- **P3**: The cost of a transpose kernel must be amortized over multiple subsequent kernels that benefit from the improved layout.

## Open Questions (for Level 3 verification)
- Q1: For TMA (cp.async.bulk.tensor), can the tensor descriptor handle layout transformation implicitly during copy?
- Q2: What is the bandwidth of a shared-memory transpose kernel on H100 compared to peak device bandwidth?
- Q3: Is cublasLtMatrixTransform faster than a custom CUDA transpose kernel for standard matrix sizes?
