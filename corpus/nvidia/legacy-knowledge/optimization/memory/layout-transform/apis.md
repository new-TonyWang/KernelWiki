# Layout Transform -- Related APIs


## Core APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `cp.async.bulk.tensor[.dim]` | PTX ISA | TMA tensor async copy with layout transformation via tensor descriptor |
| `cublasLtMatrixTransform()` | cuBLAS | Matrix layout transformation (transpose, type conversion) |
| `cudaMallocPitch` | Runtime API | Allocates pitched device memory (ensures coalesced access for 2D arrays) |

## Related APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `prmt.b32[.mode]` | PTX ISA | Byte permute across two 32-bit registers |
| `tensormap.replace` | PTX ISA | Dynamically modify tensor map parameters |
| `cudaMalloc3D` | Runtime API | Allocates logical 3D device memory with pitch alignment |
