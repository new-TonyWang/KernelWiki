# Coalescing -- Related APIs


## Core APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `ld.global[.vec][.type]` | PTX ISA | Global memory load; supports .v2/.v4 vector variants |
| `st.global[.vec][.type]` | PTX ISA | Global memory store; supports .v2/.v4 |
| `cudaHostAlloc` | Runtime API | Allocates page-locked (pinned) host memory with flags (mapped, portable, write-combined) |
| `cudaMallocPitch` | Runtime API | Allocates pitched device memory (ensures coalesced access for 2D arrays) |

## Related APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `ldu.global[.vec][.type]` | PTX ISA | Uniform global load (same address across warp) |
| `cudaMalloc3D` | Runtime API | Allocates logical 3D device memory with pitch alignment |
| `cudaMemcpy2D` | Runtime API | Copies data between host and device (2D, with pitch) |
