# Cache Load Hints -- Related APIs


## Core APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `.ca (cache-all)` | PTX ISA | Cache at all levels (L1+L2), default for ld |
| `.cg (cache-global)` | PTX ISA | Cache in L2 only, bypass L1 |
| `.cs (cache-streaming)` | PTX ISA | Streaming access, evict-first policy |
| `cp.async.cg.shared.global[.vec]` | PTX ISA | Async copy global->shared, L2 cache only |
| `ld.global.nc[.vec][.type]` | PTX ISA | Non-coherent global load via read-only texture cache path |
| `prefetch{.space}` | PTX ISA | Prefetch data into cache (.L1/.L2) |
| `__ldca()` | Runtime API | Cache at all levels |
| `__ldcg()` | Runtime API | Cache at global level, evict first |
| `__ldcs()` | Runtime API | Cache streaming (likely to be accessed once) |
| `__ldg()` | Runtime API | Read-only data cache load (through texture cache path) |
| `__ldlu()` | Runtime API | Load last use (evict after use) |
| `cudaDeviceGetCacheConfig` | Runtime API | Returns the preferred cache configuration (L1 vs shared memory split) |
| `cudaDeviceSetCacheConfig` | Runtime API | Sets the preferred cache configuration (L1 vs shared memory) |
| `cudaFuncCache` | Runtime API | L1 vs shared memory preference (PreferNone, PreferShared, PreferL1, PreferEqual) |
| `cudaFuncSetCacheConfig` | Runtime API | Sets the preferred cache configuration (L1 vs shared memory) for a device function |

## Related APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `.cv (cache-volatile)` | PTX ISA | Invalidate + refetch from memory |
| `.lu (last-use)` | PTX ISA | Hint that data will not be reused |
| `.wb (write-back)` | PTX ISA | Default store caching, write-back all levels |
| `.wt (write-through)` | PTX ISA | Write-through to system memory |
| `ld.const[.vec][.type]` | PTX ISA | Constant memory load |
| `__stcg()` | Runtime API | Store cache at global level |
| `__stcs()` | Runtime API | Store cache streaming |
| `__stwb()` | Runtime API | Store write-back |
| `__stwt()` | Runtime API | Store write-through |
| `cudaCreateChannelDesc` | Runtime API | Returns a channel descriptor using the specified format |
| `cudaCreateTextureObject` | Runtime API | Creates a texture object with specified resource, texture, and resource view descriptors |
| `cudaMemcpy3DWithAttributesAsync` | Runtime API | 3D async copy with memory copy attributes (e.g., src/dst access hints) |
| `cudaMemcpyAttributes` | Runtime API | Memory copy attributes (src/dst access flags) |
| `cudaMemcpyWithAttributesAsync` | Runtime API | Async copy with memory copy attributes (src/dst access policy hints) |
