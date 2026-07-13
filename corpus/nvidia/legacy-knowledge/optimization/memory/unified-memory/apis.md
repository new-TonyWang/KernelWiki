# Unified Memory -- Related APIs


## Core APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `cudaMallocManaged` | Runtime API | Allocates memory accessible from host and device (unified memory) |
| `cudaMemAdvise` | Runtime API | Advise about usage of a memory range (read-mostly, preferred location, accessed-by) |
| `cudaMemDiscardAndPrefetchBatchAsync` | Runtime API | Batch memory discards followed by prefetches |
| `cudaMemLocation` | Runtime API | Specifies memory location (type + id, supports device, host, host NUMA) |
| `cudaMemPrefetchAsync` | Runtime API | Prefetches memory to a specified destination location |
| `cudaMemPrefetchBatchAsync` | Runtime API | Batch prefetch of multiple memory ranges |
| `cudaMemoryAdvise` | Runtime API | Memory advice (SetReadMostly, SetPreferredLocation, SetAccessedBy and Unset variants) |
| `cudaStreamAttachMemAsync` | Runtime API | Attach memory to a stream asynchronously (for unified memory visibility) |

## Related APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `cudaMemDiscardBatchAsync` | Runtime API | Batch memory discards (informs driver contents are no longer useful) |
| `cudaMemRangeGetAttribute` | Runtime API | Query an attribute of a given memory range |
| `cudaMemRangeGetAttributes` | Runtime API | Query attributes of a given memory range |
| `cudaPointerGetAttributes` | Runtime API | Returns attributes about a specified pointer (type, device, hostPointer, devicePointer) |
