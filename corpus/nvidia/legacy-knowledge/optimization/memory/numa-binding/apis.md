# Numa Binding -- Related APIs


## Core APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `cudaDeviceEnablePeerAccess` | Runtime API | Enables direct access to peer device memory allocations |
| `cudaMemAdvise` | Runtime API | Advise about usage of a memory range (read-mostly, preferred location, accessed-by) |
| `cudaMemLocation` | Runtime API | Specifies memory location (type + id, supports device, host, host NUMA) |
| `cudaMemPrefetchAsync` | Runtime API | Prefetches memory to a specified destination location |

## Related APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `cudaDeviceCanAccessPeer` | Runtime API | Queries if a device may directly access a peer device's memory |
| `cudaDeviceGetP2PAttribute` | Runtime API | Queries attributes of the link between two devices |
| `cudaMemPoolSetAccess` | Runtime API | Controls the pool's accessibility on the specified device |
