---
id: hw-foundation-memory-hierarchy
title: Memory Hierarchy
type: hardware
vendor: nvidia
architectures:
- sm90
- sm90a
tags:
- cuda-cpp
confidence: source-reported
related: []
sources: []
aliases: []
blackwell_relevance: Foundation hardware concepts apply to both Hopper and Blackwell
---
# CUDA Memory Hierarchy Reference Card

Quick-reference for kernel optimization decisions. Focus: NVIDIA H200 SXM (Hopper, compute capability sm_90a). The H200 shares the same GH100 compute die as the H100, with upgraded HBM3e memory.

---

## 1. Memory Levels Overview

| Level | Location | Capacity (H200 SXM) | Bandwidth | Latency | Scope |
|-------|----------|---------------------|-----------|---------|-------|
| Registers | On-chip | 64K x 32-bit per SM (256 KB) | Highest (zero extra cycles typical) | ~1 cycle | Per thread |
| Shared Memory | On-chip | Up to 228 KB per SM (configurable) | ~19 TB/s aggregate across GPU | ~20-30 cycles | Per thread block |
| L1 Cache | On-chip (unified with shared mem) | 256 KB total per SM (shared mem + L1) | Same as shared memory | ~30-40 cycles | Per SM |
| L2 Cache | On-chip | 50 MB | ~12 TB/s | ~200 cycles | Global (all SMs) |
| HBM (Global Memory) | Off-chip | 141 GB HBM3e | 4.8 TB/s | ~400-600 cycles | Global (all SMs + host) |
| Constant Memory | Off-chip (cached) | 64 KB | Cached: ~constant cache BW; Uncached: HBM BW | Cache hit: ~few cycles; Miss: HBM latency | Global (read-only) |

**Key insight:** Moving data from HBM to registers crosses ~3 orders of magnitude in latency. Maximizing data reuse in registers and shared memory is the most impactful optimization.

---

## 2. Cache Line Sizes

| Cache | Line / Sector Size | Notes |
|-------|-------------------|-------|
| L1 | 128-byte cache line | Coalescing unit for global memory through L1 |
| L2 | 32-byte sector | Global memory transactions are 32-byte aligned sectors |

- Global memory is accessed via 32-byte, 64-byte, or 128-byte transactions, naturally aligned.
- On compute capability 6.0+, the data access unit is 32 bytes regardless of whether loads are cached in L1.
- A warp's global memory access coalesces into the minimum number of 32-byte transactions needed to service all threads.

---

## 3. Shared Memory Bank Structure

- **32 banks**, each 32 bits (4 bytes) wide.
- **Bank mapping:** Successive 32-bit words map to successive banks (bank index = (address / 4) % 32).
- **Bandwidth per bank:** 32 bits per clock cycle.
- **Max aggregate bandwidth:** 32 banks serviced simultaneously per warp request.

### Bank Conflict Rules

| Access Pattern | Bank Conflicts | Throughput Impact |
|---------------|---------------|-------------------|
| 32 threads access 32 distinct banks | None (ideal) | Full bandwidth |
| N threads access same bank (different words) | N-way conflict | Serialized, 1/N bandwidth |
| Multiple threads access same word in same bank | Broadcast (no conflict) | Full bandwidth (read); One write wins (write) |
| Stride of 1 word (4B) | No conflict | Ideal |
| Stride of 2 words (8B) | 2-way conflict | 1/2 bandwidth |
| Stride of 32 words (128B) | 32-way conflict (worst case) | 1/32 bandwidth |

**Optimization tip:** Avoid strides that are multiples of 32 words. Padding shared memory arrays by 1 element can break bank conflicts (e.g., `__shared__ float tile[32][33]` instead of `[32][32]`).

---

## 4. L1 / Shared Memory Configuration Modes

The L1 data cache and shared memory share a unified on-chip memory block. The split is configurable per kernel.

### H100 (Compute Capability 9.0)

- **Total unified capacity:** 256 KB per SM
- **Supported shared memory carveouts:** 0, 8, 16, 32, 64, 100, 132, 164, 196, 228 KB
- **Max shared memory per block:** 227 KB (1 KB reserved for system)
- **Static shared memory limit:** 48 KB (above requires dynamic shared memory + explicit opt-in)

### Comparison Across Architectures

| Architecture | Compute Cap. | Unified Size | Max Shared Mem/SM | Max Shared Mem/Block |
|-------------|-------------|-------------|-------------------|---------------------|
| Volta (V100) | 7.0 | 128 KB | 96 KB | 96 KB |
| Turing | 7.5 | 96 KB | 64 KB | 64 KB |
| Ampere (A100) | 8.0 | 192 KB | 164 KB | 163 KB |
| Hopper (H100) | 9.0 | 256 KB | 228 KB | 227 KB |

### Configuration API

```cpp
// Set preferred shared memory carveout (hint, not hard requirement)
cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);

// Enable large dynamic shared memory (required for > 48 KB per block)
cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, maxBytes);
```

---

## 5. Memory Coalescing Rules

Global memory coalescing determines how efficiently a warp's memory requests are serviced.

### Fundamental Rules (Compute Capability 6.0+)

1. **Transaction size:** 32-byte aligned segments.
2. **Coalescing:** A warp's accesses are grouped into the minimum number of 32-byte transactions covering all requested addresses.
3. **Alignment:** Memory allocated via `cudaMalloc()` is guaranteed aligned to at least 256 bytes.
4. **Word sizes:** Instructions support reading/writing 1, 2, 4, 8, or 16 bytes. Data must be naturally aligned for single-instruction access.

### Access Pattern Performance Impact

| Pattern | Transactions per Warp (32 threads, 4B each) | Efficiency |
|---------|---------------------------------------------|------------|
| Consecutive addresses (stride 1), aligned | 4 x 32B | 100% |
| Consecutive addresses, misaligned by N bytes | 5 x 32B | ~80% |
| Stride 2 (every other element) | 8 x 32B | 50% |
| Stride 4 | 16 x 32B | 25% |
| Stride 32 (one element per cache line) | 32 x 32B | 3.125% (worst case) |
| Random / scattered | Up to 32 x 32B | Very low |

### Best Practices for Coalescing

- **Consecutive thread IDs should access consecutive memory addresses** (stride-1 pattern).
- **Thread block width and array width should be multiples of warp size (32)** for 2D array access.
- **Use `cudaMallocPitch()` / `cudaMalloc3D()`** for 2D/3D arrays to ensure proper alignment and padding.
- **Use `__align__(8)` or `__align__(16)`** for struct types in global memory.
- **Avoid array-of-structures (AoS); prefer structure-of-arrays (SoA)** for coalesced access.
- **For strided access patterns, stage data through shared memory** to convert strided global reads into coalesced accesses.

---

## 6. Global Memory Access Patterns and Performance

### Caching Behavior

| Memory Type | L1 Cached | L2 Cached | Notes |
|-------------|-----------|-----------|-------|
| Global | Yes (CC 6.0+, default) | Yes | Read-only data may use `__ldg()` for unified L1/texture cache |
| Local | Yes | Yes | Used for register spills; coalesced across threads |
| Constant | Constant cache | n/a | Broadcast to all threads if same address; serialized if different addresses |
| Texture | Texture cache (unified with L1) | Yes | Optimized for 2D spatial locality |

### L2 Cache Persistence Control (CC 8.0+)

```cpp
// Set aside L2 cache for persisting accesses
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size);

// Mark access window as persisting via stream attributes
stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
```

- H200 L2 cache: 50 MB total (same as H100; vs. A100's 40 MB).
- Persisting accesses have priority for the set-aside L2 portion.
- Tune `hitRatio` to avoid cache thrashing when persistent data exceeds set-aside size.

### Local Memory

- Resides in device memory (HBM) -- same latency/bandwidth as global memory.
- Used for: register spills, large arrays with non-constant indices, large structures.
- Coalesced automatically: consecutive 32-bit words accessed by consecutive thread IDs.
- Cached in L1 and L2 (CC 6.0+).

---

## 7. H100 Special Memory Features

### Tensor Memory Accelerator (TMA)

- Hardware unit for asynchronous transfer of 1D-5D tensors between global and shared memory.
- Single-threaded programming model: one thread issues the TMA copy, all threads wait via `cuda::barrier`.
- Avoids register usage for address generation; frees threads for computation.
- Supports both global-to-shared and shared-to-global directions, plus reduction operations (add/min/max/and/or).

### Distributed Shared Memory (DSMEM)

- Available with Thread Block Clusters (CC 9.0).
- All thread blocks in a cluster can directly load/store/atomic to each other's shared memory.
- Total DSMEM = (blocks per cluster) x (shared memory per block).
- ~7x faster than routing through global memory for inter-block data exchange.
- Access should be coalesced and aligned to 32-byte segments for best performance.

### Inline Compression (ILC)

- Automatic hardware compression for global memory transfers.
- Can increase effective bandwidth beyond raw HBM bandwidth.
- Does not reduce memory footprint (allocation size unchanged).
- Enabled per allocation via CUDA driver API (`CU_MEM_ALLOCATION_COMP_GENERIC`).

---

## 8. Quick Decision Guide

| Scenario | Recommended Memory | Why |
|----------|-------------------|-----|
| Thread-private accumulators, loop variables | Registers | Zero-latency access |
| Tile of data reused across threads in a block | Shared memory | ~100x faster than HBM |
| Read-only data with spatial locality | Texture / `__ldg()` | Texture cache optimized for 2D |
| Frequently reused global data | L2 persistence hints | Keep hot data in L2 |
| Data shared across blocks (within cluster) | DSMEM | Avoid global memory round-trip |
| Large tensor loads into shared memory | TMA | Frees threads, hardware-managed |
| Broadcast same value to all threads | Constant memory | Single read, broadcast to warp |

---

*Sources: CUDA C++ Programming Guide 13.2, CUDA C++ Best Practices Guide 13.2, Hopper Tuning Guide 13.2, NVIDIA H100 Tensor Core Hopper Architecture Whitepaper.*
