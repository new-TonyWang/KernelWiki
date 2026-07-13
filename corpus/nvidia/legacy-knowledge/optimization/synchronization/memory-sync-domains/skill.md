# Memory Sync Domains -- Skills

```yaml
status: draft
source:
  - "Programming Guide 4.14 (Memory Synchronization Domains)"
cross_ref:
  - optimization/synchronization/thread-scopes
  - optimization/synchronization/barrier-optimization
  - optimization/latency/stream-concurrency
related_apis:
  - cudaLaunchMemSyncDomainDefault
  - cudaLaunchMemSyncDomainRemote
  - cudaLaunchAttributeMemSyncDomain
  - cudaLaunchAttributeMemSyncDomainMap
  - "fence.proxy.async"
  - "fence.sem.scope (PTX)"
unlocks:
  - Elimination of fence interference between compute and communication kernels
  - Faster device-scope fences by isolating traffic
conflicts_with:
  - None
```

---

## S1: Isolate Communication Kernels into Remote Domain

**When to Use:** When compute kernels and communication kernels (e.g., NCCL) run concurrently, and the compute kernel's fences are stalled by the communication kernel's slow NVLink/PCIe writes.

**How to Apply:**
1. Launch communication kernels with `cudaLaunchMemSyncDomainRemote`.
2. Compute kernels use the default domain (domain 0).
3. Each domain's fences only wait for their own domain's writes.

**Code Template:**
```cpp
// Launch communication kernel in remote domain
cudaLaunchAttribute domainAttr;
domainAttr.id = cudaLaunchAttrMemSyncDomain;
domainAttr.val.memSyncDomain = cudaLaunchMemSyncDomainRemote;

cudaLaunchConfig_t config = {0};
config.gridDim = grid;
config.blockDim = block;
config.stream = commStream;
config.attrs = &domainAttr;
config.numAttrs = 1;

cudaLaunchKernelEx(&config, ncclKernel, args...);

// Compute kernel uses default domain (no special attribute needed)
computeKernel<<<grid, block, 0, computeStream>>>(args...);
```

> Source: PG 4.14.2 -- "In exchange for explicit assistance from code, the GPU can reduce the net cast by a fence operation."

---

## S2: Map Physical Domains per Stream for Application-Level Isolation

**When to Use:** When different streams in the same application should have their fences isolated from each other (e.g., independent workloads using NVSHMEM).

**How to Apply:**
1. Map both logical domains (default and remote) to the same physical domain per stream.
2. Assign different physical domains to different streams.
3. Hopper supports 4 physical domains.

**Code Template:**
```cpp
// Stream A: both logical domains map to physical domain 0
cudaLaunchAttributeValue mapA;
mapA.memSyncDomainMap.default_ = 0;
mapA.memSyncDomainMap.remote = 0;
cudaStreamSetAttribute(streamA, cudaLaunchAttributeMemSyncDomainMap, &mapA);

// Stream B: both logical domains map to physical domain 1
cudaLaunchAttributeValue mapB;
mapB.memSyncDomainMap.default_ = 1;
mapB.memSyncDomainMap.remote = 1;
cudaStreamSetAttribute(streamB, cudaLaunchAttributeMemSyncDomainMap, &mapB);
```

> Source: PG 4.14.3 -- "An alternative use pattern... could be to partition parallel streams."

---

## S3: Use fence.proxy.async for Async-to-Generic Memory Ordering

**When to Use:** When mixing async proxy operations (TMA, wgmma) with normal loads/stores, and you need to ensure ordering between the two memory access proxies.

**How to Apply:**
1. After async operations complete (tracked via barrier), issue `fence.proxy.async` before reading the results with normal loads.
2. This bridges the async proxy and the generic (normal) proxy.

**Code Template:**
```cpp
// After TMA copy completes (tracked by mbarrier)
bar.wait(std::move(token));

// Proxy fence: make async writes visible to generic loads
asm volatile("fence.proxy.async.shared::cta;");

// Now safe to read shared memory written by TMA
float val = smem[threadIdx.x];
```

> Source: PTX ISA -- fence.proxy.async bridges async proxy and generic proxy memory accesses.

---

## S4: Leverage Default NCCL Domain Isolation

**When to Use:** When using NCCL 2.16+ with CUDA 12.0+ on Hopper (CC 9.0+). NCCL automatically uses the remote domain, providing fence isolation with no code changes needed in the compute path.

**How to Apply:**
1. Simply use NCCL 2.16+ with Hopper GPUs.
2. The default domain map (default->0, remote->1) is already set.
3. No application-level changes required.

**Code Template:**
```cpp
// No changes needed -- NCCL 2.16+ automatically tags launches with remote domain
// Compute kernels in the default domain benefit from reduced fence interference
computeKernel<<<grid, block, 0, computeStream>>>(args...);
ncclAllReduce(..., commStream);  // automatically in remote domain
```

> Source: PG 4.14.3 -- "NCCL 2.16 will [tag launches with the remote domain]. Together, this provides a beneficial use pattern for common applications out of the box, with no code changes needed."

---

## Cascading Opportunities

- Memory sync domains complement `thread-scopes` by reducing the cost of device-scope fences.
- Combine with `stream-concurrency` for isolated concurrent compute and communication streams.
- Graph nodes inherit domain attributes from capture, so domains work with `cuda-graphs`.

## Conflicts

- None. Domains are additive to existing thread scope semantics.

## Principles

1. **Cross-domain requires system scope:** Communication between different domains on the same GPU needs system-scope fences (PG 4.14.2).
2. **Backward compatible:** Kernels default to domain 0; behavior is unchanged for non-domain-aware code (PG 4.14.2).
3. **Hopper has 4 physical domains:** Use `cudaDevAttrMemSyncDomainCount` to query. Pre-Hopper reports 1 (PG 4.14.3).

## Open Questions

- Q1: What is the measured fence latency reduction from domain isolation in a compute+NCCL workload?
- Q2: How many physical domains does Blackwell (CC 10.0) support?
