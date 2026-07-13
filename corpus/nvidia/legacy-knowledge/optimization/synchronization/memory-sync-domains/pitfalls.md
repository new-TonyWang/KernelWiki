# Memory Sync Domains -- Pitfalls

## P1: Cross-Domain Communication Without System-Scope Fence

**Symptom:** Data written by a kernel in one domain is not visible to a kernel in another domain, despite device-scope fences.

**Detection:** Correctness failures when compute and communication kernels in different domains share data.

**Fix:** Use system-scope fences (`__threadfence_system()` or `fence.sc.sys`) for any data that crosses domain boundaries.

```cpp
// BAD: device-scope fence does not cross domain boundaries
__threadfence();  // only orders within same domain

// GOOD: system-scope fence crosses domain boundaries
__threadfence_system();
```

> Source: PG 4.14.2 -- "ordering or synchronization between distinct domains on the same GPU requires system-scope fencing."

---

## P2: Assuming Domains Work on Pre-Hopper Hardware

**Symptom:** Domain attributes are set but have no effect on devices with CC < 9.0.

**Detection:** `cudaDevAttrMemSyncDomainCount` returns 1. No performance improvement from domain settings.

**Fix:** Domain isolation is a Hopper+ feature. On older hardware, the attribute is silently ignored (1 physical domain). This is safe -- backward compatible -- but provides no benefit.

> Source: PG 4.14.3 -- "Devices of compute capability 9.0 (Hopper) have 4 domains... CUDA will report a count of 1 on devices prior to compute capability 9.0."

---

## P3: Setting Domain on the Wrong Level (Stream vs. Launch vs. Graph)

**Symptom:** Domain configuration does not take effect because it was set at the wrong API level.

**Detection:** Profiler shows all kernels in the same domain despite configuration attempts.

**Fix:**
- Logical domain (default vs. remote) is typically set per-launch or per-kernel.
- Domain mapping (logical to physical) is typically set per-stream.
- For graphs, both are set per-node at capture time; stream-level settings are NOT used during graph execution.

```cpp
// Stream-level: set the mapping
cudaStreamSetAttribute(stream, cudaLaunchAttributeMemSyncDomainMap, &mapAttr);

// Launch-level: set the logical domain
cudaLaunchConfig_t config = {};
config.attrs = &domainAttr;  // cudaLaunchAttrMemSyncDomain
config.numAttrs = 1;
```

> Source: PG 4.14.3 -- "Graphs take both attributes from the node itself... Domain-related attributes set on the stream a graph is launched into are not used in execution of the graph."

---

## P4: Overusing Domains When Scoped Fences Are Sufficient

**Symptom:** Code complexity increases without performance benefit because the workload does not have concurrent compute and communication kernels.

**Detection:** Only one type of kernel runs at a time; no fence interference to isolate.

**Fix:** Domains are beneficial specifically when concurrent kernels with different memory traffic patterns cause fence interference. If kernels run sequentially, domains provide no benefit.

> Source: PG 4.14.1 -- fence interference occurs when "a kernel is performing computation in local GPU memory, and a parallel kernel is performing communications with a peer."

## P5: This skill is inherently untestable in a single-stream or single-kernel profiling harness; it requires a multi-stream workload with concurrent fence-heavy kernels to observe any effect (discovered in verification)

**Symptom**: This skill is inherently untestable in a single-stream or single-kernel profiling harness; it requires a multi-stream workload with concurrent fence-heavy kernels to observe any effect.
**Source**: Level 3 sandbox verification (2026-04-06)

## P6: Introducing TMA + fence (discovered in verification)

**Symptom**: Introducing TMA + fence.proxy.async for workloads that fit well in direct global loads can regress performance — the async copy machinery (mbarrier setup, shared memory staging, proxy fence) adds instruction overhead that only pays off at larger data sizes or when shared memory reuse is high.
**Source**: Level 3 sandbox verification (2026-04-06)

## P7: Domain isolation benefits are only observable when NCCL collectives and compute kernels run concurrently on separate streams; a microbenchmark without multi-GPU NCCL communication will always show zero difference (discovered in verification)

**Symptom**: Domain isolation benefits are only observable when NCCL collectives and compute kernels run concurrently on separate streams; a microbenchmark without multi-GPU NCCL communication will always show zero difference.
**Source**: Level 3 sandbox verification (2026-04-06)
