# Green Contexts -- Skills

```yaml
status: draft
source:
  - "Programming Guide 4.6 (Green Contexts)"
  - "Best Practices Guide 11.6 (Multiple Contexts -- inferred)"
  - "Colfax Blog: Sharing NVIDIA GPUs -- Time-Sliced and MIG-Backed vGPUs"
cross_ref:
  - optimization/latency/context-management
  - optimization/latency/stream-concurrency
  - optimization/latency/occupancy-tuning
related_apis:
  - cudaGreenCtxCreate
  - cudaDevSmResourceSplit
  - cudaDevSmResourceSplitByCount
  - cudaDevResourceGenerateDesc
  - cudaDeviceGetDevResource
  - cudaExecutionCtxStreamCreate
unlocks:
  - Guaranteed SM availability for latency-sensitive kernels
  - Reduced interference between concurrent workloads
conflicts_with:
  - None (complements MPS and MIG rather than conflicting)
```

---

## S1: Partition SMs for Latency-Sensitive and Background Work

**When to Use:** When a latency-critical kernel (e.g., real-time inference) is being starved by a concurrent compute-heavy kernel.

**How to Apply:**
1. Query available SM resources with `cudaDeviceGetDevResource`.
2. Split SMs into groups using `cudaDevSmResourceSplit`.
3. Create green contexts from each group.
4. Launch latency-sensitive work on the green context with reserved SMs.

**Code Template:**
```cpp
cudaDevResource smResources;
cudaDeviceGetDevResource(0, &smResources, cudaDevResourceTypeSm);

// Split: 80% for background, 20% for latency-sensitive
cudaDevSmResource result[2] = {};
cudaDevSmResourceGroupParams groups[2] = {
    {.smCount = 80, .coscheduledSmCount = 0},  // background
    {.smCount = 20, .coscheduledSmCount = 0}    // latency-sensitive
};
unsigned int remainder = 0;
cudaDevSmResourceSplit(&smResources.smResource, result, groups, 2, 0, &remainder);

// Generate resource descriptors
cudaDevResource desc[2];
for (int i = 0; i < 2; i++) {
    cudaDevSmResource smRes = result[i];
    cudaDevResource res = {.smResource = smRes};
    cudaDevResourceGenerateDesc(&desc[i], &res, 1);
}

// Create green contexts
cudaGreenCtx_t bgCtx, critCtx;
cudaGreenCtxCreate(&bgCtx, desc[0], 0);
cudaGreenCtxCreate(&critCtx, desc[1], 0);

// Create streams on each green context
cudaExecCtx_t bgExecCtx, critExecCtx;
// ... obtain execution contexts and create streams ...
```

> Source: PG 4.6.1 -- "Green contexts provide a way towards that by partitioning SM resources, so a given green context can only use specific SMs."

---

## S2: Use Simple SM Count Split for Quick Prototyping

**When to Use:** When you want a quick way to test the effect of limiting SM count without kernel modifications.

**How to Apply:**
1. Use `cudaDevSmResourceSplitByCount` for a simpler API that splits by count with default granularity.
2. Create a green context with the split resources.
3. Launch kernels on streams bound to this context.

**Code Template:**
```cpp
cudaDevResource smResources;
cudaDeviceGetDevResource(0, &smResources, cudaDevResourceTypeSm);

cudaDevSmResource limitedSm;
cudaDevSmResourceSplitByCount(&limitedSm, &smResources.smResource, 16);  // use 16 SMs

cudaDevResource desc;
cudaDevResource res = {.smResource = limitedSm};
cudaDevResourceGenerateDesc(&desc, &res, 1);

cudaGreenCtx_t ctx;
cudaGreenCtxCreate(&ctx, desc, 0);
```

> Source: PG 4.6.2 -- ease of use for green context SM partitioning.

---

## S3: Configure Work Queues to Prevent Serialization

**When to Use:** When independent kernels on different streams are being serialized because they map to the same hardware work queue.

**How to Apply:**
1. Provision work queue resources during green context creation.
2. Express the expected concurrency level so the driver can allocate separate work queues.

**Code Template:**
```cpp
// Configure work queue to allow concurrent streams
cudaDevWorkqueueConfigResource wqConfig;
wqConfig.concurrencyLimit = 2;  // expect 2 concurrent stream workloads
wqConfig.sharingScope = cudaDevWorkqueueSharingScopeContext;
// Include work queue configuration in resource descriptor
```

> Source: PG 4.6.1 -- "green contexts allow the user to express the maximum concurrency they would expect in terms of expected number of concurrent stream-ordered workloads."

---

## S4: Combine Green Contexts with MIG for Multi-Tenant Isolation

**When to Use:** In cloud/multi-tenant environments where MIG provides GPU instance isolation, and green contexts add intra-instance SM partitioning.

**How to Apply:**
1. Configure MIG instances at the system level.
2. Within each MIG instance, use green contexts for finer-grained SM partitioning.
3. The SM resources available for green context partitioning are scoped to the MIG instance.

> Source: PG 4.6.1 -- "one can use green contexts alongside MIG. In that case, the SM resources available for partitioning would be the resources of the given MIG instance."

---

## Cascading Opportunities

- Green contexts pair with `stream-concurrency` to ensure concurrent streams actually get separate SM and work queue resources.
- Combine with `occupancy-tuning` to optimize kernels for the reduced SM count in their green context.
- Use with `stream-concurrency` stream priorities for multi-level QoS.

## Conflicts

- Green contexts do not conflict with other techniques but do reduce the SM count available to each workload.

## Principles

1. **No kernel changes required:** Green contexts are host-side only; kernels run unmodified (PG 4.6).
2. **Static partitioning, not guaranteed concurrency:** Green contexts remove interference factors but do not guarantee concurrent execution (PG 4.6.1).
3. **Lightweight compared to MPS/MIG:** Green context creation is much cheaper than an MPS context (PG 4.6.1).

## Open Questions

- Q1: What is the minimum SM count per green context that is practical for different kernel types?
- Q2: How does co-scheduling alignment affect SM allocation on Blackwell?
