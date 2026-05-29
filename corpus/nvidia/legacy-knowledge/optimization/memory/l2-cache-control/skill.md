---
status: unverified
verified_date: null
verified_on: null
performance_data:
  before: null
  after: null
source:
  - Best Practices Guide, Section 10.2.2 (L2 Cache)
  - Best Practices Guide, Section 10.2.2.1 (L2 Cache Access Window)
  - Best Practices Guide, Section 10.2.2.2 (Tuning the Access Window Hit-Ratio)
  - Programming Guide, Section 4.13 (L2 Cache Control)
cross_ref:
  - Programming Guide, Section 2.2.3.6 (Caches)
  - Programming Guide, Section 3.2.6 (Configuring L1/Shared Memory Balance)
related_apis:
  - cudaAccessPolicyWindow, cudaStreamSetAttribute, cudaDeviceSetLimit, cudaCtxResetPersistingL2Cache, cudaAccessPropertyPersisting, cudaAccessPropertyStreaming, .L2::cache_hint, .cg, .cs, createpolicy
related_experience: []
unlocks:
  - "memory/data-prefetch: L2 persistence combined with prefetch warms hot data path"
  - "memory/cache-load-hints: L2 control and cache hints work together for fine-grained caching"
conflicts_with:
  - "memory/unified-memory: unified memory page migration may conflict with L2 persistence windows"
---

# L2 Cache Control

## Skill 1: Set Up L2 Persisting Access Window via Stream Attributes

### When to Use
- Compute capability 8.0+ (Ampere and later)
- Kernel accesses a hot data region repeatedly (e.g., lookup tables, weight matrices)
- The hot data region fits within the L2 set-aside capacity

### How to Apply
1. Query `persistingL2CacheMaxSize` from device properties
2. Set aside L2 cache with `cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size)`
3. Configure access policy window on the stream
4. Launch kernel on that stream

### Code Template
```cuda
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
size_t l2_size = min((size_t)(prop.l2CacheSize * 0.75),
                     (size_t)prop.persistingL2CacheMaxSize);
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, l2_size);

cudaStream_t stream;
cudaStreamCreate(&stream);

cudaStreamAttrValue attr;
attr.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(hot_data);
attr.accessPolicyWindow.num_bytes = hot_data_bytes;
attr.accessPolicyWindow.hitRatio = 1.0f;
attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);

for (int i = 0; i < iterations; i++) {
    myKernel<<<grid, block, 0, stream>>>(hot_data, streaming_data);
}
```

### Source
Best Practices Guide, Section 10.2.2.1 (L2 Cache Access Window)

## Skill 2: Tune hitRatio to Avoid L2 Thrashing

### When to Use
- Persistent data region exceeds the L2 set-aside size
- Multiple concurrent kernels compete for L2 set-aside

### How to Apply
1. Set `num_bytes` to the L2 set-aside size (not the full data size)
2. Set `hitRatio` to `set_aside_size / actual_data_size`
3. This causes a random fraction of accesses to be persisting, fitting within set-aside

### Code Template
```cuda
size_t set_aside = 20 * 1024 * 1024;  // 20 MB
size_t actual_data = freq_size * sizeof(int);

cudaStreamAttrValue attr;
attr.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(data);
attr.accessPolicyWindow.num_bytes = set_aside;
attr.accessPolicyWindow.hitRatio = (float)set_aside / (float)actual_data;
attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
```

### Source
Best Practices Guide, Section 10.2.2.2 (Tuning the Access Window Hit-Ratio)

## Skill 3: Reset L2 Persisting Lines After Use

### When to Use
- After a sequence of kernels that used persisting access
- Before launching kernels with different data access patterns

### How to Apply
1. Set the access policy window `num_bytes` to 0 to disable it
2. Call `cudaCtxResetPersistingL2Cache()` to evict all persisting lines
3. Subsequent kernels use the full L2 cache normally

### Code Template
```cuda
// Disable the window
cudaStreamAttrValue attr;
attr.accessPolicyWindow.num_bytes = 0;
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);

// Evict all persisting lines
cudaCtxResetPersistingL2Cache();

// Now launch kernels that benefit from full L2
normalKernel<<<grid, block, 0, stream>>>(data);
```

### Source
Programming Guide, Section 4.13.5 (Reset L2 Access to Normal)

## Skill 4: Configure L2 Persistence for Graph Kernel Nodes

### When to Use
- Using CUDA Graphs and want to set L2 persistence per kernel node
- Enables different kernels in the graph to have different L2 policies

### How to Apply
1. Create CUDA Graph and add kernel nodes
2. Set `cudaKernelNodeAttributeAccessPolicyWindow` on individual nodes
3. Execute the graph

### Code Template
```cuda
cudaKernelNodeAttrValue node_attr;
node_attr.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(hot_data);
node_attr.accessPolicyWindow.num_bytes = hot_data_bytes;
node_attr.accessPolicyWindow.hitRatio = 0.6f;
node_attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
node_attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;

cudaGraphKernelNodeSetAttribute(node, cudaKernelNodeAttributeAccessPolicyWindow,
                                &node_attr);
```

### Source
Programming Guide, Section 4.13.2 (L2 Policy for Persisting Accesses)

## Skill 5: Query L2 Cache Properties for Runtime Configuration

### When to Use
- Writing portable code that adapts L2 policy to different GPUs
- Need to know L2 size, max persist size, and max window size

### How to Apply
1. Query `cudaDeviceProp` fields: `l2CacheSize`, `persistingL2CacheMaxSize`, `accessPolicyMaxWindowSize`
2. Clamp your window parameters to device limits

### Code Template
```cuda
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);

printf("L2 cache size: %d bytes\n", prop.l2CacheSize);
printf("Max persisting L2: %d bytes\n", prop.persistingL2CacheMaxSize);
printf("Max window size: %d bytes\n", prop.accessPolicyMaxWindowSize);

size_t window = min((size_t)prop.accessPolicyMaxWindowSize, my_data_size);
```

### Source
Programming Guide, Section 4.13.7 (Query L2 Cache Properties)

## Cascading Opportunities (unlocks)
After configuring L2 persistence:
1. Check memory/data-prefetch -- combine L2 persistence with prefetch for double benefit
2. Check memory/cache-load-hints -- use .cg or .cs cache operators in kernel code to complement stream-level policy

## Conflicts
- memory/unified-memory: page migration in unified memory may interfere with L2 persistence windows

## Principles
- **P1**: L2 persistence gives hot data priority in the L2 set-aside region. Other data (streaming) can only use this region when it is not occupied by persisting data.
- **P2**: When persistent data exceeds set-aside size, thrashing occurs. Tune hitRatio to control which fraction of data is persisting.
- **P3**: On A100, a 50% bandwidth improvement was observed when persistent data (10-30MB) fit within the L2 set-aside (30MB).
- **P4**: Always reset persisting lines when done. Stale persisting lines reduce effective L2 for subsequent kernels.
- **P5**: MIG mode disables L2 set-aside. MPS only allows setting it at server startup.

## Open Questions (for Level 3 verification)
- Q1: What is the optimal hitRatio for a GEMM kernel with 40MB of weights on A100 (40MB L2)?
- Q2: How do concurrent kernels on different streams interact with overlapping persistence windows?
- Q3: Does L2 persistence provide benefit on Hopper given larger L2 and improved caching?
