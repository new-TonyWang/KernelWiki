---
status: unverified
verified_date: null
verified_on: null
performance_data:
  before: null
  after: null
source:
  - "Programming Guide, Section 5.4.6 (Warp Functions)"
  - "Programming Guide, Section 5.4.6.1 (Warp Active Mask)"
  - "Programming Guide, Section 5.4.6.2 (Warp Vote Functions)"
  - "Programming Guide, Section 5.4.6.3 (Warp Match Functions)"
  - "Programming Guide, Section 5.4.6.4 (Warp Reduce Functions)"
  - "Programming Guide, Section 5.4.6.5 (Warp Shuffle Functions)"
  - "Programming Guide, Section 5.4.6.6 (Warp __sync Intrinsic Constraints)"
cross_ref:
  - "Best Practices Guide, Section 13.1 (Branching and Divergence)"
  - "Best Practices Guide, Section 13.2 (Branch Predication)"
  - "PTX ISA, shfl.sync / vote.sync / redux.sync / match.sync instructions"
related_apis:
  - __shfl_sync
  - __shfl_down_sync
  - __shfl_up_sync
  - __shfl_xor_sync
  - __all_sync
  - __any_sync
  - __ballot_sync
  - __reduce_add_sync
  - __reduce_min_sync
  - __reduce_max_sync
  - __match_any_sync
  - __match_all_sync
  - __syncwarp
  - __activemask
  - shfl.sync.bfly.b32
  - vote.sync.ballot.b32
  - redux.sync.add
  - elect.sync
related_experience: []
unlocks:
  - "warp-divergence: vote functions detect divergence at runtime"
  - "atomic-reduction: warp-level reduce can replace shared memory reduction"
  - "barrier-optimization: warp-level sync avoids costlier block-level barriers"
conflicts_with:
  - "warp-divergence: warp primitives require careful mask management under divergent control flow"
---

# Warp Primitives

## Skill 1: Warp-Level Butterfly Reduction with __shfl_xor_sync
### When to Use
When performing a full-warp reduction (sum, min, max) without shared memory. The butterfly pattern using XOR shuffle completes a 32-lane reduction in 5 steps.

### How to Apply
1. Start with each thread holding its value.
2. XOR shuffle with increasing lane masks: 1, 2, 4, 8, 16.
3. Accumulate the shuffled value at each step.
4. After 5 steps, all threads hold the reduction result (broadcast).

### Code Template
```cuda
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;  // All threads have the sum
}

__global__ void reduce_kernel(const float* __restrict__ input,
                               float* __restrict__ output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx < n) ? input[idx] : 0.0f;

    float warp_sum = warp_reduce_sum(val);

    // Lane 0 writes result
    if (threadIdx.x % 32 == 0)
        atomicAdd(output, warp_sum);
}
```

### Source
Programming Guide, Section 5.4.6.5 (Warp Shuffle Functions) -- Example 3: Reduction across a warp

## Skill 2: Warp-Level Inclusive Scan with __shfl_up_sync
### When to Use
When computing a prefix sum (scan) within a warp or sub-warp partition, e.g., for stream compaction, histogram, or decoupled lookback.

### How to Apply
1. Use `__shfl_up_sync` with increasing delta: 1, 2, 4 (for width 8) or 1, 2, 4, 8, 16 (for full warp).
2. Each thread adds the value from `delta` lanes below if the source lane is valid.
3. The `width` parameter partitions the warp into independent sub-groups.

### Code Template
```cuda
__device__ int warp_inclusive_scan(int val) {
    // Full-warp inclusive scan in 5 steps
    for (int delta = 1; delta <= 16; delta *= 2) {
        int tmp = __shfl_up_sync(0xFFFFFFFF, val, delta);
        if (threadIdx.x % 32 >= delta)
            val += tmp;
    }
    return val;
}
```

### Source
Programming Guide, Section 5.4.6.5 (Warp Shuffle Functions) -- Example 2: Inclusive plus-scan

## Skill 3: Warp Vote for Branch Decision
### When to Use
When you need to check a predicate across all threads in a warp (e.g., early exit if all elements are zero, or compaction if any element satisfies a condition).

### How to Apply
1. `__all_sync(mask, predicate)`: returns non-zero if ALL threads' predicates are true.
2. `__any_sync(mask, predicate)`: returns non-zero if ANY thread's predicate is true.
3. `__ballot_sync(mask, predicate)`: returns a 32-bit bitmask where each bit indicates whether that lane's predicate is true.

### Code Template
```cuda
__global__ void early_exit_kernel(const float* data, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx < n) ? data[idx] : 0.0f;

    // Skip computation if all values in the warp are zero
    if (__all_sync(0xFFFFFFFF, val == 0.0f)) {
        if (idx < n) output[idx] = 0.0f;
        return;
    }

    // Count non-zero elements in the warp
    unsigned mask = __ballot_sync(0xFFFFFFFF, val != 0.0f);
    int nonzero_count = __popc(mask);

    if (idx < n)
        output[idx] = val / (float)nonzero_count;
}
```

### Source
Programming Guide, Section 5.4.6.2 (Warp Vote Functions)

## Skill 4: Hardware Warp Reduce (sm_80+)
### When to Use
When performing a warp-level reduction on sm_80+ devices. The hardware `redux.sync` instruction completes the reduction in a single instruction, avoiding the 5-step shuffle tree.

### How to Apply
1. Use `__reduce_add_sync(mask, value)` for sum.
2. Use `__reduce_min_sync(mask, value)` / `__reduce_max_sync(mask, value)` for min/max.
3. Use `__reduce_and_sync` / `__reduce_or_sync` / `__reduce_xor_sync` for bitwise reductions.
4. Note: currently supports unsigned/signed integer types only, not floating point.

### Code Template
```cuda
// Requires sm_80+
__device__ int fast_warp_reduce_add(int val) {
    return __reduce_add_sync(0xFFFFFFFF, val);  // Single hardware instruction
}

__device__ int fast_warp_reduce_max(int val) {
    return __reduce_max_sync(0xFFFFFFFF, val);
}
```

### Source
Programming Guide, Section 5.4.6.4 (Warp Reduce Functions)

## Skill 5: Warp Broadcast with __shfl_sync
### When to Use
When a single value (e.g., computed by lane 0) must be distributed to all threads in the warp without shared memory.

### How to Apply
1. Use `__shfl_sync(0xFFFFFFFF, value, srcLane)` where `srcLane` is the lane holding the value.
2. All threads receive the value from the source lane.

### Code Template
```cuda
__device__ float warp_broadcast(float val, int src_lane) {
    return __shfl_sync(0xFFFFFFFF, val, src_lane);
}

__global__ void broadcast_example(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx < n) ? data[idx] : 0.0f;

    // Lane 0 computes something, broadcast to all
    float result;
    if (threadIdx.x % 32 == 0)
        result = expf(val);
    result = __shfl_sync(0xFFFFFFFF, result, 0);  // All lanes get lane 0's result
}
```

### Source
Programming Guide, Section 5.4.6.5 (Warp Shuffle Functions) -- Example 1: Broadcast

## Skill 6: Warp Match for Value Grouping
### When to Use
When you need to find threads in a warp that share the same value, e.g., for warp-level histogram or collision detection.

### How to Apply
1. `__match_any_sync(mask, value)`: returns a bitmask of threads with the same value.
2. `__match_all_sync(mask, value, &pred)`: returns mask if all threads have the same value, else 0.
3. Use `__popc()` on the returned mask to count matching threads.

### Code Template
```cuda
__device__ void warp_histogram_increment(int* histogram, int bin) {
    // Find all threads in the warp with the same bin
    unsigned peers = __match_any_sync(0xFFFFFFFF, bin);
    int count = __popc(peers);

    // Elect one thread per unique bin to do the atomic
    int leader = __ffs(peers) - 1;  // First set bit = lowest lane
    if (threadIdx.x % 32 == leader) {
        atomicAdd(&histogram[bin], count);
    }
}
```

### Source
Programming Guide, Section 5.4.6.3 (Warp Match Functions)

## Cascading Opportunities (unlocks)
- **warp-divergence**: `__ballot_sync` and `__any_sync` enable runtime detection of divergence patterns.
- **atomic-reduction**: warp-level reduce + single atomic per warp drastically reduces atomic contention.
- **barrier-optimization**: warp shuffle eliminates the need for `__syncthreads()` in warp-level communication.

## Conflicts
- **warp-divergence**: under divergent control flow, mask management becomes critical. Non-participating threads must NOT be included in the mask.

## Principles
- Always pass the correct mask to `__sync` intrinsics. Each calling thread must have its bit set in the mask.
- All non-exited threads named in the mask must eventually reach the same `__sync` call with the same mask.
- Best efficiency is achieved when all 32 threads participate (`mask = 0xFFFFFFFF`).
- Warp shuffle does NOT provide memory ordering. If shuffled values were loaded from memory, a prior `__syncwarp()` or fence may be needed.
- Prefer CUB `WarpReduce` / `WarpScan` for production code; they handle edge cases and are portable.
- The `width` parameter of shuffle functions must be a power of two in [1, 32].

## Open Questions
- Can `redux.sync` be extended to floating-point types in future architectures?
- What is the overhead of `__match_any_sync` compared to shared memory histogram on current hardware?
