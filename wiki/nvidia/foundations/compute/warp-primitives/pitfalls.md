---
title: Warp Primitives - Pitfalls
status: draft
evidence_level: spec
applies_to_pattern_class:
- cuda-core
applies_to_ops:
- reduction
- scan
- broadcast
requires_sm: '>=3.0'
single_kernel_useful: true
source:
- path: spec
  anchor: Reference
id: pitfall-warp-primitives
type: pitfall
vendor: nvidia
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L24203-L24290
- source_id: cuda-official/toolkit-docs-13.2
  path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L24015-L24031
- source_id: cuda-official/toolkit-docs-13.2
  path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L23988-L23990
---
## P1: Mask mismatch with active threads

**Symptom**: Kernel hangs indefinitely or produces undefined results.

**Cause**: The `mask` parameter does not match the set of threads that actually reach the intrinsic call site. Common trigger: using `0xFFFFFFFF` inside a branch where only a subset of threads is active.

**Example (from programming guide L24021)**:
```cuda
// WRONG: lane 0 does not call shfl but its bit is set in mask
int result = (laneId > 0) ? __shfl_sync(0xFFFFFFFF, value, 0) : 0;
```

**Fix**: Either ensure all threads reach the call, or use `__activemask()` to compute the correct mask (with care -- see P2). The best practice is to restructure the code so that all 32 lanes participate:
```cuda
int result = __shfl_sync(0xFFFFFFFF, value, 0);
if (laneId == 0) result = 0;
```

**Detection**: Compute-sanitizer (`--tool memcheck`) can sometimes flag incorrect mask usage on newer toolkits.

## P2: Using __activemask() as a participation mask

**Symptom**: Subtle correctness bugs where the shuffle reads stale or undefined values from threads that happen to be inactive due to warp divergence.

**Cause**: `__activemask()` returns the set of threads that are currently executing, which can be a subset of the intended participants. Using it as the mask for `__shfl_sync` may exclude threads that should participate, or include threads at different program points on architectures with independent thread scheduling (sm_70+).

**Fix**: Use explicit, compile-time-computable masks (e.g., `0xFFFFFFFF` for full-warp operations) and ensure all threads in the mask converge at the call site.

## P3: Non-power-of-two width parameter

**Symptom**: Undefined behavior; typically produces garbage values.

**Cause**: Passing a `width` that is not in `{1, 2, 4, 8, 16, 32}`.

**Example (from programming guide L24029)**:
```cuda
// WRONG: width=31 is not a power of two
__shfl_sync(0xFFFFFFFF, value, 0, /*width=*/31);
```

**Fix**: Always use a power-of-two width. If the logical subgroup size is not a power of two, pad to the next power and mask out unused lanes.

## P4: Reading from an inactive or non-participating lane

**Symptom**: Undefined (garbage) return value.

**Cause**: `__shfl_down_sync` or `__shfl_sync` with a `srcLane` that points to a thread not in the mask or not active.

**Example (from programming guide L24023-L24026)**:
```cuda
if (laneId <= 4) {
    // WRONG: lanes 3,4 try to read from lanes 5,6 which are not active
    result = __shfl_down_sync(0b11111, value, 2);
}
```

**Fix**: Ensure destination lanes referenced by the shuffle mode are all active and in the mask. For `__shfl_down_sync`, note that the upper `delta` lanes within each `width` segment will get their own value (not an error), but the source lanes must be active.

## P5: Assuming shuffle implies memory ordering

**Symptom**: Data race or stale-value bug when using shuffle results to coordinate memory accesses.

**Cause**: The programming guide states (L24034): "These intrinsics do not imply a memory barrier. They do not guarantee any memory ordering." Warp vote functions have the same warning (L23889).

**Fix**: If the result of a shuffle or vote is used to decide which thread writes to shared or global memory, insert an explicit `__syncwarp()` or `__threadfence_block()` as needed to ensure prior stores are visible.

## P6: Butterfly reduction vs. tree reduction confusion

**Symptom**: Incorrect reduction result when mixing up `__shfl_xor_sync` (butterfly) and `__shfl_down_sync` (tree) patterns.

**Cause**: The butterfly pattern (`__shfl_xor_sync`) distributes the final result to **all** lanes. The tree pattern (`__shfl_down_sync`) accumulates the result only in **lane 0** (the upper lanes hold partial, incorrect values). Using the wrong pattern and then reading from the wrong lane produces incorrect results.

**Fix**: If all lanes need the final result, use the butterfly pattern (`__shfl_xor_sync` with delta = 16, 8, 4, 2, 1). If only lane 0 needs the result, either pattern works, but `__shfl_down_sync` avoids redundant data movement in the later stages. After a `__shfl_down_sync` tree reduction, broadcast lane 0's result with `__shfl_sync(mask, val, 0)` if all lanes need it.

## P7: Overflow in chained reductions

**Symptom**: `inf` or `nan` values after many chained warp reductions.

**Cause**: Each butterfly reduction sums 32 lanes; chaining N reductions multiplies the magnitude by up to 32^N. Even with small initial values, this overflows fp32 quickly (observed in the probe: 128 chained reductions produced `inf`).

**Fix**: This is typically only a concern in microbench probes, not in production kernels where each reduction starts from fresh data. In production, ensure the input range is bounded so that a single 32-way sum does not overflow.

## P8: Forgetting to handle out-of-bounds threads in block reduction

**Symptom**: Incorrect sum when the number of elements is not a multiple of the block size.

**Cause**: Threads beyond the valid data range participate in the warp shuffle with uninitialized or stale register values.

**Fix**: Initialize `val = 0.0f` for out-of-bounds threads before the warp reduction. The identity element (0 for sum, -inf for max, +inf for min) must match the reduction operation.
