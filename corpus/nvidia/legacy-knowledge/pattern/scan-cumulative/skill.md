# Scan (Cumulative) Pattern -- Skills

## When to Apply
- Prefix sum (inclusive/exclusive): output[i] = sum(input[0..i]) or sum(input[0..i-1])
- Cumulative product, cumulative max, cumulative min
- Any operation where output[i] depends on all previous elements via an associative operator
- Building block for: stream compaction, radix sort, histogram, run-length encoding, segmented operations

## Core Characteristic
Scan has inherent **sequential dependency** along the scan dimension: each output depends on all previous outputs. The challenge is parallelizing this across thousands of threads. Three main algorithms exist:
1. **Blelloch scan** (work-efficient parallel scan)
2. **Kogge-Stone scan** (low-latency but more work)
3. **Decoupled lookback** (inter-block parallelism without a second pass)

## Skill 1: Warp-Level Inclusive Scan with Shuffle-Up
- `__shfl_up_sync` shifts values up by offset, enabling parallel scan within a warp
- For inclusive scan: log2(32) = 5 steps

### Pattern:
```cuda
float val = thread_data;
for (int offset = 1; offset < 32; offset <<= 1) {
    float neighbor = __shfl_up_sync(0xffffffff, val, offset);
    if (lane_id >= offset) val += neighbor;
}
// val now holds inclusive prefix sum for this warp
```

- Total operations: 5 additions per thread (but warp-synchronous, so ~5 steps)
- No shared memory needed
- Works for any associative operator (replace + with max, min, etc.)

## Skill 2: Block-Level Scan (Blelloch / Up-Sweep + Down-Sweep)
- For block-level scan with shared memory:

### Phase 1: Up-sweep (reduce)
- Tree reduction: same as reduction pattern, building partial sums in SMEM
- After up-sweep: SMEM[blockDim-1] holds total sum

### Phase 2: Down-sweep (distribute)
- Walk back down the tree, distributing partial sums
- Each node: pass its value to left child, add left child's old value to right child

### Alternative: Warp-scan + SMEM combine
- Each warp does warp-level scan independently
- Warp totals written to SMEM
- First warp scans the warp totals (meta-scan)
- Each thread adds the appropriate meta-scan result to its local value

### Preferred approach (simpler, less synchronization):
```cuda
// Step 1: warp-level scan
float val = warp_inclusive_scan(thread_data);  // using shfl_up

// Step 2: warp totals to SMEM
__shared__ float warp_totals[32];
if (lane_id == 31) warp_totals[warp_id] = val;
__syncthreads();

// Step 3: scan warp totals (first warp only)
if (warp_id == 0) {
    float wt = (lane_id < num_warps) ? warp_totals[lane_id] : 0.0f;
    wt = warp_inclusive_scan(wt);
    warp_totals[lane_id] = wt;
}
__syncthreads();

// Step 4: add prefix to each warp's results
if (warp_id > 0) val += warp_totals[warp_id - 1];
```

## Skill 3: Grid-Level Scan with Decoupled Lookback
- Challenge: inter-block scan requires blocks to know the prefix sum of all previous blocks
- Naive approach: two-pass (first pass = block-level reduce, second pass = distribute) -- requires 2 kernel launches
- **Decoupled lookback** (Merrill & Garland, 2016): single-pass, blocks cooperate via global memory flags

### Decoupled lookback algorithm:
1. Each block computes its local scan and publishes its aggregate (sum of all elements in block) to a global status array
2. Blocks publish status flags: INVALID -> AGGREGATE_READY -> PREFIX_READY
3. When a block needs the prefix of previous blocks, it walks backwards through the status array:
   - If previous block is PREFIX_READY: use its inclusive prefix, done
   - If AGGREGATE_READY: accumulate its aggregate, continue walking back
   - If INVALID: spin-wait
4. This achieves single-pass grid-level scan with lookback chain typically of length 1-3 (very short)

### Implementation requirements:
- Global memory array for per-block status + aggregate + inclusive prefix
- Atomic operations for flag updates: `atomicAdd` or compare-and-swap
- Memory ordering: use `__threadfence()` between writing aggregate and flag
- Block ordering: blocks must be assigned in order (blockIdx determines scan position)

## Skill 4: Exclusive vs Inclusive Scan Conversion
- Inclusive scan: output[i] = op(input[0], input[1], ..., input[i])
- Exclusive scan: output[i] = op(input[0], input[1], ..., input[i-1]), output[0] = identity
- Convert inclusive to exclusive: `exclusive[i] = inclusive[i-1]` (shift right, prepend identity)
- Convert exclusive to inclusive: `inclusive[i] = exclusive[i] op input[i]`

## Skill 5: Segmented Scan
- Scan that restarts at segment boundaries (e.g., prefix sum per row of a matrix)
- Each thread has a flag indicating whether it starts a new segment
- Modify scan operation: `val = flag ? thread_data : thread_data + neighbor`
- For warp-level: modify `__shfl_up_sync` result based on segment flag
- For block-level: propagate segment boundaries through the meta-scan

## Skill 6: Scan as a Building Block
- **Stream compaction**: prefix sum of predicate flags gives output indices
- **Radix sort**: prefix sum of digit histograms gives scatter addresses
- **Work distribution**: prefix sum of per-element work counts gives thread assignments
- Understanding scan enables implementing these algorithms efficiently

## Skill 7: Choosing the Right Approach by Problem Size
- **Small array (< 1024 elements)**: single block, warp-scan approach
- **Medium array (1K - 1M elements)**: multi-block with decoupled lookback
- **Very large array (> 1M elements)**: decoupled lookback scales well; chain length stays short
- **Batch of small scans**: one warp or one block per scan, parallelize across batch dimension

## Cross-References
- pattern/reduction -- reduction is a special case of scan (only final element needed)
- optimization/compute/warp-primitives -- shuffle instructions for warp-level scan
- optimization/synchronization/atomic-reduction -- atomics for inter-block communication in decoupled lookback
- optimization/synchronization/barrier-optimization -- __syncthreads for block-level scan
