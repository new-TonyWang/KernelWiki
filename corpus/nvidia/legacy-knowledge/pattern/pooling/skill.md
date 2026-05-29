# Pooling Pattern -- Skills

## When to Apply
- Max pooling, average pooling, min pooling over spatial windows
- Global average/max pooling (reduce entire spatial dimensions)
- Sliding-window operations that produce a single value per window
- Any spatial downsampling that combines neighboring elements

## Core Characteristic
Pooling is a **spatial reduction with sliding window**: it applies a reduction operation (max, mean, min) over a local neighborhood. The key challenges are:
1. Efficient window traversal with coalesced memory access
2. Handling overlapping windows without redundant loads
3. Balancing parallelism over output elements vs input reuse

## Skill 1: Thread-to-Output Mapping
- Map each thread to one output element
- Each thread loads all input elements in its pooling window, reduces them
- Grid dimensions: `(ceil(W_out/blockDim.x), ceil(H_out/blockDim.y), N*C)`

### Pattern for 2D max pooling:
```cuda
int out_w = blockIdx.x * blockDim.x + threadIdx.x;
int out_h = blockIdx.y * blockDim.y + threadIdx.y;
int nc = blockIdx.z;

if (out_w < W_out && out_h < H_out) {
    float max_val = -FLT_MAX;
    for (int kh = 0; kh < pool_h; kh++) {
        for (int kw = 0; kw < pool_w; kw++) {
            int in_h = out_h * stride_h + kh;
            int in_w = out_w * stride_w + kw;
            if (in_h < H_in && in_w < W_in) {
                float val = input[nc * H_in * W_in + in_h * W_in + in_w];
                max_val = fmaxf(max_val, val);
            }
        }
    }
    output[nc * H_out * W_out + out_h * W_out + out_w] = max_val;
}
```

## Skill 2: Shared Memory Tiling for Input Reuse
- When pooling windows overlap (stride < pool_size), neighboring output elements share input data
- Load a tile of input into SMEM, then each thread reads its window from SMEM
- Tile dimensions: output_tile + halo (extra border for the pooling window)
  - SMEM tile height = blockDim.y * stride_h + pool_h - stride_h
  - SMEM tile width = blockDim.x * stride_w + pool_w - stride_w
- Reduces redundant global memory loads by a factor of ~pool_size when stride = 1

### When SMEM tiling is worth it:
- Overlapping pools (stride < pool_size): significant data reuse
- Pool_size >= 3: enough reuse to amortize SMEM overhead
- NOT worth it for stride >= pool_size (no overlap, no reuse)

## Skill 3: Vectorized Input Loads
- Use `float4` or `__half2` vector loads for the input tile
- Particularly effective for 1D pooling or when the width dimension is contiguous and aligned
- For the innermost loop (width): load 4 consecutive floats at once
- PTX: `ld.global.v4.f32` loads 128 bits in one transaction

## Skill 4: Warp-Level Pooling for Large Windows
- For very large pooling windows (e.g., global average pooling): treat as a full reduction
- Map one warp per output element; warp cooperatively reduces the input
- Use `__shfl_down_sync` for warp-level reduction after each thread loads multiple elements
- For global average pooling: this is identical to the reduction pattern with division at the end

## Skill 5: Separable Pooling for Large 2D Windows
- Decompose 2D pooling into two 1D passes: horizontal then vertical (or vice versa)
- Reduces work from O(pool_h * pool_w) per element to O(pool_h + pool_w)
- Only valid for average pooling (separable) and max pooling (max is associative)
- Trade: two kernel launches and an intermediate buffer
- Particularly effective for large windows (pool_h or pool_w > 5)

## Skill 6: Average Pooling with Count Correction
- Naive average pooling divides by pool_h * pool_w (the window size)
- At tensor boundaries with padding, the actual number of valid elements may be smaller
- Divide by actual valid count per output element for correct average
- Alternative: use `count_include_pad` flag to decide behavior (PyTorch convention)

## Skill 7: Handling Padding and Boundaries
- Implicit zero-padding: treat out-of-bounds input as zero
- For max pooling: initialize accumulator to -FLT_MAX, boundary elements naturally excluded
- For average pooling: track count of valid (non-padded) elements for correct normalization
- Use `fmaxf(x, y)` (NaN-safe) to handle potential NaN inputs correctly

## Skill 8: Global Pooling as Reduction
- Global average pooling = reduce entire spatial dimensions to a single value per channel
- This is a standard reduction problem (see pattern/reduction)
- Use hierarchical reduction: warp-level -> block-level -> grid-level
- For large spatial dimensions: each block handles a chunk, atomic accumulation or two-pass

## Cross-References
- pattern/reduction -- global pooling decomposes to reduction
- optimization/memory/coalescing -- ensuring coalesced loads for input tiles
- optimization/memory/shared-memory-cache -- SMEM tiling for window reuse
- optimization/memory/vectorized-access -- vector loads for input data
