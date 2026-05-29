# Normalization Pattern -- Skills

## When to Apply
- LayerNorm: y = (x - mean) / sqrt(var + eps) * gamma + beta (over feature dimension)
- RMSNorm: y = x / sqrt(mean(x^2) + eps) * gamma (no mean subtraction)
- BatchNorm: normalization over batch dimension (at training time)
- GroupNorm: normalization over groups of channels
- Any operation combining reduction (mean, variance) with elementwise scaling

## Core Characteristic
Normalization is a **compound pattern**: reduction + elementwise + broadcast. It combines:
1. Reduction to compute statistics (mean, variance / RMS)
2. Elementwise normalization using those statistics
3. Optional affine transformation (scale + bias)

The key optimization challenge is computing statistics efficiently and fusing the normalization with the preceding/following operations.

## Skill 1: Two-Pass vs One-Pass Statistics

### Two-pass approach:
- Pass 1: compute mean = sum(x) / N
- Pass 2: compute var = sum((x - mean)^2) / N, then normalize
- Numerically stable but requires reading input twice

### One-pass (Welford's algorithm):
- Maintain running mean and M2 (sum of squared differences)
- For each new element: update delta, mean, M2 incrementally
- Variance = M2 / N at the end
- Single pass over data; better for memory-bandwidth-bound kernels

### One-pass (sum + sum_sq):
- Accumulate sum(x) and sum(x^2) simultaneously
- var = sum_sq/N - (sum/N)^2
- Simpler than Welford but numerically less stable for large values
- Sufficient for most ML workloads where values are bounded

## Skill 2: Warp-Level Statistics with Shuffle
- For feature dimensions that fit in a warp (d <= 32): each thread holds one element
  - `__shfl_down_sync` tree reduction for sum and sum_sq in 5 steps
  - Broadcast result via `__shfl_sync` from lane 0
- For larger d: each thread holds multiple elements, reduces locally first, then warp-reduces

### Pattern for warp-level LayerNorm:
```cuda
// Each thread loads and locally reduces its elements
float local_sum = 0.0f, local_sum_sq = 0.0f;
for (int i = lane_id; i < d; i += 32) {
    float val = input[row * d + i];
    local_sum += val;
    local_sum_sq += val * val;
}

// Warp reduction
for (int offset = 16; offset > 0; offset >>= 1) {
    local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    local_sum_sq += __shfl_down_sync(0xffffffff, local_sum_sq, offset);
}

// Broadcast statistics
float mean = __shfl_sync(0xffffffff, local_sum, 0) / d;
float var = __shfl_sync(0xffffffff, local_sum_sq, 0) / d - mean * mean;
float inv_std = rsqrtf(var + eps);

// Normalize
for (int i = lane_id; i < d; i += 32) {
    float val = input[row * d + i];
    output[row * d + i] = (val - mean) * inv_std * gamma[i] + beta[i];
}
```

## Skill 3: Block-Level Normalization for Large Feature Dimensions
- When d > warp_size, use multiple warps per row
- Each warp reduces its chunk of the feature dimension
- Warp leaders write to SMEM; after `__syncthreads()`, first warp reduces the partial sums
- Broadcast final statistics back to all threads via SMEM

## Skill 4: One Warp Per Row vs One Block Per Row
- **One warp per row** (d <= ~1024): minimal synchronization, high parallelism over rows
  - Multiple warps per block, each handling a different row
  - Best for typical transformer hidden dims (768, 1024, 2048)
- **One block per row** (d > 1024): all warps in block cooperate on one row
  - Needed when d is very large (e.g., vocabulary-size normalization)
  - Use SMEM for inter-warp communication

## Skill 5: Fused Normalization Kernels
- **GEMM + LayerNorm fusion**: Apply normalization in GEMM epilogue
  - Requires row-wise reduction in the epilogue -- possible with CUTLASS EVT topological visitors
  - Saves one full pass over the output matrix
- **Normalization + Activation fusion**: Combine norm and activation into one kernel
  - Straightforward: after computing normalized value, apply activation before storing
- **RMSNorm**: Simpler than LayerNorm (no mean subtraction) -- saves one reduction

## Skill 6: Using rsqrt for Inverse Standard Deviation
- Never compute `1.0f / sqrtf(var + eps)` -- two expensive operations
- Use `rsqrtf(var + eps)` which is a single hardware instruction
- For half-precision: `hrsqrt(__half)` or `h2rsqrt(__half2)` for packed operation
- Fast variant: `__frsqrt_rn(x)` with round-to-nearest

## Skill 7: Mixed-Precision Normalization
- Common pattern in transformers: input FP16/BF16, statistics computed in FP32
- Accumulate sum and sum_sq in FP32 to avoid overflow and precision loss
- Compute mean, variance, inv_std in FP32
- Apply normalization: can convert back to FP16/BF16 for output if needed
- Weight (gamma) and bias (beta) often kept in FP32 for stability

## Cross-References
- pattern/reduction -- warp/block reduction techniques
- optimization/compute/fast-math -- rsqrt and other fast math functions
- optimization/compute/half-precision-math -- packed half operations
- optimization/compute/warp-primitives -- shuffle-based reduction
- pattern/elementwise -- the elementwise normalization/scaling step
