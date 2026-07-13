# Normalization Pattern

| Skill | Focus | Key Optimization |
|-------|-------|-----------------|
| Two-Pass vs One-Pass Statistics | Choose Welford or sum+sum_sq for mean/variance | One-pass reduces memory reads; Welford for numerical stability |
| Warp-Level Statistics with Shuffle | Reduce within warp using __shfl_down_sync | Zero shared memory overhead for d <= 32 |
| Block-Level Reduction for Large Feature Dims | Two-stage warp+SMEM reduction for statistics | Handles feature dimensions > 1024 |
| Vectorized Load/Store for Normalize Pass | float4 loads for input, gamma, beta in normalize pass | Maximize memory bandwidth utilization |
| Fused LayerNorm + Residual + Bias | Single kernel for residual-add + bias + LayerNorm | Eliminate intermediate global memory writes |
| RMSNorm Specialization | Skip mean computation; only compute RMS | Fewer reductions and no mean subtraction step |
| Online/Streaming Normalization | Welford update for streaming mean/variance | Single pass with incremental statistics |
