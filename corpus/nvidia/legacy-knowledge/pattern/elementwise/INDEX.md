# Elementwise Pattern

| Skill | Focus | Key Optimization |
|-------|-------|-----------------|
| Vectorized Memory Access | Process multiple elements per thread via float4 | 128-bit coalesced transactions reduce load count |
| Kernel Fusion | Combine multiple elementwise ops into one kernel | Eliminate intermediate global memory round-trips |
| Grid-Stride Loop | Handle arbitrary N with fixed grid size | Persistent thread pattern; amortize launch overhead |
| Half-Precision Packed Ops | Use __half2 for 2x throughput on fp16 data | Single SIMD instruction processes two fp16 values |
| Activation Function Selection | Choose intrinsic vs standard math per accuracy need | __expf, __tanhf for speed; expf, tanhf for precision |
| Tail Handling | Process remainder elements when N not divisible by vector width | Scalar fallback for last 1-3 elements |
| Multi-Input Fusion | Fuse binary/ternary ops (add+mul+activation) | Single pass over all inputs and output |
