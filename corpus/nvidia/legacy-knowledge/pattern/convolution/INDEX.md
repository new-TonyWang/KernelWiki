# Convolution Pattern

| Skill | Focus | Key Optimization |
|-------|-------|-----------------|
| cuDNN Algorithm Selection | Choose IMPLICIT_PRECOMP_GEMM / WINOGRAD / IMPLICIT_GEMM by shape | Match algorithm to conv type for 2-4x speedup |
| cuDNN State Caching | Persist handles, descriptors, workspace across calls | Avoid per-call creation overhead (10-100x) |
| Depthwise Custom Kernel | One thread per output element, grid-stride loop | `__ldg` + `#pragma unroll` for memory-bound depthwise |
| Depthwise-Separable Fusion | Fuse depthwise + pointwise in one kernel | Shared memory weight caching eliminates intermediate tensor |
| NCHW vs NHWC Layout | Choose data layout for coalescing and tensor core | NHWC for tensor core path; NCHW for depthwise spatial access |
| Depthwise via cuDNN Groups | Use `cudnnSetConvolutionGroupCount(groups=channels)` | IMPLICIT_GEMM algorithm for memory-bound depthwise |
| Winograd Transform | 2.25x fewer multiplications for 3x3 stride=1 | cuDNN WINOGRAD_NONFUSED with fallback chain |
| Transposed Convolution | Use cuDNN backward-data API for upsampling | Avoid explicit zero-insertion + forward conv |
