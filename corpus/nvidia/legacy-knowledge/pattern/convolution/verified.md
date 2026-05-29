# Convolution — Verification Data

## Preliminary Evidence from KernelBench Experiments (2026-04-07)

Level 3 sandbox verification for convolution skills has not yet been run. The following evidence comes from the knowledge base validation experiment (KernelBench Level 1, H200 SXM).

### Skill 1 (cuDNN Algorithm Selection): supported

**Problem**: 80_conv_2D_dilated_padded
**Baseline**: 16387.6us (naive cuDNN without tensor core math)
**Optimized**: 2310.2us (IMPLICIT_PRECOMP_GEMM + CUDNN_TENSOR_OP_MATH)
**Improvement**: 85.9%
**Key change**: Adding `cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH)` enabled tensor core acceleration

### Skill 2 (cuDNN State Caching): supported

**Problem**: 87_conv_pointwise_2D
**Baseline**: 45525.4us (handle created/destroyed per call)
**Optimized**: 3426.1us (persistent ConvState struct with cached descriptors)
**Improvement**: 92.5%
**Key change**: Caching cudnnHandle, descriptors, and workspace across invocations

### Skill 3 (Depthwise Custom Kernel): supported

**Problem**: 83_conv_depthwise_2D_asymmetric_kernel
**Baseline**: 6472.1us (cuDNN)
**Optimized**: 659.3us (custom CUDA kernel with __ldg + grid-stride + unroll)
**Improvement**: 89.8%
**Key change**: Custom kernel with per-element thread mapping beat cuDNN for asymmetric (Kx1) kernel shape

### Skill 6 (Depthwise via cuDNN Groups): supported

**Problem**: 82_conv_depthwise_2D_square_kernel
**Approach**: cuDNN with `cudnnSetConvolutionGroupCount(conv_desc, channels)` + IMPLICIT_GEMM
**Result**: Competitive with torch baseline; algorithm comment notes "lower register pressure, enabling higher occupancy, ideal for memory-bound depthwise convolutions"

### Skill 7 (Winograd): supported

**Problem**: 54_conv_standard_3D
**Approach**: `CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED` with fallback chain
**Result**: Code comment: "Winograd is significantly faster for 3x3 convolutions"

## Pending

- Full Level 3 sandbox verification for all 8 skills (run skill_verifier.py with custom experiment designs)
- Quantitative verification of Skills 4 (depthwise-separable fusion), 5 (NHWC layout), 8 (transposed conv)
