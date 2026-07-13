# Convolution -- Related APIs

## cuDNN Core APIs

| API | Description |
|-----|-------------|
| `cudnnCreate` / `cudnnDestroy` | Create/destroy cuDNN handle (cache across calls) |
| `cudnnSetStream` | Bind handle to CUDA stream |
| `cudnnCreateTensorDescriptor` / `cudnnSetTensor4dDescriptor` | Describe input/output tensor layout (NCHW or NHWC) |
| `cudnnCreateFilterDescriptor` / `cudnnSetFilter4dDescriptor` | Describe conv weight (out_c, in_c/groups, kH, kW) |
| `cudnnCreateConvolutionDescriptor` / `cudnnSetConvolution2dDescriptor` | Set padding, stride, dilation, cross-correlation mode |
| `cudnnSetConvolutionMathType` | Enable `CUDNN_TENSOR_OP_MATH` for tensor core acceleration |
| `cudnnSetConvolutionGroupCount` | Set group count (= channels for depthwise) |
| `cudnnGetConvolutionForwardAlgorithm_v7` | Query best algorithm with timing heuristic |
| `cudnnGetConvolutionForwardWorkspaceSize` | Query workspace requirement for selected algorithm |
| `cudnnConvolutionForward` | Execute forward convolution |
| `cudnnConvolutionBackwardData` | Execute transposed convolution (backward-data path) |
| `cudnnAddTensor` | Add bias tensor to output |

## cuDNN Algorithm Constants

| Constant | When to use |
|----------|-------------|
| `CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM` | General purpose, supports tensor cores, dilated conv |
| `CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM` | Depthwise/memory-bound, lower register pressure |
| `CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED` | 3x3 kernel, stride=1, dilation=1 |
| `CUDNN_CONVOLUTION_FWD_ALGO_FFT` | Large kernels (>= 7x7) |
| `CUDNN_CONVOLUTION_FWD_ALGO_GEMM` | Explicit im2col + GEMM, high workspace |
| `CUDNN_TENSOR_OP_MATH` | Enable tensor core (TF32 for FP32 input) |

## CUDA Device APIs (custom kernels)

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `__ldg()` | Runtime API | Read-only cache hint for input/weight loads |
| `__restrict__` | Compiler | Pointer aliasing hint for optimizer |
| `#pragma unroll` | Compiler | Unroll kernel-size loops (3x3, 5x5) |
| `__syncthreads()` | Runtime API | Block barrier for shared memory tiling |
| `float4` / `reinterpret_cast<float4*>` | Runtime API | Vectorized memory access (4 floats per load) |
