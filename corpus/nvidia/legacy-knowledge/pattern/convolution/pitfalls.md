# Convolution Pattern -- Pitfalls

## P1: cuDNN Handle Created Per Call
- **Symptom**: cuDNN convolution 10-100x slower than expected
- **Root cause**: `cudnnCreate()` + `cudnnDestroy()` called every forward pass; handle creation involves driver-level initialization
- **Fix**: Cache `cudnnHandle_t` and all descriptors in a persistent struct; only call `cudnnSetStream()` per invocation
- **Evidence**: KernelBench pointwise conv improved 93% (45ms → 3.4ms) from state caching alone

## P2: CUDNN_TENSOR_OP_MATH Not Set
- **Symptom**: FP32 convolution not using tensor cores on Ampere/Hopper
- **Root cause**: cuDNN defaults to `CUDNN_DEFAULT_MATH` which avoids TF32 tensor cores for backward compatibility
- **Fix**: Call `cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH)` before selecting algorithm
- **Speedup**: ~2-4x on H100/H200 for compute-bound convolutions
- **Note**: TF32 has lower precision than IEEE FP32 (10-bit mantissa vs 23-bit); acceptable for inference, verify for training

## P3: Wrong Algorithm for Depthwise Convolution
- **Symptom**: Depthwise conv slow despite being memory-bound
- **Root cause**: Using `IMPLICIT_PRECOMP_GEMM` (high register pressure) instead of `IMPLICIT_GEMM` (lower registers, better occupancy)
- **Fix**: For depthwise (group_count = channels), prefer `CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM`
- **Also**: Must call `cudnnSetConvolutionGroupCount(conv_desc, in_channels)` and set weight shape to `(channels, 1, kH, kW)`

## P4: Winograd Used with Dilation or Stride > 1
- **Symptom**: `cudnnGetConvolutionForwardWorkspaceSize` returns error for Winograd
- **Root cause**: Winograd only works with kernel=3x3, stride=1, dilation=1
- **Fix**: Check constraints before selecting Winograd; implement fallback chain:
  ```
  try WINOGRAD_NONFUSED → if error → fallback to IMPLICIT_PRECOMP_GEMM
  ```

## P5: Workspace Not Allocated
- **Symptom**: cuDNN returns `CUDNN_STATUS_NOT_SUPPORTED` or falls back to slow algorithm
- **Root cause**: Some algorithms (Winograd, FFT, GEMM) require workspace memory; if not provided, cuDNN silently uses a slower zero-workspace algorithm
- **Fix**: Always call `cudnnGetConvolutionForwardWorkspaceSize()` and allocate workspace before `cudnnConvolutionForward()`
- **Fix**: Cache workspace allocation across calls (same conv shape reuses same workspace)

## P6: Depthwise Custom Kernel — Non-Coalesced Weight Access
- **Symptom**: Custom depthwise kernel slower than expected despite simple computation
- **Root cause**: Weight layout `[C, kH, kW]` means accessing weights for same (kh,kw) across different channels is strided
- **Fix**: Since depthwise weights are small (C * kH * kW), load all weights into shared memory once per block
- **Fix**: Or use `__ldg()` to leverage read-only cache (weights are read-only and small)

## P7: Transposed Conv Implemented as Zero-Insert + Forward Conv
- **Symptom**: Transposed convolution wastes computation on zero-valued input
- **Root cause**: Inserting zeros between input elements (fractional stride) then running forward conv computes many multiply-by-zero operations
- **Fix**: Use cuDNN's `cudnnConvolutionBackwardData()` which implements transposed conv natively without zero insertion

## P8: NCHW Layout for Pointwise (1x1) Conv
- **Symptom**: 1x1 convolution underperforms despite being a simple channel-mixing GEMM
- **Root cause**: In NCHW layout, channel dimension is not contiguous in memory for a single spatial position; this prevents vectorized channel loads
- **Fix**: Use NHWC layout for pointwise conv — channels are contiguous, enabling float4 vectorized loads and better tensor core utilization
- **Fix**: Or reformulate as explicit GEMM: reshape input (N*H*W, C_in) x weight (C_in, C_out) → output (N*H*W, C_out)

## P9: Forgetting Bias Addition After cuDNN Convolution
- **Symptom**: Output is missing bias term
- **Root cause**: `cudnnConvolutionForward()` does not add bias; it only computes conv(input, weight)
- **Fix**: Call `cudnnAddTensor()` after forward conv to add bias:
  ```cuda
  cudnnAddTensor(handle, &alpha, bias_desc, bias, &alpha, output_desc, output);
  ```
- **Note**: bias descriptor shape must be `(1, out_channels, 1, 1)` for NCHW

## P10: 3D Convolution with Wrong Descriptor Dimensions
- **Symptom**: 3D convolution crashes or produces wrong results
- **Root cause**: Using `cudnnSetTensor4dDescriptor` for 5D tensor (N, C, D, H, W)
- **Fix**: Use `cudnnSetTensorNdDescriptor` with 5 dimensions for 3D conv:
  ```cuda
  int dims[] = {N, C, D, H, W};
  int strides[] = {C*D*H*W, D*H*W, H*W, W, 1};
  cudnnSetTensorNdDescriptor(desc, CUDNN_DATA_FLOAT, 5, dims, strides);
  ```

## P11: PyTorch Extension Build Flags Conflict with Half Precision
- **Symptom**: Compile error when using `__half` in conv kernels built via `torch.utils.cpp_extension`
- **Root cause**: PyTorch adds `-D__CUDA_NO_HALF_OPERATORS__` and `-D__CUDA_NO_HALF_CONVERSIONS__`, disabling `__half` operator overloads and implicit float→half conversion
- **Fix**: Use explicit conversion functions: `__float2half_rn()`, `__half2float()`, `__hgt()` instead of operators
- **Fix**: Always `#include <cuda_fp16.h>` when using `__half` types
- **Source**: Discovered in KernelBench experiment — 14 compile failures in half-precision conv attempts
