# Convolution Pattern -- Skills

## When to Apply
- Standard 1D/2D/3D convolution (with stride, padding, dilation)
- Depthwise convolution (each channel independently)
- Pointwise (1x1) convolution (equivalent to channel-mixing GEMM)
- Depthwise-separable convolution (depthwise + pointwise fused)
- Grouped convolution (channels split into groups)
- Transposed convolution (deconvolution / fractionally-strided convolution)

## Core Characteristic
Convolution is a **sliding window + accumulation** pattern. For 2D convolution:
- Output[n,co,oh,ow] = sum over ci,kh,kw of Input[n,ci,ih,iw] * Weight[co,ci,kh,kw]
- where ih = oh * stride_h - pad_h + kh * dilation_h

The key optimization challenge is data reuse: each input element participates in multiple output elements. The ratio of computation to memory access depends heavily on kernel size, channel count, and spatial dimensions.

## Decision Framework: Custom Kernel vs cuDNN

### Use cuDNN when
- Standard convolution shapes (spatial >= 7, channels >= 32, batch >= 4)
- No custom epilogue needed beyond bias + activation
- Dilated or strided convolutions (cuDNN handles complex indexing internally)
- 3D convolution (custom kernels are significantly harder)
- Winograd-eligible shapes (3x3 kernel, stride=1, dilation=1)

### Write custom kernel when
- Depthwise convolution with unusual kernel shapes (e.g., Kx1 or 1xK asymmetric)
- Fused depthwise-separable (depthwise + pointwise in one kernel launch)
- Very small spatial dimensions where cuDNN overhead dominates
- Custom epilogue (e.g., conv + custom activation + residual)
- Need to avoid cuDNN library dependency

### Experimental evidence
In KernelBench Level 1 experiments (H200 GPU):
- Standard 2D/3D convolution: cuDNN with proper configuration beats custom kernels
- Depthwise with asymmetric kernel: custom CUDA kernel beat cuDNN by **90%** (659us vs 6472us)
- Pointwise 1x1: cuDNN with cached state beat naive by **93%** (3426us vs 45525us)
- Dilated+padded: cuDNN with `CUDNN_TENSOR_OP_MATH` beat naive by **86%** (2310us vs 16387us)

## Skill 1: cuDNN Algorithm Selection

Different cuDNN algorithms suit different convolution shapes:

| Algorithm | Best for | Notes |
|-----------|----------|-------|
| `IMPLICIT_PRECOMP_GEMM` | General purpose, 1x1 (pointwise), dilated | Supports tensor cores via `CUDNN_TENSOR_OP_MATH`. Most flexible. |
| `IMPLICIT_GEMM` | Depthwise, memory-bound | Lower register pressure, higher occupancy. Use with `cudnnSetConvolutionGroupCount`. |
| `WINOGRAD_NONFUSED` | 3x3 kernel, stride=1, dilation=1 | 2.25x fewer multiplications. Not applicable to all shapes. |
| `FFT` / `FFT_TILING` | Large kernels (>= 7x7) | Transforms conv to pointwise multiply in frequency domain. |
| `GEMM` | Small batch, large channels | Explicit im2col + GEMM. High workspace cost. |

### Algorithm selection pattern:
```cuda
// Option 1: Let cuDNN choose (reliable but may not find fastest)
cudnnConvolutionFwdAlgo_t algo;
cudnnGetConvolutionForwardAlgorithm_v7(
    handle, input_desc, weight_desc, conv_desc, output_desc,
    requested_algo_count, &returned_algo_count, perf_results);
algo = perf_results[0].algo;  // Pick fastest

// Option 2: Direct selection (when you know the best algorithm)
// For 3x3 stride=1: try Winograd first
algo = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;
// Fallback to IMPLICIT_PRECOMP_GEMM if Winograd not supported
```

### Enable tensor core acceleration:
```cuda
// Critical for performance on Ampere/Hopper GPUs
cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH);
// This allows cuDNN to use TF32 tensor cores for FP32 convolution
// Provides ~2-4x speedup with minimal precision loss
```

Source: cuDNN Developer Guide — "Convolution Algorithm Selection", "Math Type" sections

## Skill 2: cuDNN State Caching

Creating cuDNN handles and descriptors per call is expensive. Cache and reuse:

```cuda
// Anti-pattern: create/destroy per call (causes 10-100x overhead)
void conv_naive(const float* input, ...) {
    cudnnHandle_t handle;
    cudnnCreate(&handle);           // expensive!
    // ... do convolution ...
    cudnnDestroy(handle);           // wasted setup
}

// Correct pattern: persistent state
struct ConvState {
    cudnnHandle_t handle;
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnFilterDescriptor_t weight_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnConvolutionFwdAlgo_t algo;
    size_t workspace_size;
    void* workspace;
    // Cache dimensions for change detection
    int cached_batch, cached_height, cached_width;
};

void conv_init(ConvState* s, int batch, int C, int H, int W, ...) {
    cudnnCreate(&s->handle);
    // Create all descriptors once
    cudnnCreateTensorDescriptor(&s->input_desc);
    // ... set descriptors, query workspace, allocate ...
    s->cached_batch = batch;
}

void conv_forward(ConvState* s, const float* input, ...) {
    cudnnSetStream(s->handle, stream);  // just set stream
    // Use cached descriptors directly
    cudnnConvolutionForward(s->handle, &alpha,
        s->input_desc, input, s->weight_desc, weight,
        s->conv_desc, s->algo, s->workspace, s->workspace_size,
        &beta, s->output_desc, output);
}
```

In KernelBench experiments, state caching turned a 45ms pointwise convolution into 3.4ms (93% improvement).

## Skill 3: Depthwise Convolution — Custom Kernel

Depthwise convolution processes each channel independently: Output[n,c,oh,ow] = sum over kh,kw of Input[n,c,ih,iw] * Weight[c,kh,kw]. This has very low arithmetic intensity (kernel_size^2 FLOPs per output element), making it **memory-bound**.

### Thread mapping: one thread per output element
```cuda
__global__ void depthwise_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch, int channels, int H, int W,
    int kH, int kW, int stride_h, int stride_w,
    int pad_h, int pad_w, int dilation_h, int dilation_w,
    int out_H, int out_W
) {
    int n_elems = batch * channels * out_H * out_W;

    // Grid-stride loop for arbitrary tensor sizes
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_elems;
         idx += gridDim.x * blockDim.x)
    {
        // Decode flat index → (n, c, oh, ow)
        int ow = idx % out_W;
        int oh = (idx / out_W) % out_H;
        int c  = (idx / (out_W * out_H)) % channels;
        int n  = idx / (out_W * out_H * channels);

        float sum = 0.0f;
        #pragma unroll
        for (int kh = 0; kh < kH; ++kh) {
            int ih = oh * stride_h - pad_h + kh * dilation_h;
            if (ih < 0 || ih >= H) continue;
            #pragma unroll
            for (int kw = 0; kw < kW; ++kw) {
                int iw = ow * stride_w - pad_w + kw * dilation_w;
                if (iw < 0 || iw >= W) continue;
                int in_idx = ((n * channels + c) * H + ih) * W + iw;
                int w_idx  = (c * kH + kh) * kW + kw;
                sum += __ldg(&input[in_idx]) * __ldg(&weight[w_idx]);
            }
        }
        output[idx] = sum;
    }
}
```

Key optimizations:
- **`__ldg()` for read-only inputs**: Leverages texture cache, avoids polluting L1
- **`#pragma unroll` on kernel loops**: Reduces loop overhead for small kernel sizes (3x3, 5x5)
- **Grid-stride loop**: Handles arbitrary batch/spatial sizes without recomputing grid dims
- **NCHW layout**: Weight access `weight[c * kH * kW + kh * kW + kw]` has stride-1 access along kw

In KernelBench experiments, this pattern beat cuDNN by 90% for asymmetric (Kx1) kernels.

## Skill 4: Depthwise-Separable Fusion

Depthwise-separable = depthwise conv + pointwise (1x1) conv. Fusing into one kernel avoids the intermediate tensor write/read:

```cuda
// Key optimization: cache depthwise weights in shared memory
__shared__ float s_dw_weight[MAX_CHANNELS * KH * KW];

// Phase 1: cooperatively load depthwise weights
for (int i = threadIdx.x; i < channels * kH * kW; i += blockDim.x)
    s_dw_weight[i] = weight_dw[i];
__syncthreads();

// Phase 2: for each output pixel, compute depthwise then pointwise
for (int idx = ...; idx < n_elems; idx += ...) {
    // Depthwise: accumulate across kernel window using shared memory weights
    float dw_out[MAX_CHANNELS];
    for (int c = 0; c < in_channels; ++c) {
        float sum = 0.0f;
        for (int kh = 0; kh < kH; ++kh)
            for (int kw = 0; kw < kW; ++kw)
                sum += __ldg(&input[...]) * s_dw_weight[c * kH * kW + kh * kW + kw];
        dw_out[c] = sum;
    }
    // Pointwise: channel mixing
    for (int oc = 0; oc < out_channels; ++oc) {
        float sum = 0.0f;
        for (int c = 0; c < in_channels; ++c)
            sum += dw_out[c] * __ldg(&weight_pw[oc * in_channels + c]);
        output[...] = sum;
    }
}
```

Benefits:
- Eliminates intermediate tensor (saves 2x HBM bandwidth for depthwise output)
- Depthwise weights cached in shared memory (one load per block, not per thread)
- Pointwise weights accessed via `__ldg()` for read-only caching

## Skill 5: Data Layout — NCHW vs NHWC

| Layout | Memory order | Coalescing | Tensor Core | cuDNN support |
|--------|-------------|------------|-------------|---------------|
| NCHW | batch → channel → height → width | Width-dim coalesced, channel-dim strided | Requires internal transpose | Full support |
| NHWC | batch → height → width → channel | Channel-dim coalesced | Native alignment | Full support, often faster |

### When to use NHWC:
- Tensor core operations (cuDNN internally uses NHWC for tensor core paths)
- Channel-wise vectorized loads (`float4` loads 4 channels at once)
- When followed by channel-wise operations (BN, activation)

### When to use NCHW:
- Depthwise convolution (each channel processed independently — spatial coalescing matters more)
- When input is already in NCHW (avoid layout conversion overhead)
- Custom kernels where spatial locality is key (e.g., stencil-like access)

For cuDNN:
```cuda
// NHWC format descriptor — often faster for standard conv
cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, N, C, H, W);

// NCHW format descriptor — default, safe choice
cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W);
```

Source: cuDNN Developer Guide — "Data Layout" section; NVIDIA Best Practices Guide — "Memory Coalescing"

## Skill 6: Depthwise via cuDNN Group Count API

cuDNN supports depthwise convolution via the grouped convolution API with groups = channels:

```cuda
// Set convolution as depthwise: group_count = in_channels
cudnnSetConvolutionGroupCount(conv_desc, in_channels);

// Use IMPLICIT_GEMM for depthwise (memory-bound, lower register pressure)
algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

// Weight descriptor: (channels, 1, kH, kW) — one filter per channel
cudnnSetFilter4dDescriptor(weight_desc, CUDNN_DATA_FLOAT,
    CUDNN_TENSOR_NCHW, channels, 1, kernel_h, kernel_w);
```

Key: `IMPLICIT_GEMM` is preferred over `IMPLICIT_PRECOMP_GEMM` for depthwise because it has lower register pressure and higher occupancy, which benefits the memory-bound access pattern of depthwise conv.

## Skill 7: Winograd Transform for 3x3 Convolutions

Winograd reduces multiplication count for small-kernel convolutions at the cost of more additions:

| Tile | Kernel | Multiplications | Standard | Reduction |
|------|--------|----------------|----------|-----------|
| F(2,3) | 3x3 | 16 | 36 | 2.25x |
| F(4,3) | 3x3 | 36 | 72 | 2.0x |
| F(6,3) | 3x3 | 64 | 108 | 1.69x |

### When Winograd applies:
- Kernel size = 3x3
- Stride = 1
- Dilation = 1 (not compatible with dilated conv)
- FP32 or FP16 (numerical stability limits larger tiles)

### cuDNN Winograd usage:
```cuda
algo = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;
// Verify it's supported for this configuration:
status = cudnnGetConvolutionForwardWorkspaceSize(
    handle, input_desc, weight_desc, conv_desc, output_desc,
    algo, &workspace_size);
if (status != CUDNN_STATUS_SUCCESS) {
    // Fallback to IMPLICIT_PRECOMP_GEMM
    algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
}
```

Source: cuDNN Developer Guide — "Winograd Convolution"; Lavin & Gray (2016) "Fast Algorithms for Convolutional Neural Networks"

## Skill 8: Transposed Convolution

Transposed convolution (deconvolution) upsamples spatial dimensions. Two implementation approaches:

### Approach 1: cuDNN backward-data API
```cuda
// Transposed conv is mathematically the backward pass of a forward conv
cudnnConvolutionBackwardData(
    handle, &alpha,
    weight_desc, weight,        // same weight
    input_desc, input,          // "gradient" is actually the input
    conv_desc,
    algo,
    workspace, workspace_size,
    &beta,
    output_desc, output         // upsampled output
);
```

### Approach 2: im2col with scattered writes
- Insert zeros between input elements (fractional stride)
- Run standard convolution on the zero-padded input
- Less efficient than approach 1 due to wasted computation on zeros

Recommendation: always use cuDNN backward-data API for transposed conv.

## Cross-References
- pattern/gemm — pointwise (1x1) conv reduces to GEMM; im2col + GEMM approach
- pattern/reduction — global/average pooling often follows conv
- pattern/elementwise — activation fusion after conv
- optimization/memory/vectorized-access — float4 loads for NHWC channel dimension
- optimization/memory/shared-memory-cache — weight caching for depthwise
- optimization/memory/coalescing — NCHW vs NHWC coalescing tradeoffs
- optimization/compute/tensor-core — WMMA for mixed-precision conv
- optimization/latency/gpu-library-usage — cuDNN integration patterns
- hardware/tensor-core-specs — valid MMA shapes for conv with tensor cores
