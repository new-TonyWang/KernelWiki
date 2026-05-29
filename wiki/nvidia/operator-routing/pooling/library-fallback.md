---
title: Pooling Pattern -- Library Fallback Paths
pattern_class: cuda-core
op: pooling
status: draft
source:
- path: spec
  anchor: Reference
id: routing-pooling-library-fallback
type: operator-routing
vendor: nvidia
operator: pooling
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L538-L542
---
# Pooling -- Library Fallback Paths

Before writing a custom pooling kernel, use one of the following library paths. These are production-quality, GPU-optimized implementations that handle edge cases (non-power-of-two sizes, various padding modes, multi- dimensional windows) and are well-tested across CUDA toolkit releases.

---

## 1. PyTorch pooling ops

**When to use**: the caller's workflow is PyTorch-based and the pooling operation is a standard max-pool, avg-pool, or adaptive-pool on a `torch.Tensor` residing on a CUDA device. PyTorch dispatches these operations to the cuDNN backend by default, so performance is near optimal for standard configurations.

### torch.nn.functional.max_pool2d

Computes 2D max-pooling over an input tensor of shape (N, C, H, W).

```python
import torch
import torch.nn.functional as F

x = torch.randn(32, 64, 224, 224, device="cuda", dtype=torch.float32)

# Standard 2x2 max-pool with stride 2
out = F.max_pool2d(x, kernel_size=2, stride=2)
# out.shape: (32, 64, 112, 112)

# 3x3 max-pool with stride 2 and padding 1
out = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
# out.shape: (32, 64, 112, 112)

# With return_indices (needed for max_unpool2d)
out, indices = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)

# ceil_mode: use ceil instead of floor for output size computation
out = F.max_pool2d(x, kernel_size=3, stride=2, padding=1, ceil_mode=True)

# Dilation (dilated/atrous max-pooling)
out = F.max_pool2d(x, kernel_size=3, stride=1, padding=1, dilation=2)
```

**Shape range**: any (N, C, H, W) that fits in GPU memory. H and W must be >= kernel_size (after padding). Typical use: ImageNet-scale inputs (224x224) down to feature maps (7x7).

**Supported dtypes**: float16, bfloat16, float32, float64.

### torch.nn.functional.avg_pool2d

Computes 2D average-pooling.

```python
import torch
import torch.nn.functional as F

x = torch.randn(32, 64, 224, 224, device="cuda", dtype=torch.float32)

# Standard 2x2 avg-pool with stride 2
out = F.avg_pool2d(x, kernel_size=2, stride=2)
# out.shape: (32, 64, 112, 112)

# count_include_padding: whether to include zero-padding in the average
out = F.avg_pool2d(x, kernel_size=3, stride=2, padding=1,
                   count_include_padding=False)

# divisor_override: use a custom divisor instead of window_area
out = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
                   divisor_override=8)
```

**Note on count_include_padding**: when `count_include_padding=True` (the default), border elements where the window extends beyond the input are divided by the full window area (kH * kW). When `False`, the divisor is the number of valid (non-padded) input elements in the window.

### torch.nn.functional.adaptive_avg_pool2d

Computes adaptive average-pooling: the output size is specified, and the kernel size and stride are computed automatically.

```python
import torch
import torch.nn.functional as F

x = torch.randn(32, 64, 224, 224, device="cuda", dtype=torch.float32)

# Adaptive avg-pool to 7x7 (common before FC layers)
out = F.adaptive_avg_pool2d(x, output_size=(7, 7))
# out.shape: (32, 64, 7, 7)

# Adaptive avg-pool to 1x1 (global average pooling)
out = F.adaptive_avg_pool2d(x, output_size=(1, 1))
# out.shape: (32, 64, 1, 1)
```

**Global average pooling** (`output_size=(1, 1)`) is the most common adaptive-pool use case, used in ResNet/EfficientNet before the final classifier. This reduces to a per-channel mean over all spatial positions.

### torch.nn.functional.adaptive_max_pool2d

```python
import torch
import torch.nn.functional as F

x = torch.randn(32, 64, 224, 224, device="cuda", dtype=torch.float32)

out = F.adaptive_max_pool2d(x, output_size=(7, 7))
# out.shape: (32, 64, 7, 7)

# With return_indices
out, indices = F.adaptive_max_pool2d(x, output_size=(7, 7),
                                      return_indices=True)
```

### 1D and 3D variants

PyTorch provides matching 1D and 3D pooling functions:

| Variant | Function |
|---|---|
| 1D max-pool | `F.max_pool1d(x, kernel_size, stride, padding)` |
| 1D avg-pool | `F.avg_pool1d(x, kernel_size, stride, padding)` |
| 3D max-pool | `F.max_pool3d(x, kernel_size, stride, padding)` |
| 3D avg-pool | `F.avg_pool3d(x, kernel_size, stride, padding)` |
| 1D adaptive avg-pool | `F.adaptive_avg_pool1d(x, output_size)` |
| 3D adaptive avg-pool | `F.adaptive_avg_pool3d(x, output_size)` |

Input shapes: 1D expects (N, C, L), 3D expects (N, C, D, H, W).

---

## 2. cuDNN pooling (cudnnPoolingForward)

**When to use**: writing a standalone CUDA/C++ program (not PyTorch) and the pooling is a standard max-pool or avg-pool over a 4D tensor in NCHW or NHWC layout. cuDNN provides heavily optimized pooling kernels that are the backend for PyTorch's pooling ops.

**Header**: `#include <cudnn.h>`

**Link**: `-lcudnn`

### Setup pattern

cuDNN pooling requires creating descriptors for the pooling operation, input tensor, and output tensor.

```cpp
#include <cudnn.h>

// 1. Create a cuDNN handle
cudnnHandle_t handle;
cudnnCreate(&handle);

// 2. Create a pooling descriptor
cudnnPoolingDescriptor_t poolDesc;
cudnnCreatePoolingDescriptor(&poolDesc);
cudnnSetPooling2dDescriptor(
    poolDesc,
    CUDNN_POOLING_MAX,                  // mode
    CUDNN_NOT_PROPAGATE_NAN,            // NaN propagation
    3, 3,                               // windowHeight, windowWidth
    1, 1,                               // verticalPadding, horizontalPadding
    2, 2                                // verticalStride, horizontalStride
);

// 3. Create tensor descriptors for input and output
cudnnTensorDescriptor_t inDesc, outDesc;
cudnnCreateTensorDescriptor(&inDesc);
cudnnCreateTensorDescriptor(&outDesc);

// Set input descriptor (NCHW)
cudnnSetTensor4dDescriptor(
    inDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    32, 64, 224, 224  // N, C, H, W
);

// Compute output dimensions
int outN, outC, outH, outW;
cudnnGetPooling2dForwardOutputDim(
    poolDesc, inDesc, &outN, &outC, &outH, &outW
);

// Set output descriptor
cudnnSetTensor4dDescriptor(
    outDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    outN, outC, outH, outW
);

// 4. Run the forward pooling
float alpha = 1.0f, beta = 0.0f;
cudnnPoolingForward(
    handle,
    poolDesc,
    &alpha,
    inDesc, d_input,
    &beta,
    outDesc, d_output
);

// 5. Cleanup
cudnnDestroyPoolingDescriptor(poolDesc);
cudnnDestroyTensorDescriptor(inDesc);
cudnnDestroyTensorDescriptor(outDesc);
cudnnDestroy(handle);
```

### Pooling modes

| Mode constant | Behavior |
|---|---|
| `CUDNN_POOLING_MAX` | Maximum over the window |
| `CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING` | Average including zero-padded positions |
| `CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING` | Average excluding zero-padded positions |
| `CUDNN_POOLING_MAX_DETERMINISTIC` | Deterministic max-pool (for reproducibility) |

### N-dimensional pooling

For 3D pooling or non-standard dimensionality, use the Nd API:

```cpp
cudnnSetPoolingNdDescriptor(
    poolDesc,
    CUDNN_POOLING_MAX,
    CUDNN_NOT_PROPAGATE_NAN,
    3,                       // nbDims
    windowDimA,              // int array: {kD, kH, kW}
    paddingA,                // int array: {padD, padH, padW}
    strideA                  // int array: {strD, strH, strW}
);
```

**Performance note**: cuDNN selects the fastest available algorithm internally based on the tensor layout, window size, and hardware. For standard configurations (2x2 or 3x3 windows, stride 1 or 2, NCHW/NHWC layout), the cuDNN implementation approaches the memory- bandwidth limit of the device.

---

## Decision summary

| Scenario | Recommended library | API |
|---|---|---|
| PyTorch tensor, standard max-pool | PyTorch (cuDNN) | `F.max_pool2d` |
| PyTorch tensor, standard avg-pool | PyTorch (cuDNN) | `F.avg_pool2d` |
| PyTorch tensor, adaptive pool (e.g., global avg) | PyTorch (cuDNN) | `F.adaptive_avg_pool2d` |
| PyTorch tensor, 1D or 3D pooling | PyTorch (cuDNN) | `F.max_pool1d`, `F.avg_pool3d`, etc. |
| CUDA C++, standard 2D/3D pooling | cuDNN | `cudnnPoolingForward` |
| Fused pooling + activation | Custom kernel | See INDEX.md Step 1 |
| Non-standard window / reduction rule | Custom kernel | See INDEX.md Step 1 |
| Weighted / Lp-norm / stochastic pooling | Custom kernel | See INDEX.md Step 1 |
