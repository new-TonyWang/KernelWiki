---
title: Normalization Pattern -- Library Fallback Paths
pattern_class: cuda-core
op: normalization
status: draft
source:
- path: spec
  anchor: Reference
id: routing-normalization-library-fallback
type: operator-routing
vendor: nvidia
operator: normalization
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Tools/nsight-systems/nsight-systems_index.html.md
  anchor: L6220-L6224
---
# Normalization -- Library Fallback Paths

Before writing a custom normalization kernel, use one of the following library paths. These are production-quality, GPU-optimized implementations that handle edge cases (arbitrary dtypes, mixed precision, running statistics) and are well-tested.

---

## 1. PyTorch normalization ops

**When to use**: the caller's workflow is PyTorch-based and the normalization is a standard variant on a CUDA tensor.

### torch.nn.functional.layer_norm

Applies Layer Normalization over the last D dimensions (the `normalized_shape`). Computes per-token mean and variance, then applies affine transform.

```python
import torch
import torch.nn.functional as F

# LayerNorm over hidden dim H=1024
x = torch.randn(32, 512, 1024, device="cuda", dtype=torch.float32)
gamma = torch.ones(1024, device="cuda")
beta = torch.zeros(1024, device="cuda")
out = F.layer_norm(x, normalized_shape=(1024,), weight=gamma, bias=beta, eps=1e-5)
# out.shape: (32, 512, 1024)
```

**Shape range**: any tensor shape; `normalized_shape` specifies the trailing dimensions to normalize over. Most common: `(H,)` for transformers.

**Supported dtypes**: float16, bfloat16, float32, float64. Mixed precision (fp16 input with fp32 gamma/beta) is handled automatically.

**Performance notes**: PyTorch's LayerNorm dispatches to an internal fused CUDA kernel. For standard transformer shapes (H in 768..8192), this is highly optimized. Custom kernels typically only win when fusing with adjacent ops (residual add, dropout).

### torch.nn.functional.rms_norm

Applies Root Mean Square Layer Normalization. Available in PyTorch >= 2.4. Computes `x / sqrt(mean(x^2) + eps) * gamma` without subtracting the mean.

```python
import torch
import torch.nn.functional as F

x = torch.randn(32, 512, 4096, device="cuda", dtype=torch.bfloat16)
gamma = torch.ones(4096, device="cuda", dtype=torch.bfloat16)
out = F.rms_norm(x, normalized_shape=(4096,), weight=gamma, eps=1e-6)
```

**When to prefer over LayerNorm**: RMSNorm skips the mean computation, saving one reduction pass. Modern LLMs (LLaMA, Gemma, Mistral) use RMSNorm. Approximately 1.3-1.5x faster than LayerNorm for the same shape.

### torch.nn.functional.batch_norm

Applies Batch Normalization over the channel dimension of a 4-D (NCHW) or 5-D (NCDHW) tensor. Computes per-channel statistics across the batch and spatial dimensions.

```python
import torch
import torch.nn.functional as F

x = torch.randn(64, 256, 14, 14, device="cuda", dtype=torch.float32)
gamma = torch.ones(256, device="cuda")
beta = torch.zeros(256, device="cuda")
running_mean = torch.zeros(256, device="cuda")
running_var = torch.ones(256, device="cuda")

# Training mode
out = F.batch_norm(x, running_mean, running_var,
                   weight=gamma, bias=beta,
                   training=True, momentum=0.1, eps=1e-5)

# Inference mode (uses running statistics)
out = F.batch_norm(x, running_mean, running_var,
                   weight=gamma, bias=beta,
                   training=False, eps=1e-5)
```

**Shape range**: 4-D `(N, C, H, W)` or 5-D `(N, C, D, H, W)`.

**Performance notes**: PyTorch dispatches to cuDNN's BatchNorm implementation, which is heavily optimized. Custom kernels are rarely needed for standalone BatchNorm.

### torch.nn.functional.group_norm

Applies Group Normalization. Divides channels into groups and computes per-group statistics across spatial dimensions.

```python
import torch
import torch.nn.functional as F

x = torch.randn(32, 256, 14, 14, device="cuda", dtype=torch.float32)
num_groups = 32
gamma = torch.ones(256, device="cuda")
beta = torch.zeros(256, device="cuda")
out = F.group_norm(x, num_groups, weight=gamma, bias=beta, eps=1e-5)
```

**Shape range**: any tensor with at least 3 dimensions. Channels dimension must be divisible by `num_groups`.

---

## 2. cuDNN BatchNormalization

**When to use**: writing a standalone CUDA/C++ program (not PyTorch) that needs BatchNorm. cuDNN provides the highest-performance implementation with support for training (forward + backward) and inference.

**API**: `cudnnBatchNormalizationForwardTraining` / `cudnnBatchNormalizationForwardInference`

cuDNN BatchNorm supports NCHW and NHWC tensor layouts and handles running mean/variance accumulation, exponential moving average, and epsilon parameters.

**Note**: cuDNN does not provide LayerNorm, RMSNorm, or GroupNorm as standalone operations. For these variants in C++, a custom kernel is needed.

---

## 3. NVIDIA Apex fused normalization kernels

**When to use**: the caller uses PyTorch and needs maximum performance for LayerNorm or RMSNorm, especially in mixed-precision (fp16/bf16) training.

### apex.normalization.FusedLayerNorm

```python
from apex.normalization import FusedLayerNorm

norm = FusedLayerNorm(normalized_shape=1024).cuda()
x = torch.randn(32, 512, 1024, device="cuda", dtype=torch.float16)
out = norm(x)
```

### apex.normalization.FusedRMSNorm

```python
from apex.normalization import FusedRMSNorm

norm = FusedRMSNorm(normalized_shape=4096).cuda()
x = torch.randn(32, 512, 4096, device="cuda", dtype=torch.bfloat16)
out = norm(x)
```

**Performance notes**: Apex fused kernels are hand-tuned CUDA implementations that can be 10-30% faster than PyTorch's built-in LayerNorm for typical transformer shapes. They also support fused residual-add + normalization via `FusedLayerNorm` with residual inputs.

**Availability**: requires `pip install apex` with CUDA extensions. Apex is maintained by NVIDIA but is not part of the core PyTorch distribution.

---

## 4. Custom kernel (last resort)

**When to use**: none of the above library paths meet the performance or functionality requirements. Common scenarios:

- Fusing normalization with non-standard adjacent ops (e.g., residual + layernorm + dropout in a single kernel).
- Custom normalization variants not supported by any library.
- The measured library latency exceeds the theoretical bandwidth-bound limit by more than 10% for the given shape.

Proceed to `INDEX.md` Step 1 for the custom kernel strategy.

---

## Decision summary

| Scenario | Recommended library | API |
|---|---|---|
| PyTorch, LayerNorm, any shape | PyTorch | `F.layer_norm` |
| PyTorch, RMSNorm, any shape | PyTorch (>= 2.4) | `F.rms_norm` |
| PyTorch, BatchNorm, NCHW/NHWC | PyTorch (cuDNN backend) | `F.batch_norm` |
| PyTorch, GroupNorm | PyTorch | `F.group_norm` |
| PyTorch, max perf LayerNorm/RMSNorm | Apex | `FusedLayerNorm` / `FusedRMSNorm` |
| CUDA C++, BatchNorm | cuDNN | `cudnnBatchNormalizationForward*` |
| CUDA C++, LayerNorm/RMSNorm/GroupNorm | Custom kernel | See INDEX.md Step 1 |
| Fused residual + norm + dropout | Custom kernel | See INDEX.md Step 1 |
