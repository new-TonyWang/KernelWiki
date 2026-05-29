---
title: Elementwise Library Fallback Paths
pattern_class: cuda-core
op: elementwise
source:
- path: spec
  anchor: Reference
id: routing-elementwise-library-fallback
type: operator-routing
vendor: nvidia
operator: elementwise
source_refs:
- source_id: source-code/cuda-samples
  path: Samples/0_Introduction/vectorAdd/vectorAdd.cu
  anchor: L47-L54
- source_id: cuda-official/toolkit-docs-13.2
  path: cuda-official/cuda-toolkit-documentation-13.2/CUDA API References/thrust/thrust.md
  anchor: L31965-L32063
---
# Elementwise -- Library Fallback Paths

Before writing a custom CUDA kernel for an elementwise operation, consider these library paths. They cover the vast majority of use cases with less code and comparable performance.

---

## 1. PyTorch built-in ops

PyTorch provides optimized ATen kernels for all common elementwise operations. For contiguous tensors, these kernels are vectorized and coalesced.

### Single-op examples

```python
import torch

x = torch.randn(1 << 20, device="cuda", dtype=torch.float32)
y = torch.randn_like(x)

# Arithmetic
z = torch.add(x, y)          # or: z = x + y
z = torch.mul(x, y)          # or: z = x * y
z = torch.sub(x, y)          # or: z = x - y
z = torch.div(x, y)          # or: z = x / y

# Activations
z = torch.relu(x)            # max(0, x)
z = torch.sigmoid(x)         # 1 / (1 + exp(-x))
z = torch.nn.functional.gelu(x)
z = torch.nn.functional.silu(x)   # x * sigmoid(x), also called swish

# Type casting
z = x.to(torch.float16)      # fp32 -> fp16
z = x.to(torch.bfloat16)     # fp32 -> bf16
z = x.half()                 # shorthand for fp16 cast

# In-place variants (avoid intermediate allocation)
x.add_(y)                    # x += y
x.relu_()                    # x = max(0, x)
x.mul_(2.0)                  # x *= 2.0
```

### When this is enough

- The operation is a single torch built-in (add, mul, relu, sigmoid, gelu, silu, abs, neg, exp, log, sqrt, rsqrt, clamp, ...).
- The input tensors are contiguous in memory.
- No fusion with adjacent operations is needed (or the operation is already the fused variant, e.g., `torch.addcmul`).

### When this is NOT enough

- Multiple consecutive elementwise ops that materialize intermediate tensors (each op reads from and writes to HBM separately).
- Non-contiguous tensor layouts causing uncoalesced access in the ATen kernel (e.g., after a transpose or slice).

---

## 2. torch.compile (Triton codegen for fused chains)

`torch.compile` with the default `inductor` backend detects chains of pointwise operations and fuses them into a single Triton kernel. This eliminates intermediate memory traffic.

### Fused chain examples

```python
import torch

@torch.compile
def fused_bias_gelu(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(x + bias)

@torch.compile
def fused_scale_add_relu(x: torch.Tensor, scale: float,
                         y: torch.Tensor) -> torch.Tensor:
    return torch.relu(x * scale + y)

@torch.compile
def fused_cast_and_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    # fp32 compute, then cast result to fp16
    return (x * scale).to(torch.float16)

# Usage -- first call triggers compilation, subsequent calls are fast
x = torch.randn(4096, 4096, device="cuda")
bias = torch.randn(4096, device="cuda")
out = fused_bias_gelu(x, bias)
```

### When this is enough

- The fused chain is a composition of standard torch ops (arithmetic + activations + casts).
- The tensors are contiguous.
- The compilation overhead of the first call is acceptable (typically a few seconds; amortized over many subsequent calls).

### When this is NOT enough

- The operation involves custom logic that torch.compile cannot trace (e.g., data-dependent control flow, custom memory layouts).
- Compilation latency is unacceptable for the use case (e.g., single-shot inference on a new shape).
- The operation must integrate with a larger custom CUDA kernel (e.g., a custom fused attention block).

---

## 3. thrust::transform (C++ device-side)

For C++ CUDA code, `thrust::transform` applies a user-defined functor to each element of a device range. Thrust handles grid sizing and launches an efficient elementwise kernel internally.

### Unary transform example

```cpp
#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>

// Negate every element in-place
thrust::device_vector<float> d_vec(N);
// ... fill d_vec ...
thrust::transform(d_vec.begin(), d_vec.end(),
                  d_vec.begin(),
                  ::cuda::std::negate<float>{});
```

Source: thrust documentation, thrust::transform (`corpus/nvidia/cuda-official/cuda-toolkit-documentation-13.2/CUDA API References/thrust/thrust.md` L31965-L32063).

### Binary transform example

```cpp
#include <thrust/transform.h>
#include <thrust/device_vector.h>

// C[i] = A[i] + B[i]
thrust::device_vector<float> A(N), B(N), C(N);
// ... fill A, B ...
thrust::transform(A.begin(), A.end(),
                  B.begin(),
                  C.begin(),
                  thrust::plus<float>{});
```

### Custom functor example

```cpp
struct ScaleAddRelu {
    float scale;
    __host__ __device__
    float operator()(float x, float y) const {
        float val = x * scale + y;
        return val > 0.0f ? val : 0.0f;
    }
};

thrust::transform(X.begin(), X.end(),
                  Y.begin(),
                  Out.begin(),
                  ScaleAddRelu{2.0f});
```

### When this is enough

- Writing a C++ CUDA application (not Python/PyTorch).
- The operation fits a simple unary or binary functor.
- Performance does not need to be at the absolute peak (thrust's generated kernel is good but may not use vectorized loads).

### When this is NOT enough

- The operation requires vectorized loads (float4) for maximum bandwidth -- thrust does not expose this control.
- The functor needs access to neighboring elements (not elementwise).
- The operation must be fused with non-elementwise stages in the same kernel launch.

---

## 4. CUDA Samples reference: vectorAdd

The canonical CUDA elementwise example is the `vectorAdd` sample from NVIDIA's cuda-samples repository. It demonstrates the minimal structure of an elementwise kernel.

```cuda
// From: cuda-samples/Samples/0_Introduction/vectorAdd/vectorAdd.cu L47-L54
__global__ void vectorAdd(const float *A, const float *B,
                          float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

// Launch configuration:
int threadsPerBlock = 256;
int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
```

This kernel is correctly coalesced (stride-1 access) but does not use vectorized loads. For a custom kernel that needs peak bandwidth, the next step is to apply the coalescing skill and then vectorized loads (once the vectorized-access skill is built).
