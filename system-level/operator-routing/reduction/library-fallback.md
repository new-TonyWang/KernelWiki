---
title: Reduction Pattern -- Library Fallback Paths
pattern_class: cuda-core
op: reduction
status: draft
source:
- path: spec
  anchor: Reference
id: routing-reduction-library-fallback
type: operator-routing
vendor: nvidia
operator: reduction
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA API References/cub/cub.md
  anchor: L26607-L26923
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA API References/cub/cub.md
  anchor: L26886-L26910
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA API References/cub/cub.md
  anchor: L26956-L26982
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA API References/cccl/cccl.md
  anchor: L10641-L10643
languages:
- cuda-cpp
- python
techniques:
- kernel-fusion
kernel_types:
- fused-kernel
confidence: inferred
tags:
- kernel-fusion
- fused-kernel
- cuda-cpp
- python
---
# Reduction -- Library Fallback Paths

Before writing a custom reduction kernel, use one of the following library paths. These are production-quality, GPU-optimized implementations that handle edge cases (non-power-of-two sizes, arbitrary dtypes, stream integration) and are well-tested across CUDA toolkit releases.

---

## 1. PyTorch reduction ops

**When to use**: the caller's workflow is PyTorch-based and the reduction is a standard operation on a `torch.Tensor` residing on a CUDA device.

### torch.sum

Reduces a tensor along one or more dimensions (or all dimensions).

```python
import torch

# Full reduction (scalar output)
x = torch.randn(4096, device="cuda", dtype=torch.float32)
total = torch.sum(x)                 # shape: ()

# Axis reduction (reduce dim=1 of a 2-D tensor)
x = torch.randn(128, 4096, device="cuda", dtype=torch.float32)
row_sums = torch.sum(x, dim=1)       # shape: (128,)

# Keep dimension for broadcasting
row_sums_keepdim = torch.sum(x, dim=1, keepdim=True)  # shape: (128, 1)
```

**Shape range**: any tensor shape that fits in GPU memory. PyTorch internally dispatches to CUB or custom CUDA kernels depending on shape and dtype. For very small tensors (<1024 elements), kernel launch overhead may dominate.

**Supported dtypes**: float16, bfloat16, float32, float64, int32, int64, and complex types.

### torch.mean

```python
x = torch.randn(128, 4096, device="cuda", dtype=torch.float32)
row_means = torch.mean(x, dim=1)     # shape: (128,)
```

Equivalent to `torch.sum(x, dim) / x.shape[dim]`. Same shape/dtype support as `torch.sum`.

### torch.max / torch.min

```python
# Global max (scalar)
x = torch.randn(4096, device="cuda", dtype=torch.float32)
global_max = torch.max(x)            # shape: ()

# Per-axis max (returns values and indices)
x = torch.randn(128, 4096, device="cuda", dtype=torch.float32)
values, indices = torch.max(x, dim=1)  # shapes: (128,), (128,)
```

### torch.argmax / torch.argmin

```python
x = torch.randn(128, 4096, device="cuda", dtype=torch.float32)
idx = torch.argmax(x, dim=1)         # shape: (128,), dtype=int64
```

**Limitation**: PyTorch does not guarantee deterministic tie-breaking for argmax/argmin when multiple elements share the maximum/minimum value.

---

## 2. CUB device-wide reduction (cub::DeviceReduce)

**When to use**: writing a standalone CUDA/C++ program (not PyTorch) and the reduction is over a flat, contiguous 1-D buffer with a standard or custom binary operator. CUB provides the highest-performance single-kernel reduction for this case.

**Include**: `#include <cub/cub.cuh>` or `#include <cub/device/device_reduce.cuh>`

**Common pattern**: CUB device APIs use a two-pass temp-storage idiom:
1. Call with `d_temp_storage = nullptr` to query required temp size.
2. Allocate temp storage.
3. Call again to run the reduction.

### cub::DeviceReduce::Sum

Computes the sum of all elements. Initial value is `0`.

```cpp
#include <cub/cub.cuh>

int   num_items = 1 << 20;  // 1M elements
float *d_in;                 // device input, num_items floats
float *d_out;                // device output, 1 float

// Step 1: query temp storage
void  *d_temp_storage = nullptr;
size_t temp_storage_bytes = 0;
cub::DeviceReduce::Sum(
    d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);

// Step 2: allocate
cudaMalloc(&d_temp_storage, temp_storage_bytes);

// Step 3: run
cub::DeviceReduce::Sum(
    d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
```

**Performance note**: CUB's work-complexity is O(n) and throughput plateaus once the problem size is large enough to saturate the GPU. Provides run-to-run determinism for floating-point reductions on the same device (same compute capability).

### cub::DeviceReduce::Min / Max

```cpp
#include <cub/cub.cuh>

int  num_items = 1 << 20;
int  *d_in;   // e.g., [8, 6, 7, 5, 3, 0, 9, ...]
int  *d_out;  // single output

void  *d_temp_storage = nullptr;
size_t temp_storage_bytes = 0;
cub::DeviceReduce::Min(
    d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
cudaMalloc(&d_temp_storage, temp_storage_bytes);
cub::DeviceReduce::Min(
    d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
// d_out <-- minimum element
```

`cub::DeviceReduce::Max` has the identical API, using `>` comparison. Initial values default to `numeric_limits<T>::max()` (for Min) and `numeric_limits<T>::lowest()` (for Max).

### cub::DeviceReduce::ArgMin / ArgMax

Returns both the extreme value and its index as a `cub::KeyValuePair<int, T>`.

```cpp
#include <cub/cub.cuh>

int   num_items = 1 << 20;
float *d_in;
cub::KeyValuePair<int, float> *d_argmin;  // output: {index, value}

void  *d_temp_storage = nullptr;
size_t temp_storage_bytes = 0;
cub::DeviceReduce::ArgMin(
    d_temp_storage, temp_storage_bytes, d_in, d_argmin, num_items);
cudaMalloc(&d_temp_storage, temp_storage_bytes);
cub::DeviceReduce::ArgMin(
    d_temp_storage, temp_storage_bytes, d_in, d_argmin, num_items);
```

### cub::DeviceReduce::Reduce (custom operator)

For non-standard reduction operators (e.g., product, bitwise OR, custom user-defined functor):

```cpp
#include <cub/cub.cuh>

struct CustomMin {
    template <typename T>
    __device__ __forceinline__
    T operator()(const T &a, const T &b) const {
        return (b < a) ? b : a;
    }
};

int       num_items = 1 << 20;
int       *d_in, *d_out;
CustomMin min_op;
int       init = INT_MAX;

void  *d_temp_storage = nullptr;
size_t temp_storage_bytes = 0;
cub::DeviceReduce::Reduce(
    d_temp_storage, temp_storage_bytes,
    d_in, d_out, num_items, min_op, init);
cudaMalloc(&d_temp_storage, temp_storage_bytes);
cub::DeviceReduce::Reduce(
    d_temp_storage, temp_storage_bytes,
    d_in, d_out, num_items, min_op, init);
```

**Constraint**: the binary operator must be commutative and associative (or pseudo-associative for floating-point types).

**Supported types**: any type for which the binary operator is defined. CUB is templated and works with `int`, `float`, `double`, `__half`, `__nv_bfloat16`, and user-defined structs.

---

## 3. Thrust reduction (thrust::reduce)

**When to use**: simpler API than CUB (no manual temp storage), suitable for quick prototyping or when the caller is already using Thrust containers. Thrust internally dispatches to CUB for the CUDA backend, so performance is comparable.

```cpp
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

// Sum of all elements
thrust::device_vector<double> d_vec(1 << 20);
// ... fill d_vec ...
double total = thrust::reduce(d_vec.begin(), d_vec.end());

// Sum with explicit init and operator
double total2 = thrust::reduce(
    d_vec.begin(), d_vec.end(),
    0.0,                     // init
    thrust::plus<double>()   // binary op
);
```

**Shape range**: flat 1-D range defined by iterators. For multi-dimensional reductions, the caller must flatten the range or iterate over slices.

**When to prefer CUB over Thrust**:
- When explicit stream control is needed (CUB accepts a `cudaStream_t`).
- When avoiding Thrust header bloat in compilation.
- When the two-pass temp-storage pattern is acceptable and the caller wants maximum control.

---

## Decision summary

| Scenario | Recommended library | API |
|---|---|---|
| PyTorch tensor, any shape, standard op | PyTorch | `torch.sum`, `torch.mean`, `torch.max`, etc. |
| CUDA C++, flat buffer, standard op (sum/min/max) | CUB | `cub::DeviceReduce::Sum` / `Min` / `Max` |
| CUDA C++, flat buffer, custom op | CUB | `cub::DeviceReduce::Reduce` |
| CUDA C++, flat buffer, argmin/argmax | CUB | `cub::DeviceReduce::ArgMin` / `ArgMax` |
| Quick prototype, Thrust containers already in use | Thrust | `thrust::reduce` |
| Fused reduction (reduce + elementwise in one kernel) | Custom kernel | See INDEX.md Step 1 |
| Per-row/per-column reduction of 2-D tensor | Custom kernel (or PyTorch) | See INDEX.md Step 1 |
