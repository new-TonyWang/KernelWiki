---
title: Scan (Cumulative) Pattern -- Library Fallback Paths
pattern_class: cuda-core
op: scan-cumulative
status: draft
source:
- path: spec
  anchor: Reference
id: routing-scan-cumulative-library-fallback
type: operator-routing
vendor: nvidia
operator: scan-cumulative
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA API References/cub/cub.md
  anchor: L28474-L28508
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA API References/cub/cub.md
  anchor: L28553-L28578
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA API References/cub/cub.md
  anchor: L29201-L29227
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA API References/cub/cub.md
  anchor: L28764-L28802
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA API References/cub/cub.md
  anchor: L29340-L29381
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA API References/cccl/cccl.md
  anchor: L99812-L99891
---
# Scan (Cumulative) -- Library Fallback Paths

Before writing a custom scan kernel, use one of the following library paths. These are production-quality, GPU-optimized implementations that handle edge cases (non-power-of-two sizes, arbitrary dtypes, stream integration) and are well-tested across CUDA toolkit releases.

CUB's DeviceScan internally uses the **decoupled look-back** algorithm, which performs only a single pass through the data with ~2n data movement (n reads + n writes), achieving near-memcpy throughput for large arrays. Writing a custom kernel that beats CUB on flat scans is extremely difficult.

---

## 1. PyTorch cumulative ops

**When to use**: the caller's workflow is PyTorch-based and the scan is a standard cumulative operation on a `torch.Tensor` residing on a CUDA device.

### torch.cumsum

Computes the cumulative sum (inclusive prefix sum) along a dimension.

```python
import torch

# 1-D inclusive prefix sum
x = torch.tensor([1, 2, 3, 4, 5], device="cuda", dtype=torch.float32)
y = torch.cumsum(x, dim=0)       # [1, 3, 6, 10, 15]

# 2-D cumulative sum along dim=1 (per-row scan)
x = torch.randn(128, 4096, device="cuda", dtype=torch.float32)
row_cumsums = torch.cumsum(x, dim=1)  # shape: (128, 4096)

# 2-D cumulative sum along dim=0 (per-column scan)
col_cumsums = torch.cumsum(x, dim=0)  # shape: (128, 4096)
```

**Shape range**: any tensor shape that fits in GPU memory. PyTorch internally dispatches to efficient CUDA kernels. For very small tensors (<1024 elements), kernel launch overhead may dominate.

**Supported dtypes**: float16, bfloat16, float32, float64, int32, int64, and complex types.

**Note**: `torch.cumsum` always computes an **inclusive** scan. PyTorch does not provide a built-in exclusive scan; to obtain one, compute `torch.cumsum` and subtract the input, or prepend a zero and drop the last element.

### torch.cumprod

Computes the cumulative product along a dimension.

```python
x = torch.tensor([1, 2, 3, 4], device="cuda", dtype=torch.float32)
y = torch.cumprod(x, dim=0)      # [1, 2, 6, 24]
```

**Caution**: cumulative product is prone to overflow (for large values) or underflow (for values < 1). Use float64 if precision is critical.

### torch.cummax / torch.cummin

Computes the cumulative maximum or minimum along a dimension. Returns both the cumulative values and the indices of the running extrema.

```python
x = torch.tensor([3, 1, 4, 1, 5, 9], device="cuda", dtype=torch.float32)
values, indices = torch.cummax(x, dim=0)
# values:  [3, 3, 4, 4, 5, 9]
# indices: [0, 0, 2, 2, 4, 5]
```

---

## 2. CUB device-wide scan (cub::DeviceScan)

**When to use**: writing a standalone CUDA/C++ program (not PyTorch) and the scan is over a flat, contiguous 1-D buffer with a standard or custom binary operator. CUB provides the highest-performance single-kernel scan for this case, using the decoupled look-back algorithm.

**Include**: `#include <cub/cub.cuh>` or `#include <cub/device/device_scan.cuh>`

**Common pattern**: CUB device APIs use a two-pass temp-storage idiom:
1. Call with `d_temp_storage = nullptr` to query required temp size.
2. Allocate temp storage.
3. Call again to run the scan.

### cub::DeviceScan::InclusiveSum

Computes the inclusive prefix sum. Output[i] = Input[0] + Input[1] + ... + Input[i].

```cpp
#include <cub/cub.cuh>

int   num_items = 1 << 20;  // 1M elements
int   *d_in;                // e.g., [8, 6, 7, 5, 3, 0, 9]
int   *d_out;               // output, same size as input

// Step 1: query temp storage
void  *d_temp_storage = nullptr;
size_t temp_storage_bytes = 0;
cub::DeviceScan::InclusiveSum(
    d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);

// Step 2: allocate
cudaMalloc(&d_temp_storage, temp_storage_bytes);

// Step 3: run
cub::DeviceScan::InclusiveSum(
    d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);

// d_out <-- [8, 14, 21, 26, 29, 29, 38]  (for the 7-element example)
```

**In-place variant**: when `d_in == d_out`, the scan operates in-place.

### cub::DeviceScan::ExclusiveSum

Computes the exclusive prefix sum. Output[i] = Input[0] + Input[1] + ... + Input[i-1]. The initial value (identity) is 0.

```cpp
#include <cub/cub.cuh>

int   num_items = 7;
int   *d_in;     // e.g., [8, 6, 7, 5, 3, 0, 9]
int   *d_out;

void  *d_temp_storage = nullptr;
size_t temp_storage_bytes = 0;
cub::DeviceScan::ExclusiveSum(
    d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
cudaMalloc(&d_temp_storage, temp_storage_bytes);
cub::DeviceScan::ExclusiveSum(
    d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);

// d_out <-- [0, 8, 14, 21, 26, 29, 29]
```

### cub::DeviceScan::InclusiveScan (custom operator)

For non-standard scan operators (e.g., max, min, bitwise OR):

```cpp
#include <cub/cub.cuh>

struct CustomMin {
    template <typename T>
    __host__ __device__ __forceinline__
    T operator()(const T &a, const T &b) const {
        return (b < a) ? b : a;
    }
};

int       num_items = 7;
int       *d_in;    // e.g., [8, 6, 7, 5, 3, 0, 9]
int       *d_out;
CustomMin min_op;

void  *d_temp_storage = nullptr;
size_t temp_storage_bytes = 0;
cub::DeviceScan::InclusiveScan(
    d_temp_storage, temp_storage_bytes,
    d_in, d_out, min_op, num_items);
cudaMalloc(&d_temp_storage, temp_storage_bytes);
cub::DeviceScan::InclusiveScan(
    d_temp_storage, temp_storage_bytes,
    d_in, d_out, min_op, num_items);

// d_out <-- [8, 6, 6, 5, 3, 0, 0]
```

### cub::DeviceScan::ExclusiveScan (custom operator)

Exclusive scan with a user-defined operator and initial value:

```cpp
#include <cub/cub.cuh>
#include <cuda/std/climits>

CustomMin min_op;
int       init = INT_MAX;

void  *d_temp_storage = nullptr;
size_t temp_storage_bytes = 0;
cub::DeviceScan::ExclusiveScan(
    d_temp_storage, temp_storage_bytes,
    d_in, d_out, min_op, init, num_items);
cudaMalloc(&d_temp_storage, temp_storage_bytes);
cub::DeviceScan::ExclusiveScan(
    d_temp_storage, temp_storage_bytes,
    d_in, d_out, min_op, init, num_items);

// d_out <-- [INT_MAX, 8, 6, 6, 5, 3, 0]  (for the 7-element example)
```

**Constraint**: the binary operator must be associative (but not necessarily commutative -- CUB supports non-commutative operators as of CCCL 2.2.0). For floating-point types, results may vary between runs due to pseudo-associativity.

**Supported types**: any type for which the binary operator is defined. CUB is templated and works with `int`, `float`, `double`, `__half`, `__nv_bfloat16`, and user-defined structs.

---

## 3. Thrust scan (thrust::inclusive_scan / thrust::exclusive_scan)

**When to use**: simpler API than CUB (no manual temp storage), suitable for quick prototyping or when the caller is already using Thrust containers. Thrust internally dispatches to CUB for the CUDA backend, so performance is comparable.

```cpp
#include <thrust/device_vector.h>
#include <thrust/scan.h>

// Inclusive prefix sum
thrust::device_vector<float> d_in(1 << 20);
thrust::device_vector<float> d_out(1 << 20);
// ... fill d_in ...
thrust::inclusive_scan(d_in.begin(), d_in.end(), d_out.begin());

// Exclusive prefix sum (init = 0 by default)
thrust::exclusive_scan(d_in.begin(), d_in.end(), d_out.begin());

// Exclusive scan with custom init and operator
thrust::exclusive_scan(
    d_in.begin(), d_in.end(), d_out.begin(),
    0.0f,                        // init
    thrust::plus<float>()        // binary op
);

// In-place scan
thrust::inclusive_scan(d_in.begin(), d_in.end(), d_in.begin());
```

**Shape range**: flat 1-D range defined by iterators. For multi-dimensional scans, the caller must flatten the range or iterate over slices.

**When to prefer CUB over Thrust**:
- When explicit stream control is needed (CUB accepts a `cudaStream_t`).
- When avoiding Thrust header bloat in compilation.
- When the two-pass temp-storage pattern is acceptable and the caller wants maximum control.

---

## Decision summary

| Scenario | Recommended library | API |
|---|---|---|
| PyTorch tensor, any shape, inclusive cumulative sum | PyTorch | `torch.cumsum` |
| PyTorch tensor, cumulative product | PyTorch | `torch.cumprod` |
| PyTorch tensor, cumulative max/min | PyTorch | `torch.cummax` / `torch.cummin` |
| CUDA C++, flat buffer, inclusive prefix sum | CUB | `cub::DeviceScan::InclusiveSum` |
| CUDA C++, flat buffer, exclusive prefix sum | CUB | `cub::DeviceScan::ExclusiveSum` |
| CUDA C++, flat buffer, inclusive custom op | CUB | `cub::DeviceScan::InclusiveScan` |
| CUDA C++, flat buffer, exclusive custom op + init | CUB | `cub::DeviceScan::ExclusiveScan` |
| Quick prototype, Thrust containers in use | Thrust | `thrust::inclusive_scan` / `thrust::exclusive_scan` |
| Fused scan (scan + elementwise in one kernel) | Custom kernel | See INDEX.md Step 1 |
| Segmented scan with non-standard boundaries | Custom kernel | See INDEX.md Step 1 |
