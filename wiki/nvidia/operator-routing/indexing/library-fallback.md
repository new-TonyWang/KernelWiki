---
title: Indexing Pattern -- Library Fallback Paths
pattern_class: cuda-core
op: indexing
status: draft
source:
- path: spec
  anchor: Reference
id: routing-indexing-library-fallback
type: operator-routing
vendor: nvidia
operator: indexing
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA API References/cccl/cccl.md
  anchor: L101875-L101930
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA API References/cccl/cccl.md
  anchor: L112950-L113010
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA API References/cccl/cccl.md
  anchor: L113098-L113200
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA API References/cub/cub.md
  anchor: L22478-L22555
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
# Indexing -- Library Fallback Paths

Before writing a custom indexing kernel, use one of the following library paths. These are production-quality, GPU-optimized implementations that handle edge cases and are well-tested.

---

## 1. PyTorch indexing ops

**When to use**: the caller's workflow is PyTorch-based and the indexing operation is a standard operation on `torch.Tensor` residing on a CUDA device.

### torch.gather

Gathers values along an axis specified by `dim`.

```python
import torch

# Gather along dim=1: select one element per row
src = torch.randn(4, 5, device="cuda", dtype=torch.float32)
index = torch.tensor([[0, 2], [1, 3], [4, 0], [2, 1]],
                     device="cuda", dtype=torch.int64)  # shape (4, 2)
out = torch.gather(src, dim=1, index=index)              # shape (4, 2)
# out[i][j] = src[i][index[i][j]]
```

**Shape constraint**: `index` must have the same number of dimensions as `src`. Along the gather dimension, `index` can have any size; along all other dimensions, `index.size(d)` must equal `src.size(d)`.

**Supported dtypes**: all standard numeric types (float16, bfloat16, float32, float64, int32, int64, etc.).

### torch.scatter / torch.scatter\_add

Writes values into a destination tensor at positions given by `index`.

```python
import torch

# Scatter: write src values into dst at specified indices
dst = torch.zeros(4, 5, device="cuda", dtype=torch.float32)
index = torch.tensor([[0, 2], [1, 3], [4, 0], [2, 1]],
                     device="cuda", dtype=torch.int64)
src = torch.ones(4, 2, device="cuda", dtype=torch.float32)
dst.scatter_(dim=1, index=index, src=src)
# dst[i][index[i][j]] = src[i][j]

# Scatter-add: accumulate values (handles index collisions)
dst2 = torch.zeros(4, 5, device="cuda", dtype=torch.float32)
dst2.scatter_add_(dim=1, index=index, src=src)
# dst2[i][index[i][j]] += src[i][j]
```

**Conflict handling**: `scatter_` with duplicate indices produces undefined behavior (last write wins). Use `scatter_add_` when multiple source elements may map to the same destination index.

### torch.index\_select

Selects entire slices along a dimension using a 1-D index tensor.

```python
import torch

src = torch.randn(1000, 256, device="cuda", dtype=torch.float32)
index = torch.tensor([3, 7, 42, 999], device="cuda", dtype=torch.int64)
out = torch.index_select(src, dim=0, index=index)  # shape (4, 256)
# out[j] = src[index[j]]  (selects 4 full rows)
```

**Performance note**: when `dim=0` and the tensor is contiguous, this is effectively a batched memcpy of selected rows. PyTorch dispatches an efficient kernel for this case.

### torch.topk

Returns the `k` largest (or smallest) elements along a dimension.

```python
import torch

x = torch.randn(128, 4096, device="cuda", dtype=torch.float32)
values, indices = torch.topk(x, k=10, dim=1)
# values shape: (128, 10), indices shape: (128, 10)

# Smallest k:
values_sm, indices_sm = torch.topk(x, k=10, dim=1, largest=False)
```

**Performance note**: for large `dim` sizes (>10K) with small `k`, `torch.topk` uses a radix-select algorithm internally. For very small tensors, the kernel launch overhead may dominate.

### torch.nn.functional.one\_hot

Converts class indices to one-hot encoded vectors.

```python
import torch
import torch.nn.functional as F

labels = torch.tensor([0, 2, 1, 4], device="cuda", dtype=torch.int64)
one_hot = F.one_hot(labels, num_classes=5)  # shape (4, 5)
# one_hot = [[1,0,0,0,0], [0,0,1,0,0], [0,1,0,0,0], [0,0,0,0,1]]
```

### torch.nn.functional.embedding

Looks up rows from an embedding table. Equivalent to `index_select` on dim=0 of the weight matrix.

```python
import torch
import torch.nn.functional as F

weight = torch.randn(10000, 256, device="cuda", dtype=torch.float32)
indices = torch.tensor([1, 42, 999, 7], device="cuda", dtype=torch.int64)
out = F.embedding(indices, weight)  # shape (4, 256)
# Equivalent to: weight[indices]
```

---

## 2. Thrust gather / scatter

**When to use**: writing a standalone CUDA/C++ program (not PyTorch) and the operation is a straightforward gather or scatter over contiguous buffers.

### thrust::gather

Copies elements from a source array into a destination range according to an index map. Semantics: `result[i] = input[map[i]]`.

```cpp
#include <thrust/gather.h>
#include <thrust/device_vector.h>

int N_map = 1 << 16;   // number of gather indices
int N_src = 1 << 20;   // source array size

thrust::device_vector<float> d_src(N_src);
thrust::device_vector<int>   d_map(N_map);
thrust::device_vector<float> d_out(N_map);
// ... fill d_src and d_map ...

thrust::gather(d_map.begin(), d_map.end(),
               d_src.begin(),
               d_out.begin());
// d_out[i] = d_src[d_map[i]]
```

### thrust::gather\_if

Conditional gather: only gathers elements where the stencil predicate is true.

```cpp
#include <thrust/gather.h>
#include <thrust/device_vector.h>

thrust::device_vector<float> d_src(N_src);
thrust::device_vector<int>   d_map(N_map);
thrust::device_vector<int>   d_stencil(N_map);  // 0 or 1
thrust::device_vector<float> d_out(N_map, 0.0f);

thrust::gather_if(d_map.begin(), d_map.end(),
                  d_stencil.begin(),
                  d_src.begin(),
                  d_out.begin());
// d_out[i] = d_src[d_map[i]] only where d_stencil[i] != 0
```

### thrust::scatter

Copies elements from a source range into a destination array at positions specified by an index map. Semantics: `result[map[i]] = src[i]`.

**Important**: if the same index appears more than once in the map, the result is undefined (data race).

```cpp
#include <thrust/scatter.h>
#include <thrust/device_vector.h>

int N = 1 << 16;

thrust::device_vector<float> d_values(N);
thrust::device_vector<int>   d_map(N);
thrust::device_vector<float> d_out(1 << 20, 0.0f);
// ... fill d_values and d_map ...

thrust::scatter(d_values.begin(), d_values.end(),
                d_map.begin(),
                d_out.begin());
// d_out[d_map[i]] = d_values[i]
```

### thrust::scatter\_if

Conditional scatter: only scatters elements where the stencil predicate is true.

```cpp
#include <thrust/scatter.h>
#include <thrust/device_vector.h>

thrust::device_vector<float> d_values(N);
thrust::device_vector<int>   d_map(N);
thrust::device_vector<int>   d_stencil(N);  // 0 or 1
thrust::device_vector<float> d_out(1 << 20, 0.0f);

thrust::scatter_if(d_values.begin(), d_values.end(),
                   d_map.begin(),
                   d_stencil.begin(),
                   d_out.begin());
// d_out[d_map[i]] = d_values[i] only where d_stencil[i] != 0
```

**When to prefer custom over Thrust scatter**: Thrust scatter does NOT support conflict resolution. If multiple elements map to the same index and must be summed (scatter-add) or compared (scatter-max), a custom kernel with atomics is required.

---

## 3. CUB radix sort for topk

**When to use**: topk over a large array when sorting the full array is acceptable. Sort key-value pairs, then take the first (or last) k entries.

```cpp
#include <cub/cub.cuh>

int   N = 1 << 20;
int   k = 64;

float *d_keys_in, *d_keys_out;   // values to sort
int   *d_vals_in, *d_vals_out;   // original indices
// Allocate and fill d_keys_in with values, d_vals_in with 0..N-1

// Step 1: query temp storage
void  *d_temp_storage = nullptr;
size_t temp_storage_bytes = 0;
cub::DeviceRadixSort::SortPairsDescending(
    d_temp_storage, temp_storage_bytes,
    d_keys_in, d_keys_out,
    d_vals_in, d_vals_out,
    N);

// Step 2: allocate
cudaMalloc(&d_temp_storage, temp_storage_bytes);

// Step 3: sort (descending to get top-k at the front)
cub::DeviceRadixSort::SortPairsDescending(
    d_temp_storage, temp_storage_bytes,
    d_keys_in, d_keys_out,
    d_vals_in, d_vals_out,
    N);

// The first k entries of d_keys_out / d_vals_out are the top-k.
```

**Trade-off**: full radix sort is O(N) but touches all N elements. For very large N with tiny k, a custom partial-select kernel (bucket select or radix select that terminates early) can be faster. However, cub::DeviceRadixSort is well-optimized and often sufficient.

---

## Decision summary

| Scenario | Recommended library | API |
|---|---|---|
| PyTorch tensor, gather along axis | PyTorch | `torch.gather` |
| PyTorch tensor, scatter (no conflicts) | PyTorch | `torch.scatter_` |
| PyTorch tensor, scatter-add (conflicts) | PyTorch | `torch.scatter_add_` |
| PyTorch tensor, select full rows/slices | PyTorch | `torch.index_select` |
| PyTorch tensor, top-k elements | PyTorch | `torch.topk` |
| PyTorch tensor, one-hot encoding | PyTorch | `F.one_hot` |
| PyTorch tensor, embedding lookup | PyTorch | `F.embedding` |
| CUDA C++, simple gather | Thrust | `thrust::gather` |
| CUDA C++, conditional gather | Thrust | `thrust::gather_if` |
| CUDA C++, simple scatter (no conflicts) | Thrust | `thrust::scatter` |
| CUDA C++, conditional scatter | Thrust | `thrust::scatter_if` |
| CUDA C++, topk via full sort | CUB | `cub::DeviceRadixSort::SortPairsDescending` + truncate |
| Fused gather + compute in one kernel | Custom kernel | See INDEX.md Step 1 |
| Scatter-add / scatter-max (C++) | Custom kernel | See INDEX.md Step 1 (atomics) |
| Topk with k << N, need sub-sort perf | Custom kernel | See INDEX.md Step 1 (radix/bucket select) |
