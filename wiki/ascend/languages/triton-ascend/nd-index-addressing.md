---
id: skill-triton-ascend-nd-index-addressing
title: "Triton Ascend N-D Index Addressing"
type: skill
vendor: ascend
tags:
- triton-ascend
evidence_level: measured
applies_to:
- ascend910b
source:
- path: local
  anchor: nd-index-addressing gather precision probe
architectures:
- ascend910b
languages:
- triton-ascend
aliases:
- N-D gather
- ND gather
- n-dimensional gather
- index_select addressing
- gather dim 0
- rank 3 dim 0
- gather stride
- real stride
- source offset
- index addressing
- multi-dimensional index
- last axis bug
- last dim assumption
- non-contiguous gather
- gather output zero
- torch.gather
- index_select
- wrong address calculation
---
# Triton Ascend N-D Gather / Index Addressing

## Search Keywords

Simple search terms: N-D gather, ND gather, n-dimensional gather, torch.gather, index_select, gather dim 0, rank 3 dim 0, gather stride, real stride, source offset, index addressing, multi-dimensional index, last axis bug, last dim assumption, non-contiguous gather, gather output zero, wrong address calculation.

## 1. Core problem

The most common bug in N-D gather / index kernels is assuming that the indexed axis is the last dimension, or that the input is always 2-D.

Such kernels often flatten the tensor as:

```text
[rows, K]
```

and gather only along the inner axis. This passes many `dim=-1` or 2-D cases, but reads the wrong addresses for higher-rank tensors, especially `rank=3, dim=0`. A typical symptom is near-zero or otherwise wrong output only for a small subset of shapes.

A measured example: Gather passed 44/47 cases; the three failures were exactly `rank=3, dim=0`, which indicates that the kernel only implemented last-dimension / 2-D addressing.

## 2. Correct semantics

For:

```python
out = torch.gather(x, dim, index)
```

let the output coordinate be:

```text
c = (c0, c1, ..., c_{rank-1})
```

The source coordinate is:

```text
src_coord = c
src_coord[dim] = index[c]
```

Only the `dim` axis is replaced by `index[c]`; all other coordinates are unchanged. The input address must be computed from the real strides of `x`:

```text
src_offset = Σ_a src_coord[a] * x_stride[a]
```

Use the **real stride of `x`**, not a contiguous-last-axis assumption and not a hard-coded `row * K + col` formula.

For:

```python
out = torch.index_select(x, dim, index)
```

`index` is 1-D and the semantics are:

```python
out[..., i, ...] = x[..., index[i], ...]
```

The output coordinate on axis `dim` maps to `index[i]`; all other axes stay the same.

## 3. Common wrong patterns

### 3.1 Assuming gather is always on the last dimension

Wrong idea:

```text
offset = row * K + index[col]
```

This only works for the last dimension. If the real dimension is `dim=0`, the correct address looks like:

```text
offset = index[c] * stride0 + c1 * stride1 + c2 * stride2
```

not one where `index[c]` is multiplied by the last-axis stride.

### 3.2 Supporting only 2-D

Decomposing only:

```text
row = linear // K
col = linear % K
```

cannot represent arbitrary axes for `rank=3/4/...`. An N-D kernel should decode the output linear id into multi-dimensional coordinates, or generate a rank-specialized formula that explicitly handles each dimension.

### 3.3 Ignoring real strides

If the input may be a view, transpose, or non-contiguous tensor, contiguous assumptions read the wrong memory. Pass and use `x.stride()`. Even when a benchmark mostly uses contiguous inputs, using real strides avoids hidden bugs.

## 4. Implementation template: rank-3 gather

This example shows arbitrary `dim` address calculation for `rank=3`. In production, dispatch to rank-specialized kernels or use constexpr branches to generate the target formula.

```python
@triton.jit
def gather_rank3_kernel(
    x_ptr, index_ptr, out_ptr,
    S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,
    XSTR0: tl.constexpr, XSTR1: tl.constexpr, XSTR2: tl.constexpr,
    ISTR0: tl.constexpr, ISTR1: tl.constexpr, ISTR2: tl.constexpr,
    OSTR0: tl.constexpr, OSTR1: tl.constexpr, OSTR2: tl.constexpr,
    DIM: tl.constexpr,
    INDEX_DIM_SIZE: tl.constexpr,
    X_DIM_SIZE: tl.constexpr,
    NUMEL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < NUMEL

    c2 = offs % S2
    t = offs // S2
    c1 = t % S1
    c0 = t // S1

    idx_off = c0 * ISTR0 + c1 * ISTR1 + c2 * ISTR2
    g = tl.load(index_ptr + idx_off, mask=mask, other=0)
    g = tl.maximum(0, tl.minimum(g, X_DIM_SIZE - 1))

    s0 = tl.where(DIM == 0, g, c0)
    s1 = tl.where(DIM == 1, g, c1)
    s2 = tl.where(DIM == 2, g, c2)

    x_off = s0 * XSTR0 + s1 * XSTR1 + s2 * XSTR2
    out_off = c0 * OSTR0 + c1 * OSTR1 + c2 * OSTR2

    v = tl.load(x_ptr + x_off, mask=mask, other=0.0)
    tl.store(out_ptr + out_off, v, mask=mask)
```

Notes:

- `S0/S1/S2` are output shape dimensions; for `torch.gather` they usually match `index.shape`.
- `XSTR*`, `ISTR*`, and `OSTR*` are element strides.
- `DIM` is the actual gathered axis.
- `X_DIM_SIZE` protects gather index bounds and avoids device OOB faults.
- If exact PyTorch out-of-bounds error semantics are required, validate indices on host or in the evaluator; the kernel should still avoid OOB memory access.

## 5. `index_select` address calculation

For `index_select`, `index` is 1-D and the output shape is the input shape except that axis `dim` has length `len(index)`. For output coordinate `c`:

```text
src_coord[a] = c[a]                 if a != dim
src_coord[dim] = index[c[dim]]      if a == dim
```

The only difference from `gather` is that the index load address is:

```text
index_offset = c[dim]
```

The source address is still:

```text
src_offset = Σ src_coord[a] * x_stride[a]
```

## 6. Validation checklist

1. Pass or constexpr-specialize `rank`, `dim`, shape, and strides.
2. Decode each output element into N-D coordinates.
3. Replace only the `dim` coordinate; keep all other axes unchanged.
4. Use `x_stride[a]` to compute `src_offset`.
5. Protect gather indices to avoid OOB device faults.
6. Always test `rank=3, dim=0`; testing only `dim=-1` is not meaningful.
7. Cover the full `(rank, dim)` grid because evaluators often enumerate these combinations.

## 7. Relationship to duplicate scatter semantics

N-D addressing solves only “which address is read or written.” If scatter / `index_put` has duplicate targets, duplicate-index semantics must be handled separately:

- `accumulate=True`: scatter-add, use atomic add.
- `accumulate=False + duplicate index`: the reference may be undefined and cannot be fixed by an address formula or precision patch.

See [scatter / index_put duplicate index semantics](scatter-duplicate-semantics.md).
