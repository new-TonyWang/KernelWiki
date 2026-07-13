---
id: skill-triton-ascend-scatter-duplicate-semantics
title: "Triton Ascend Scatter Duplicate Index Semantics"
type: skill
vendor: ascend
tags:
- triton-ascend
evidence_level: measured
applies_to:
- ascend910b
source:
- path: local
  anchor: scatter duplicate semantics precision probe
architectures:
- ascend910b
languages:
- triton-ascend
aliases:
- scatter duplicate
- duplicate indices
- duplicate index
- index_put duplicate
- scatter duplicate index
- accumulate false duplicate
- accumulate=True
- accumulate=False
- undefined reference
- nondeterministic reference
- nondeterministic index_put
- atomic_add scatter
- atomic_max winner
- last writer wins
- order agnostic gate
- unique indices
- duplicate targets
- scatter precision cannot align
- write conflict
---
# Triton Ascend Scatter / index_put Duplicate Index Semantics

## Search Keywords

Simple search terms: scatter duplicate, duplicate indices, duplicate index, index_put duplicate, duplicate targets, accumulate false duplicate, accumulate=True, accumulate=False, undefined reference, nondeterministic reference, nondeterministic index_put, atomic_add scatter, atomic_max winner, last writer wins, order agnostic gate, unique indices, write conflict.

## 1. Key conclusion

The difficulty of `scatter` / `index_put` is determined mostly by `accumulate`:

- `accumulate=True`: duplicate indices mean scatter-add; the semantics are defined and `tl.atomic_add` is appropriate.
- `accumulate=False` with unique indices: semantics are defined; this is a plain write/copy.
- **`accumulate=False` with duplicate indices: the reference semantics are undefined, so there is no unique correct answer.**

If a verifier compares duplicate-heavy `accumulate=False` cases elementwise, the failure is usually not Triton precision, not a missing fp16/bf16 patch, and not lost Ascend atomic updates. The torch_npu / PyTorch reference itself has no stable ground truth for that input regime.

## 2. Why precision cannot be aligned

This failure is often misdiagnosed as a fp16/bf16 precision issue, but it is not numerical precision.

For `accumulate=False`:

```python
out[index[i]] = values[i]
```

If multiple `i` values write the same `index[i]`, the implementation must decide which source is the final writer. PyTorch defines `accumulate=False` with duplicate indices as undefined. torch_npu is also observed to be run-to-run nondeterministic on larger duplicate-heavy inputs.

Therefore precision cannot be aligned because:

1. **There is no stable reference**: identical inputs can produce different torch_npu outputs across runs.
2. **This is not an allclose tolerance issue**: the output may come from a different source value, not from a 1-ULP rounding difference.
3. **This is not a NaN/Inf mask issue**: values are usually finite and normal, just selected from a different source position.
4. **This is not an atomic race by default**: controlled probes show Ascend `tl.atomic_add` and `tl.atomic_max` are stable on colliding addresses.
5. **Any deterministic Triton policy picks one tie-break**, for example largest source index wins, but cannot guarantee matching an undefined torch_npu execution order.

Signature: fp32 also fails, and mismatches look like values from a different member of the same collision set rather than rounding-boundary errors.

## 3. `accumulate=True`: scatter-add is solvable

`accumulate=True` sums all sources that hit the same target:

```python
out[index[i]] += values[i]
```

Use `tl.atomic_add`:

```python
@triton.jit
def scatter_add_kernel(index_ptr, value_ptr, out_ptr, L, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < L
    tgt = tl.load(index_ptr + offs, mask=mask, other=0)
    val = tl.load(value_ptr + offs, mask=mask, other=0.0)
    tl.atomic_add(out_ptr + tgt, val, mask=mask)
```

Floating-point scatter-add may accumulate in a different order from torch. With many collisions this can produce around 1 ULP of difference. Treat that as normal floating-point accumulation-order error and compare within the real evaluation tolerance; do not confuse it with lost atomic updates.

## 4. `accumulate=False` with unique indices is solvable

If every target is written at most once, `accumulate=False` is just:

```python
out[index[i]] = values[i]
```

There is no conflict and no tie-break, so Triton-Ascend can reproduce fp16/bf16/fp32 writes exactly.

## 5. `accumulate=False` with duplicate indices is not elementwise-alignable

When duplicate indices exist and the spec does not define a conflict rule, there is no unique output. Do not spend optimization rounds tuning `BLOCK`, changing atomics, or adding precision patches. The correct fix is to change the evaluation contract:

- generate **unique indices** for `accumulate=False`; or
- use an **order-agnostic gate** that accepts any source value that targets the same output position; or
- define an explicit tie-break such as “largest source position wins”.

If you own the op spec, a deterministic two-pass implementation can implement a chosen tie-break. It is correct for that spec, but it is not guaranteed to reproduce undefined torch_npu behavior.

## 6. Deterministic last-writer two-pass, when the spec defines it

```python
# Pass 0: winner[t] = largest source position i that writes target t; untouched = -1.
@triton.jit
def _argmax_src(index_ptr, winner_ptr, L, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < L
    tgt = tl.load(index_ptr + offs, mask=mask, other=0)
    tl.atomic_max(winner_ptr + tgt, offs.to(tl.int32), mask=mask)

# Pass 1: out[t] = values[winner[t]] if written else x[t].
@triton.jit
def _resolve(x_ptr, values_ptr, winner_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    w = tl.load(winner_ptr + offs, mask=mask, other=-1)
    written = w >= 0
    src = tl.where(written, w, 0)
    v = tl.load(values_ptr + src, mask=mask & written, other=0.0)
    xv = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, tl.where(written, v, xv), mask=mask)
```

Important details:

- Initialize `winner` to `-1`, not `0`; otherwise target 0 looks as if source 0 wrote it.
- `atomic_max` is over the **source position**, not the value.
- Untouched targets keep their original `x[t]`; do not zero them.
- This is a deterministic custom semantic, not a universal reproduction of an undefined reference.

## 7. Checklist

1. Branch on `accumulate`; do not use one semantic for all cases.
2. For `accumulate=True`, use `tl.atomic_add` and tolerate normal accumulation-order differences.
3. For `accumulate=False`, first check whether indices are unique.
4. If duplicate + `accumulate=False` fails and fp32 also fails, classify it as undefined reference behavior, not precision.
5. Do not assume Ascend atomics are racing; controlled colliding atomics are stable.
6. If elementwise comparison is required, ask for unique-index input generation, an order-agnostic gate, or a defined tie-break.
