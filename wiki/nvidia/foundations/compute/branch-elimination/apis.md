---
title: Branch Elimination APIs
status: draft
source:
- path: spec
  anchor: Reference
apis:
- func_name: fmaxf
  namespace: cuda-runtime
  kind: fp32-minmax
  signature: float fmaxf(float x, float y);
  notes: 'Single-precision max. PTX `max.f32` → SASS `FMNMX` (1 instruction). Measured
    on H200 sm_9.0a: 14.02 ns/op in the branchless-patterns probe — 1.56x faster than
    the equivalent `if/else` form which emits FSETP+FSEL (2 SASS ops). The canonical
    branchless-ReLU implementation.'
- func_name: fminf
  namespace: cuda-runtime
  kind: fp32-minmax
  signature: float fminf(float x, float y);
  notes: Single-precision min. PTX `min.f32` → SASS `FMNMX` (with min flag). Same
    1-op SASS lowering as `fmaxf`. Use `fminf(fmaxf(x, lo), hi)` for branchless clamp
    (2 FMNMX ops).
- func_name: fabsf
  namespace: cuda-runtime
  kind: fp32-bitwise
  signature: float fabsf(float x);
  notes: 'Single-precision absolute value. PTX `abs.f32` → SASS `LOP3.LUT` (1 instruction,
    IEEE sign-bit mask). Measured on H200 sm_9.0a: 9.27 ns/op, 2.36x faster than the
    `if (x<0) x=-x` form and 1.51x faster than explicit bit-trick `b & 0x7FFFFFFF`.'
- func_name: copysignf
  namespace: cuda-runtime
  kind: fp32-bitwise
  signature: float copysignf(float x, float y);
  notes: 'Returns `x` with the sign of `y`. Expected to lower to a single LOP3.LUT
    (bit-level sign transplant); not re-measured by the branchless-patterns probe.
    The branchless equivalent of `(y < 0) ? -fabsf(x) : fabsf(x)`.'
- func_name: __fmaxf
  namespace: cuda-runtime
  kind: fp32-minmax-intrinsic
  signature: float __fmaxf(float x, float y);
  notes: Identical PTX / SASS lowering to `fmaxf` on sm_9.0a (both emit `max.f32`
    / `FMNMX`). The `__` prefix marks it as the CUDA-intrinsic form; no behavioural
    difference on modern archs.
- func_name: __fminf
  namespace: cuda-runtime
  kind: fp32-minmax-intrinsic
  signature: float __fminf(float x, float y);
  notes: Identical PTX / SASS lowering to `fminf` on sm_9.0a. See `__fmaxf` above.
- func_name: max
  namespace: cuda-runtime
  kind: template-minmax
  signature: template<typename T> T max(T a, T b);
  notes: 'CUDA overload resolution: `max(float, float)` lowers to `fmaxf` / `FMNMX`;
    `max(int, int)` lowers to PTX `max.s32` / SASS `IMNMX`. Avoid implicit conversions
    that round-trip through fp — call the typed form directly.'
- func_name: min
  namespace: cuda-runtime
  kind: template-minmax
  signature: template<typename T> T min(T a, T b);
  notes: As above but min. `min(int, int)` → `IMNMX`; `min(float, float)` → `FMNMX`.
- func_name: __int_as_float
  namespace: cuda-runtime
  kind: bit-cast
  signature: float __int_as_float(int x);
  notes: Reinterpret-cast int bit pattern as float. Zero-cost at SASS (no instruction
    emitted; register alias). Used in hand-coded bit tricks.
- func_name: __float_as_int
  namespace: cuda-runtime
  kind: bit-cast
  signature: int __float_as_int(float x);
  notes: Reverse of `__int_as_float`. Also zero-cost. Pair of casts lets you do integer
    ops on the fp bit pattern; but measured on H200, explicit bit-trick paths emit
    more SASS ops (LOP3.LUT ×2) than the intrinsic form (LOP3.LUT ×1) — use the intrinsic
    whenever possible.
- func_name: max.f32
  namespace: ptx
  kind: ptx-fp32-max
  signature: max.f32 d, a, b;
  notes: PTX single-precision max. Lowers to SASS `FMNMX` (1 instruction). Emitted
    from `fmaxf` / `__fmaxf` / `max(float,float)`.
- func_name: min.f32
  namespace: ptx
  kind: ptx-fp32-min
  notes: PTX fp32 min. SASS `FMNMX` with min flag.
- func_name: abs.f32
  namespace: ptx
  kind: ptx-fp32-abs
  notes: PTX fp32 absolute value. SASS `LOP3.LUT` (1 instruction, IEEE sign-bit mask).
    Emitted from `fabsf`.
- func_name: selp.f32
  namespace: ptx
  kind: ptx-predicated-select
  signature: selp.f32 d, a, b, p;
  notes: 'Predicated select: `d = p ? a : b`. Emitted from simple-body `if/else` and
    ternary expressions. SASS `FSEL` (1 instruction). This is why C-source ''branches''
    in simple contexts are already branchless at SASS on H200 sm_9.0a.'
- func_name: selp.b32
  namespace: ptx
  kind: ptx-predicated-select-integer
  notes: Integer variant of `selp`. Same SASS lowering (`SEL` / `FSEL` family).
- func_name: FMNMX
  namespace: sass
  kind: sass-fp32-minmax
  notes: H200 SASS opcode for fp min / max. Encodes comparison + select in one instruction.
    Emitted from PTX `max.f32` / `min.f32`.
- func_name: FSEL
  namespace: sass
  kind: sass-predicated-select
  notes: H200 SASS predicated-select. Emitted from PTX `selp` — the hardware instruction
    behind `if/else` and ternary rewrites for simple bodies.
- func_name: FSETP.GEU.AND
  namespace: sass
  kind: sass-fp-setp
  notes: H200 SASS floating-point compare (producing a predicate). Paired with `FSEL`
    in the 2-op emission from `if (x<0) y = 0;` ReLU form. Not needed when the intrinsic
    `fmaxf` is used (that's a single FMNMX).
- func_name: LOP3.LUT
  namespace: sass
  kind: sass-3-input-logic
  notes: H200 SASS 3-input bitwise LUT — can compute any 3-input boolean with an 8-bit
    truth table. Used by `fabsf` (1 LOP3) for IEEE sign-bit mask; also emitted from
    explicit bit-tricks (typically ×2 because the round-trip through int registers
    prevents fusion).
id: api-branch-elimination-ref
type: api-definition
vendor: nvidia
func_name: Branch Elimination APIs
namespace: runtime
header: cuda_runtime.h
signature: See documentation
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L25650-L25760
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/parallel-thread-execution/cuda_parallel-thread-execution_index.html.md
  anchor: L6800-L6950
---
## Core rewrite intrinsics (H200 sm_9.0a, measured)

| C-source form | Intrinsic | PTX | SASS | ns/op |
|---------------|-----------|-----|------|------:|
| `x < 0 ? 0 : x` | `fmaxf(x, 0)` | `max.f32` | `FMNMX` | 14.0 |
| `if (x<0) x=-x` | `fabsf(x)` | `abs.f32` | `LOP3.LUT` | 9.3 |
| `cond ? a : b` | *no intrinsic needed* | `selp.f32` | `FSEL` | ~22 |

## Rewrite-pattern decision table

| C-source pattern | Compiler emits | Right form on H200 |
|-------------------|----------------|--------------------|
| `if (x < c) y = 0; else y = x;` | `FSETP + FSEL` (2 ops) | Use `fmaxf(x, c)` for 1 FMNMX |
| `if (x < 0) x = -x;` | `FSETP + FSEL` (2-3 ops) | Use `fabsf(x)` for 1 LOP3.LUT |
| `if (cond) acc = v;` | `FSEL` (1 op) | Already optimal — keep as-is |
| `acc = cond ? a : b;` | `FSEL` (1 op) | Already optimal — keep as-is |
| `acc * (1-c) + v*c` | `FFMA×2 + FMUL + FADD` (4 ops) | **Anti-pattern** — rewrite as ternary |
| `b & 0x7FFFFFFF` (bit-abs) | `LOP3.LUT ×2` (2 ops) | Use `fabsf(x)` for 1 LOP3.LUT |

## Cross-references

- **`warp-divergence` skill** (warp-divergence): cost model for *real* divergence when the branch body is non-predicable. The branchless-patterns probe measures the *predicable* regime where the compiler emits `FSEL` automatically; warp-divergence measures what happens when it cannot.
- **`fast-math` skill**: `-use_fast_math` affects fp32 transcendentals but not min/max/abs. The intrinsics in this skill are always-on regardless of the flag.
- **`compiler-hints` skill**: `__restrict__` / `__launch_bounds__` interact with predication — tight register budgets can cause the compiler to emit branches where it otherwise would predicate. Measure via `cuobjdump --dump-sass | grep BRA` if performance surprises you.

## Related Probes

- sources/experience/hw-probes/branchless-patterns.md — 10-variant SASS-audited probe. Load-bearing findings: `fmaxf` is 1.56× the `if/else` form, `fabsf` is 2.36×, `cond-arith` is 1.27× *slower* than the ternary form.
- sources/experience/hw-probes/warp-divergence-cost.md — the real-divergence cost companion measurement.
