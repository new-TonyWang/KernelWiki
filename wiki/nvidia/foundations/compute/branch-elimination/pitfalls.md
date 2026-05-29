---
title: Branch Elimination — Pitfalls
status: verified
related_skill: wiki/nvidia/foundations/compute/branch-elimination/skill.md
source:
- path: spec
  anchor: Reference
experience_refs:
- sources/experience/hw-probes/branchless-patterns/2026-04-23-branchless-patterns.md
id: pitfall-branch-elimination
type: pitfall
vendor: nvidia
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/parallel-thread-execution/cuda_parallel-thread-execution_index.html.md
  anchor: L6800-L6950
---
All pitfalls below are grounded either in the branchless-patterns probe 2026-04-23 (10-variant SASS audit) or in the PTX ISA documentation cited in the skill's frontmatter. None is anecdotal.

## P1. Arithmetic simulation of select is slower than `if/else` / ternary

**Symptom**: Replacing `if (cond) acc = v;` with `acc = acc * (1 - cond) + v * cond;` (or a similar FMA-based "branchless" rewrite) makes the kernel measurably slower instead of faster.
**Detection**: On H200 sm_9.0a the measured per-iter cost for `cond-arith` is **28.67 ns/op** vs **22.63 ns/op** for the `if/else` and ternary forms — 1.27× slower. SASS audit confirms the arithmetic form emits `FFMA ×2 + FMUL + FADD` (4 ops) while the `if/else` form already lowers to a single `FSEL`.
**Fix**: Use the plain `if/else` or ternary form. The compiler pattern-matches simple-body conditionals to `FSEL` automatically; there is no branch to eliminate at the SASS level.
**Source**: branchless-patterns probe 2026-04-23, SASS audit.

## P2. Bit-trick `abs` / `relu` is not faster than the intrinsic

**Symptom**: Hand-coded bit-twiddling like `__int_as_float(__float_as_int(x) & 0x7FFFFFFF)` for absolute value is not faster — and can be slower — than `fabsf(x)`.
**Detection**: Measured on H200 sm_9.0a: `fabsf` takes 9.27 ns/op (single `LOP3.LUT`); the explicit bit-trick takes 14.02 ns/op (two `LOP3.LUT`). The bit-trick is 1.51× slower than the intrinsic because the int↔float round-trip through integer registers prevents the compiler from fusing the mask into a single opcode.
**Fix**: Use `fabsf(x)` / `fmaxf(x, 0.0f)` etc. Modern compilers generate the tightest SASS for the intrinsic; hand bit tricks compete against the compiler's output and typically lose.
**Source**: branchless-patterns probe 2026-04-23, SASS audit.

## P3. `if/else` with a non-trivial body blocks predication

**Symptom**: The compiler emits a real branch (visible as `BRA` in SASS) instead of `FSEL`, and a divergent warp pays the full serialization cost.
**Detection**: `cuobjdump --dump-sass <binary> | grep -c BRA` shows more than the expected count (per-kernel epilogue contributes a few BRA; inner-loop BRA is the red flag). If wall-clock regresses when `cond` is made lane-variant, the body is divergence-sensitive.
**Fix**: Simplify the per-arm body. Move function calls, memory stores, and long expression chains *out* of the branch; compute the value unconditionally in registers, then use a final `FSEL` (ternary or `if` with a register assignment):

```cuda
// BAD — real branch, divergence-sensitive:
if (cond) dst[i] = expensive_helper(x, y, z);

// GOOD — one unconditional compute, one FSEL:
float v = expensive_helper(x, y, z);
dst[i] = cond ? v : dst[i];
```

The trade-off is a small amount of wasted compute on lanes that will ignore `v`; usually worth it when the helper is cheap. If the helper is expensive, consider warp-vote (`__any_sync`, `__all_sync`) to short-circuit uniform-false warps — see the `warp-divergence` skill.
**Source**: PTX ISA §selp / §bra (predication fallback conditions).

## P4. `fmaxf` and `fminf` do not protect against NaN the same way

**Symptom**: `fmaxf(NaN, x)` returns `x` (NaN is "quiet" and the other operand wins); `fminf(NaN, x)` returns `x` too. This differs from C++ `std::max` / `std::min` which are unordered-aware in different ways.
**Detection**: Kernel output contains unexpected non-NaN values downstream of an op on partially-NaN data.
**Fix**: If NaN propagation matters, either pre-check with `isnan(x)` or do the compare explicitly. For most ML/kernel workloads with well-defined inputs this is not a concern — but be aware of the semantic.
**Source**: PG §Standard Math Functions (IEEE 754 behaviour of the `fmax/fmin` family).

## P5. The compiler's if-conversion threshold is version-dependent

**Symptom**: A kernel that emitted `FSEL` under CUDA 12.4 emits `BRA` under CUDA 12.9 (or vice versa), and a performance regression appears in a minor toolkit bump.
**Detection**: Compare `cuobjdump --dump-sass` output between toolkit versions for the same source; diff the presence/absence of `BRA` in the inner loop.
**Fix**: If a specific branch MUST be predicated for performance, make the body unambiguously simple (scalar register assignment, no function calls, no store). A body that is 1-2 arithmetic ops is almost always predicated. A body that calls another function will often not be predicated.
**Source**: Empirical observation; ptxas if-conversion heuristics are not publicly specified. branchless-patterns probe 2026-04-23 verified ptxas 12.9 predicates the 3 simple-body patterns on sm_9.0a.

## P6. "Branchless" is a misleading label

**Symptom**: Code review / tutorials frame rewrites like `acc * mask + v * (1-mask)` as "branchless optimizations" that "avoid the branch penalty". This framing leads developers to rewrite already-fast code into slower code.
**Detection**: Any time you see a claim that a bitwise / arithmetic form is faster "because it avoids branching", check the SASS. On H200 sm_9.0a with simple bodies, the `if/else` form usually already emits `FSEL` and is byte-identical to the ternary form. A "branchless" rewrite that adds arithmetic ops is a pessimization.
**Fix**: Think in SASS op counts, not branch counts. The right question is "does my rewrite emit fewer SASS instructions than the compiler's output for the `if/else` form?" — not "does my rewrite eliminate a branch?".
**Source**: branchless-patterns probe 2026-04-23. The skill body documents the specific op-count relationships.
