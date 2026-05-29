---
title: Branch Elimination (predicated patterns + intrinsic-first rewrites)
status: verified
evidence_level: measured
cuda_version_tested: '12.9'
driver_version_tested: 570.124.06
toolchain: nvcc 12.9.86 + ptxas 12.9
measured_on: H200-SXM
applies_to_pattern_class:
- cuda-core
applies_to_ops:
- elementwise
- indexing
- reduction
- normalization
applies_to_dtypes:
- fp32
- fp16
- bf16
- int32
requires_sm: '>=7.0'
requires_features:
- selp
- fsel
single_kernel_useful: true
source:
- path: spec
  anchor: Reference
artifacts:
  code: sources/experience/hw-probes/branchless-patterns/artifacts/branchless_patterns_probe.cu
  build: sources/experience/hw-probes/branchless-patterns/artifacts/build.sh
  introspection: sources/experience/hw-probes/branchless-patterns/artifacts/device.json
  profile: ''
related_apis:
- fmaxf
- fminf
- fabsf
- copysignf
- __fmaxf
- __fminf
- max.f32
- min.f32
- abs.f32
- selp.f32
- selp.b32
related_skills:
- warp-divergence
- fast-math
- compiler-hints
experience_refs:
- sources/experience/hw-probes/branchless-patterns/2026-04-23-branchless-patterns.md
id: skill-branch-elimination
type: skill
vendor: nvidia
tags:
- cuda-cpp
applies_to:
- general
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L25650-L25760
- source_id: cuda-official/toolkit-docs-13.2
  path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/parallel-thread-execution/cuda_parallel-thread-execution_index.html.md
  anchor: L6800-L6950
---
## What

Branch elimination on H200 sm_9.0a is **not what the folklore teaches**. The conventional framing is "rewrite `if/else` as arithmetic to avoid a branch" — but on sm_9.0a the compiler pattern-matches simple-body `if/else` and ternary expressions to `FSEL` (the SASS predicated-select instruction) during codegen. These so-called "branches" are already branchless at the hardware level. The real distinction that matters on H200 is **how many SASS ops** each C-source pattern emits, because wall-clock at compute-bound scales with op count, not branch count.

This skill is therefore a **catalog of rewrites ordered by measured SASS op count**, not a catalog of "branchless" replacements. The three canonical patterns plus their measured op counts on H200 are:

- **ReLU / clamp-to-zero** (`max(x, 0)`): `if/else` and ternary → `FSETP + FSEL` (2 ops); `fmaxf` → `FMNMX` (1 op). Measured **1.56× speedup** from using `fmaxf`.
- **Absolute value** (`|x|`): `if (x<0) x=-x` → `FSETP + FSEL` (2 ops, with `-x` fused into FSEL); `fabsf` → `LOP3.LUT` (1 op, IEEE sign-bit mask); explicit bit-trick → `LOP3.LUT ×2` (2 ops, inefficient). Measured **2.36× speedup** from using `fabsf`.
- **Conditional register assignment** (`acc = cond ? a : b`): `if/else` and ternary → `FSEL` (1 op, already optimal); arithmetic simulation `a*(1-c) + b*c` → `FFMA×2 + FMUL + FADD` (4 ops). The arithmetic form is **1.27× slower**.

## Why

Two reasons this skill exists as a pattern catalog rather than a "always use branchless" doctrine:

1. **Intrinsics (`fmaxf`, `fabsf`, `fminf`, `copysignf`) emit fewer SASS ops than the `if/else` form**, and fewer ops at compute-bound scale directly with fewer wall-clock ns. This is the real win — not branch elimination per se, but op-count reduction through hardware-provided primitives.
2. **Arithmetic simulations of select (`a*(1-c) + b*c`) are strictly worse than the `if/else` form** because FSEL is already a one-op SASS primitive; adding 4 FP ops to "avoid a branch" that doesn't exist at SASS is a net loss. This is a common anti-pattern the skill exists to warn against.

The related but distinct concern — **real branch divergence from non-predicable bodies** (function calls, long compute chains in each arm) — is covered by the sibling `warp-divergence` skill. That cost is real and large (measured 1.82× at compute-bound in that probe). This skill covers only the simple-body rewrites where `FSEL` emission is automatic.

## When to use

### S1. Use `fmaxf` / `fminf` for clamp — 1.56× faster than `if/else`

Any "max against a constant" or "min against a constant" pattern should use the intrinsic, not the ternary or `if/else` form. Measured on H200:

```cuda
// 2 SASS ops (FSETP + FSEL) — 21.87 ns/op in the probe harness:
y = (x < 0.0f) ? 0.0f : x;
y = x < 0.0f ? 0.0f : x;
if (x < 0.0f) y = 0.0f; else y = x;

// 1 SASS op (FMNMX) — 14.02 ns/op, 1.56× faster:
y = fmaxf(x, 0.0f);
```

Same applies for `min(x, upper)` and for full `clamp(x, lo, hi)` built as `fminf(fmaxf(x, lo), hi)` (2 FMNMX ops).

### S2. Use `fabsf` for absolute value — 2.36× faster than `if/else`, 1.51× faster than bit trick

`fabsf` emits a single `LOP3.LUT` doing the IEEE sign-bit mask. The `if (x<0) x=-x` form emits `FSETP + FSEL` (2 ops). The hand-coded bit trick `__int_as_float(__float_as_int(x) & 0x7FFFFFFF)` emits `LOP3.LUT ×2` (2 ops — the round-trip through integer registers is not fused). `fabsf` is strictly best.

```cuda
// 2-3 SASS ops (FSETP + FSEL, possibly + FNEG) — 21.85 ns/op:
if (x < 0.0f) x = -x;

// 2 SASS ops (LOP3.LUT ×2) — 14.02 ns/op, 1.56× faster:
x = __int_as_float(__float_as_int(x) & 0x7FFFFFFF);

// 1 SASS op (LOP3.LUT) — 9.27 ns/op, 2.36× faster than branch, 1.51× faster than bit-trick:
x = fabsf(x);
```

### S3. For conditional register assign, use the form that reads clearest — `if/else`, ternary, and `selp` are all 1 SASS op

`FSEL` is already one SASS op. Writing `if (cond) acc = v;` or `acc = cond ? v : acc;` both pattern-match to `FSEL` at codegen. Do **not** replace them with arithmetic (`acc * (1-c) + v * c`) — that form emits 4 FP ops (FFMA ×2 + FMUL + FADD) and measured **1.27× slower** on H200.

```cuda
// 1 SASS op (FSEL) — 22.63 ns/op. Read-clearest form; pick whichever is cleanest for context:
if (cond) acc = v;
acc = cond ? v : acc;

// 4 SASS ops — 28.67 ns/op, 1.27× slower. DO NOT USE:
float c = static_cast<float>(cond);
acc = acc * (1.0f - c) + v * c;
```

The exception is when the body of each arm is **non-trivial** (function call, long chain of ops) — in that case the compiler cannot pattern-match to `FSEL`, a real branch is emitted, and divergence cost applies. See the `warp-divergence` skill for the measurement (1.82× at compute-bound, non-predicable body). But for simple assignments like these three forms above, FSEL emission is reliable and all three are equivalent.

### S4. When the compiler will NOT emit `FSEL` — force it with a simpler body

If the compiler emits a real branch (you can verify via `cuobjdump --dump-sass | grep BRA`), it is usually because the body is too complex — function call, long expression, store to memory, etc. To force predication, rewrite the body to a scalar value:

```cuda
// Compiler may emit real branch — heavy body blocks predication:
if (cond) {
    dst[i] = complicated_helper(x, y, z);
}

// Predicate-friendly: compute the value unconditionally, select:
float v = complicated_helper(x, y, z);        // always compute
dst[i] = cond ? v : dst[i];                   // 1 FSEL
```

This trades some wasted compute for guaranteed predication. The trade-off is usually worth it when `complicated_helper` is cheap; see pitfall P1.

## When NOT to use

- **When the branch body is non-trivial AND most lanes take the same arm.** If a warp is uniform or near-uniform, the hardware early-exits the inactive arm and divergence cost is minimal. Branchless rewrites with unconditional compute then waste work on the majority of lanes for no win. See pitfall P2.
- **When the arithmetic rewrite adds ops** — as demonstrated by cond-arith at 1.27× slower, writing `a*(1-c) + b*c` is worse than the `if/else` form the compiler already lowers to `FSEL`. Measure instruction count, not branch count.
- **When hoping to avoid a divergence penalty that does not exist.** If the body is already predication-friendly, `FSEL` is emitted and there is no divergence penalty to avoid. Re-writing solely for "branchless-ness" is a common no-op.

## Measured Characteristics

Measured on H200-SXM (sm_9.0a, CUDA 12.9, driver 570.124.06) using sources/experience/hw-probes/branchless-patterns/ — 10 variants × 3 pattern families, compute-bound 4-chain ILP × 1024 inner iters harness, lane-variant input so branchful variants would see worst-case 50% divergence if the compiler had actually emitted branches. Full record: sources/experience/hw-probes/branchless-patterns/2026-04-23-branchless-patterns.md.

### Per-iter cost on one SM chain

| Variant | ns / op | vs in-group branch |
|---------|--------:|-------------------:|
| relu-branch / relu-ternary | 21.87 | 1.00× |
| **relu-fmaxf** | **14.02** | **1.56×** |
| relu-bittrick | 22.34 | 0.98× |
| abs-branch | 21.85 | 1.00× |
| **abs-fabsf** | **9.27** | **2.36×** |
| abs-bittrick | 14.02 | 1.56× |
| cond-branch / cond-ternary | 22.63 | 1.00× |
| **cond-arith** | **28.67** | **0.79× (slower)** |

### PTX / SASS emission (verified 2026-04-23)

| Pattern | C-source form | PTX | SASS |
|---------|---------------|-----|------|
| ReLU | `if/else` or ternary | `setp.lt.f32 + selp.f32` | `FSETP.GEU.AND + FSEL` |
| ReLU | `fmaxf(x, 0)` | `max.f32` | `FMNMX` |
| ReLU | `__int_as_float(b & ~(b>>31))` | `shr.s32 + and.b32 + and.b32` | `SHF.R.S32.HI + LOP3.LUT ×2` |
| abs  | `if/else -x` | `setp + selp + neg` | `FSETP.GEU.AND + FSEL` (neg fused) |
| abs  | `fabsf(x)` | `abs.f32` | `LOP3.LUT` |
| abs  | `b & 0x7FFFFFFF` | `and.b32` | `LOP3.LUT ×2` |
| cond | `if/else` or ternary | `selp.f32` | `FSEL` |
| cond | `acc*(1-c) + v*c` | `fma.rn.f32 ×2 + sub + mul` | `FFMA ×2 + FMUL + FADD` |

Key measured findings:

- **"If/else" is NOT a branch at SASS** for simple-body conditionals on sm_9.0a. Compiler emits `FSEL` reliably. C-source style (`if/else` vs ternary) is cosmetic.
- **`fmaxf` / `fabsf` win by emitting fewer SASS ops, not by "avoiding branches".** Single-op `FMNMX` / `LOP3.LUT` vs 2-3 ops for the `if/else` path.
- **Bit tricks are equivalent-or-slower than the intrinsic** on H200 — the compiler can synthesise `fabsf` to one LOP3.LUT, while explicit bit code goes through two LOP3s.
- **Arithmetic simulation of select is a performance anti-pattern** (cond-arith 1.27× slower). Write `if/else` or ternary; the compiler already produces optimal `FSEL`.

## Principles

1. **Count SASS ops, not branches.** The ns/op measured correlates with SASS op count. The word "branchless" is misleading on H200 — most C-source "branches" are already branchless at SASS.
2. **Prefer single-op intrinsics.** `fmaxf`, `fminf`, `fabsf`, `copysignf` each map to a single SASS instruction. Use them whenever a pattern matches.
3. **Do not rewrite `if/else` as arithmetic.** `acc*(1-c) + v*c` is slower than `if (c) acc = v;`. FSEL is already a 1-op branchless primitive; arithmetic simulation adds 3-4 extra ops.
4. **If the compiler emits a real branch, check the body complexity.** Non-predicable bodies (function calls, stores, long chains) block FSEL emission. Simplify the body or move work out of the branch to restore predication.
5. **Cross-reference `warp-divergence`** for the cost of real (non-predicated) branches. This skill covers the cheap predicable rewrites; real divergence is that skill's domain.

## Open questions

- Q1. Does `fminf`/`fmaxf` scale the same way with fp16 (`__hmax`, `__hmin`) and bf16 inputs? Not measured here. Follow-up probe `sources/experience/hw-probes/half-minmax/` (open).
- Q2. Does `copysignf` emit a single `LOP3.LUT` like `fabsf`, or a different sequence? Not measured here; SASS inspection sufficient if needed.
- Q3. When the `if/else` body performs a memory store (not a register update), does the compiler still emit `FSEL` or does it fall back to a real branch? Probe not done; expected answer from PTX documentation is "predicated store". Follow-up if a specific workload exhibits surprising divergence.
- Q4. How much harder does it become to force predication as the predicated body grows? The `warp-divergence` probe establishes the upper-bound cost (1.82×) when predication fails; the crossover point from "compiler predicates" to "compiler emits branch" is compiler-version-dependent and not measured here.

## Legacy references

The legacy KB at `corpus/nvidia/legacy-advanced/branch-elimination/{branchless-patterns.md, predicated-execution.md, uniform-control-flow.md}` consisted of three empty placeholder files; there is no prior content to import. This skill is a fresh build whose structure and findings come entirely from the branchless-patterns probe and from the PG / PTX ISA citations above. The legacy placeholder path is recorded here solely so future migrations can confirm the "no content to import" state; all pattern recommendations in the body of this skill stand on the probe 2026-04-23 SASS audit and measured wall-clock.
