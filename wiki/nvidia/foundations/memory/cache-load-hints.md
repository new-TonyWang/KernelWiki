---
title: Cache Load Hints (__ldg / __ldca / __ldcg / __ldcs / __ldcv / __ldlu)
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
- normalization
- reduction
requires_sm: '>=3.5'
requires_features:
- ptx-ld-cache-operators
single_kernel_useful: true
source:
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L25083-L25130
  excerpt: 'Read-Only Data Cache Load Function: T __ldg(const T* address); reads memory
    through the non-coherent read-only data cache. Additional intrinsics __ldca /
    __ldcg / __ldcs / __ldlu / __ldcv expose the ld.global.{ca,cg,cs,lu,cv} PTX cache
    operators.'
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/parallel-thread-execution/cuda_parallel-thread-execution_index.html.md
  anchor: L10400-L10490
  excerpt: 'PTX ld cache operators: .ca (cache all — default), .cg (cache global,
    L2 only, bypass L1), .cs (cache streaming, evict-first), .lu (last-use, evict
    after), .cv (cache volatile, always fetch from system memory).'
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1289-L1330
  excerpt: The read-only cache path can reduce pressure on the unified L1/Tex cache
    for const __restrict__ inputs. On modern architectures the compiler routes const
    __restrict__ loads through this path automatically.
artifacts:
  code: sources/experience/hw-probes/cache-hint/artifacts/cache_hint_probe.cu
  build: sources/experience/hw-probes/cache-hint/artifacts/build.sh
  introspection: sources/experience/hw-probes/cache-hint/artifacts/device.json
  profile: ''
related_apis:
- __ldg
- __ldca
- __ldcg
- __ldcs
- __ldcv
- __ldlu
- ld.global.nc
- ld.global.ca
- ld.global.cg
- ld.global.cs
- ld.global.cv
- ld.global.lu
related_skills:
- l2-access-policy
- coalescing
- vectorized-access
- register-pressure
experience_refs:
- sources/experience/hw-probes/cache-hint/2026-04-23-cache-hint.md
id: skill-cache-load-hints
type: skill
vendor: nvidia
tags:
- cuda-cpp
applies_to:
- general
---
## What

CUDA exposes **six** per-load cache operators that steer a global-memory
read into a specific cache level. At the PTX level they are
`ld.global.{ca, cg, cs, lu, cv}` plus the non-coherent `ld.global.nc`;
in CUDA C they appear as `__ldca / __ldcg / __ldcs / __ldlu / __ldcv`
plus the specially-named `__ldg` (which emits `ld.global.nc`).

Measured instruction mapping on H200 sm_9.0a (verified via `nvcc -ptx` + `cuobjdump --dump-sass` on the probe's six template specialisations, 2026-04-23):

| Intrinsic | PTX emitted | SASS emitted | Caches used | Intended use |
|-----------|-------------|--------------|-------------|--------------|
| *default* (`*p` with `const __restrict__`) | `ld.global.nc` | `LDG.E.CONSTANT` | Read-only / L1-TEX | Compiler auto-promotes read-only pointers to the non-coherent path; identical to `__ldg` at HW level |
| `__ldg`     | `ld.global.nc` | `LDG.E.CONSTANT`    | Read-only / L1-TEX    | Read-only data known immutable for kernel lifetime |
| `__ldca`    | `ld.global.ca` | `LDG.E.STRONG.SM`   | L1 + L2               | Explicit cache-all; distinct from default on H200 (default is `.nc` not `.ca` when pointer is `const __restrict__`) |
| `__ldcg`    | `ld.global.cg` | `LDG.E.STRONG.GPU`  | L2 only (bypass L1)   | Data not reused within SM |
| `__ldcs`    | `ld.global.cs` | `LDG.E.EF`          | L1 + L2, **evict-first** | Streaming one-shot data |
| `__ldlu`    | `ld.global.lu` | (not in probe)      | L1 + L2, **last-use** | Final read before discard |
| `__ldcv`    | `ld.global.cv` | `LDG.E.STRONG.SYS`  | **always re-fetch** from memory | Volatile / cross-SM synchronization |

Two things are specific to sm_9.0a (H200) and critical to using the
hints correctly:

1. The L1 and Tex caches are **unified** (one physical L1/TEX unit). `__ldg`'s old benefit — tapping a parallel texture pipeline — evaporates. Measured: `__ldg` and default differ by < 0.5% on H200, AND both lower to the same SASS opcode `LDG.E.CONSTANT` (audit 2026-04-23).
2. With `const __restrict__` parameters, the compiler already emits
   the read-only form (`ld.global.nc` or an equivalent optimized form)
   automatically. Explicit `__ldg` is redundant on modern toolchains
   and only helps when the compiler cannot prove immutability (no
   `const`, aliased pointers, etc.).

The skill is therefore not "always sprinkle cache hints"; it is a
narrow toolbox with one measurably useful operator (`__ldcs` under
contended L2) and one measurably dangerous one (`__ldcg` when the
data is L2-resident and reused within an SM).

## Why

On H200's unified L1/TEX, the **only measurable per-load decision**
is which levels of the hierarchy to use: L1+L2 vs L2-only vs
L1+L2+evict-first vs bypass-all. Measured wall-clock differences:

- **DRAM-bound single-pass reads** (working set > L2, each element
  read once): all six operators within **0.5%** — hint choice is free
  and irrelevant. Spend your optimization time on ILP or
  vectorized-access instead.
- **L2-resident reused reads** (working set < L2, multiple passes
  per launch): choosing `__ldcg` or `__ldcv` costs **2.26×** wall-
  clock because L1 is bypassed. The compiler's default (`ld.global.nc` with `const __restrict__`, else `ld.global.ca`)
  or its semantic alias (`__ldca`, or `__ldg` on read-only data) is
  the right choice.
- **Evict-first (`__ldcs`)** shows no measurable effect in an
  under-contended L2 — it tags lines for eviction preference but
  only matters when another stream is fighting for the same L2
  capacity. Same regime-gap pattern as `l2-access-policy` skill.

The legacy skill's framing ("use `__ldcg` when data is not reused by
the same SM") is **a trap** for typical cuda-core kernels, because
single-kernel reuse patterns (multiple passes in reduction /
normalization / scan) are exactly the SM-local reuse that L1 staging
accelerates. `__ldcg` is for multi-kernel pipelines where the reuse
is cross-SM (one block writes, a different block in a later launch
reads) — a pattern that doesn't show up in cuda-core MVP.

## When to use

### S1. Default / `__ldca` — the correct choice for almost everything

```cuda
float v = in[i];                 // default with const __restrict__: ld.global.nc / LDG.E.CONSTANT (audited)
// or explicitly:
float v = __ldca(&in[i]);        // ld.global.ca / LDG.E.STRONG.SM (distinct from default!)
```

With `const __restrict__`, default lowers to `ld.global.nc` (identical to `__ldg`); without the qualifier, or with explicit `__ldca`, it lowers to `ld.global.ca`. Both forms are safe for any read pattern
where the data is reused — within a warp, within a block, within
an SM, across loop iterations. **This is the right choice in ≥ 90%
of kernels.**

### S2. `__ldg` — equivalent to default on H200, harmless legacy pattern

```cuda
__global__ void k(const float* __restrict__ in, ...) {
    float v = __ldg(&in[i]);     // ld.global.nc
}
```

Maps to `ld.global.nc` via the non-coherent read-only path. On H200
the unified L1/TEX makes this the same physical cache as the default
load, and measured 0.1-0.4% spread vs `default` is within noise.
Writing `__ldg` is *not wrong* — it documents read-only intent and
matches older codebases — but it is not *optimization*. Prefer
`const __restrict__` on the parameter and let the compiler emit the
same instruction automatically.

### S3. `__ldcg` — only for cross-SM non-reuse patterns

```cuda
// ONLY when this buffer is NOT touched again by any thread in this
// kernel's SM residency window. Typical use: producer kernel writes,
// different consumer kernel (possibly on a different SM) reads once.
float v = __ldcg(&in[i]);        // ld.global.cg (L2 only, bypass L1)
```

Maps to `ld.global.cg` (cache-global, bypass L1). **Measured on H200:
2.26× slower** than default when the buffer is L2-resident and
re-read 16 times within a single kernel launch. The "save L1 capacity
for other data" argument (legacy Skill 2) only holds when:

- The "other data" actually exists (another resident buffer benefits
  from staying in L1), AND
- The `__ldcg`-loaded buffer is NOT re-read by the same SM during
  the same launch.

Single-kernel reductions, normalizations, and scans typically make
multiple passes over the input — so they are firmly **not** the
right place for `__ldcg`.

### S4. `__ldcs` — measurably a null on under-contended L2

```cuda
float v = __ldcs(&in[i]);        // ld.global.cs (evict-first)
```

Maps to `ld.global.cs`. Marks the cache line for preferential
eviction when L2 is under pressure. **Measured on H200 at L2 regime
(8 MiB buffer, 16 inner passes): identical to default** — because a
60 MiB L2 with 8 MiB of hot data has no pressure to resolve.

The legitimate use case is a **contended L2** — when a concurrent
stream is touching > 20 MiB of its own data and would otherwise
evict our streaming buffer. In single-kernel MVPs, drop `__ldcs` and
use the default.

### S5. `__ldcv` — for correctness, not performance

```cuda
int ready = __ldcv(&flag);       // ld.global.cv (always re-fetch)
```

Maps to `ld.global.cv` (cache-volatile). Bypasses all caches and
always re-fetches from L2/DRAM. **Use only when polling a flag
written by another thread/kernel** (correctness). Measured: 2.26×
slower than default at L2 regime on H200, with no compensating
benefit for non-volatile data.

### S6. `__ldlu` — last-use eviction hint (not re-measured on H200)

```cuda
float v = __ldlu(&temp[i]);      // ld.global.lu (last-use)
```

Maps to `ld.global.lu`. Tags the line for eviction after this load.
Its effect is on **subsequent** L2 pressure from later kernels — a
cross-kernel optimization that a single-kernel harness cannot
surface. Retained as inferred pending a multi-kernel probe
(`sources/experience/hw-probes/cache-hint-contended/`, open).

## When NOT to use

- **DRAM-bound streaming kernels**. Measured: all six variants land
  at ~1.38 TB/s within 0.5% on H200. The hint is free and irrelevant.
- **Any time L1 staging is useful**. That includes every typical
  reduction / normalization / scan / elementwise-with-shared-input
  pattern. `__ldcg` / `__ldcv` cost 2.26× in those regimes.
- **As a substitute for `cudaStreamAttributeAccessPolicyWindow`**.
  The runtime-level window (skill `l2-access-policy`) and
  per-instruction hints are *complementary*, not alternatives. For
  a hot buffer shared across kernels, use the window **and** default
  loads inside the kernel.
- **To replace `const __restrict__` hygiene**. The compiler's
  auto-emit path covers all the `__ldg` use cases when pointers are
  correctly qualified. Fix your parameter declarations first.

## Measured Characteristics

Measured on H200-SXM (sm_9.0a, CUDA 12.9, driver 570.124.06) using
[sources/experience/hw-probes/cache-hint/](../../../sources/experience/hw-probes/cache-hint/) —
6 load variants × 2 working-set regimes. Full record:
[sources/experience/hw-probes/cache-hint/2026-04-23-cache-hint.md](../../../sources/experience/hw-probes/cache-hint/2026-04-23-cache-hint.md).

### DRAM regime (256 MiB, single pass) — variants collapse to DRAM BW

| variant   | GB/s_eff | vs default |
| --------- | -------: | ---------: |
| default   |  1377.44 |      1.000 |
| `__ldg`   |  1383.35 |      1.004 |
| `__ldca`  |  1381.52 |      1.003 |
| `__ldcg`  |  1382.66 |      1.004 |
| `__ldcs`  |  1384.26 |      1.005 |
| `__ldcv`  |  1384.26 |      1.005 |

Spread: **0.5%**. At DRAM-bound scale the per-load cache hint has
zero effect; every variant runs at ~29% HBM3e SOL.

### L2 regime (8 MiB, 16 inner passes) — L1 bypass is the only measurable loss

| variant       | GB/s_eff | vs default | note |
| ------------- | -------: | ---------: | ---- |
| default       |  6043.67 |      1.000 | L1+L2 staged |
| `__ldg`       |  6052.39 |      1.001 | same as default |
| `__ldca`      |  6034.97 |      0.998 | same as default |
| **`__ldcg`**  | **2668.13** | **0.441** | **L1 bypass** |
| `__ldcs`      |  6043.67 |      1.000 | evict-first = null under un-contended L2 |
| **`__ldcv`**  | **2673.23** | **0.442** | bypass-all behaves like L1 bypass |

Key measured findings:

- **`__ldg` vs default = 0.1% on L2, 0.4% on DRAM** — within noise.
  Legacy pitfall P6 ("const __restrict__ already emits the right
  instruction") upgraded from inferred to measured.
- **`__ldcg` / `__ldcv` are 2.26× slower at L2 regime** because L1
  staging is bypassed. For any kernel pattern with SM-local reuse
  (including the typical grid-stride multi-pass pattern) these are
  the wrong choice.
- **`__ldcs` is a measured null at L2 regime**. Legacy pitfall P4
  ("streaming hint on reused data") confirmed on H200 in the
  un-contended case; the hint matters only when another stream is
  fighting for L2.
- **DRAM regime is variant-independent** at 0.5% spread. Spend
  optimization effort on ILP / vector-load width instead.

Follow-up probes open:

- `sources/experience/hw-probes/cache-hint-contended/` — add a second
  stream touching 50+ MiB of distinct data so the evict-first tag
  (`__ldcs`) and last-use tag (`__ldlu`) can be measured.
- `sources/experience/hw-probes/store-hint/` — counterpart for
  `__stcs` / `__stwb` / `__stwt`; legacy Skill 4 not re-measured.

## Principles

1. **Default / `__ldca` / `__ldg` are equivalent on H200 for
   `const __restrict__` pointers.** Pick whichever the codebase
   reads clearest; don't chase micro-optimizations between them.
2. **Never bypass L1 on data with SM-local reuse.** `__ldcg` and
   `__ldcv` are 2.26× slower when the buffer is L2-resident and
   re-read, which is the typical reduction / normalization pattern.
3. **`__ldcv` is for correctness, not performance.** Use it to poll
   flags written by other kernels. Never to "bypass stale caches"
   in a single-kernel context — the caches are coherent for
   non-`__ldg` loads within a kernel.
4. **Cache hints compose with the runtime-level policy window.**
   The `cudaStreamAttributeAccessPolicyWindow` picks which lines
   L2 keeps; the cache hint picks which level the *load instruction*
   routes through. They operate on different axes.

## Open questions

- Q1. Does `__ldcs` measurably help when a concurrent stream is
  evicting the hot buffer? Follow-up probe
  `sources/experience/hw-probes/cache-hint-contended/` (open).
- Q2. **RESOLVED** (audit 2026-04-23): `nvcc -arch=sm_90a -O3 -ptx` emits six distinct `ld.global.{nc,ca,cg,cs,lu,cv}` instructions — no compiler folding. `cuobjdump --dump-sass` confirms six distinct SASS opcodes (`LDG.E.CONSTANT` / `LDG.E.STRONG.SM` / `LDG.E.STRONG.GPU` / `LDG.E.EF` / `LDG.E.STRONG.SYS`). The DRAM-regime 0.5% collapse is 'all hints honoured identically at DRAM-bound scale', not compiler folding. Default and `__ldg` share the same SASS (`LDG.E.CONSTANT`) with `const __restrict__` pointers, explaining the 0.1-0.4% wall-clock identity. See probe record §"PTX / SASS audit".
- Q3. How do `__stcs` / `__stwb` / `__stwt` store hints behave on
  H200? Legacy Skill 4 claimed use cases but was never re-measured.
  Follow-up probe `sources/experience/hw-probes/store-hint/` (open).
- Q4. Is `cudaFuncSetCacheConfig` observable on sm_9.0a? Legacy
  pitfall P10 claims "no observable effect on Ampere+". Not
  re-measured in this pass — belongs in a dedicated `L1/shared
  carveout` probe (also open).

## Legacy references

- `legacy_sandbox_path`:
  `corpus/nvidia/legacy-optimization/memory/cache-load-hints/skill.md`.
  Legacy kept six sub-skills (S1 `__ldg`, S2 `__ldcg`, S3 `__ldcs`,
  S4 store hints, S5 `cudaFuncSetCacheConfig`, S6 `__ldlu`); this
  port reorganises by the H200 measurement:
  - S1 folded into S2 (same wall-clock on H200) and demoted.
  - S2 (cache-bypass L1) promoted to "measurably harmful in
    SM-local-reuse patterns" (S3 in this skill).
  - S3 (streaming) kept but marked as null outside contended regimes.
  - S4 (store hints) deferred to a separate probe; retained in
    legacy section as inferred.
  - S5 (cache-config) routed to the `register-pressure` / shared-
    carveout space (pending probe).
  - S6 (`__ldlu`) retained as inferred pending contended-L2 probe.
- Legacy pitfalls P1 (staleness), P2 (hints ignored), P5 (config
  override), P9 (store hints irrelevant when load-bound), P10
  (sm_80+ auto-managed L1/shared), P11 (`__ldlu` single-kernel
  null) are retained anecdotally. P6 (redundant `__ldg`), P7
  (default ≈ `__ldcg` on streaming), P8 (DRAM-bytes shift without
  wall-clock impact) are **upgraded to measured** by the H200 probe.
- **Related but distinct**:
  - `wiki/nvidia/foundations/memory/l2-access-policy/` operates at the stream-level
    — which lines L2 keeps under pressure. Composes with this skill.
  - `wiki/nvidia/foundations/memory/coalescing/` fixes the address pattern; cache
    hints are a downstream concern.
