---
title: Cache Load Hints — Pitfalls
status: verified
related_skill: 30-skill/memory/cache-load-hints/skill.md
source:
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L25083-L25230
  excerpt: Read-only data cache load function + low-level load/store cache hint intrinsics.
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/parallel-thread-execution/cuda_parallel-thread-execution_index.html.md
  anchor: L10400-L10490
  excerpt: 'PTX ld.global cache operators: .ca .cg .cs .lu .cv.'
experience_refs:
- 80-experience/hw-probes/cache-hint/2026-04-23-cache-hint.md
id: pitfall-cache-load-hints
type: pitfall
vendor: nvidia
---
Legacy P1–P5 came from L3-sandbox verification on pre-H200 hardware.
P6–P11 are L3-sandbox findings copied from the legacy KB. The H200
probe on 2026-04-23 re-measured P6, P7, and the `__ldcg` /
`__ldcv` regressions; annotations below distinguish
"**Measured on H200 sm_9.0a**" results from legacy claims.

## P1. Overusing `__ldg` on data that is modified (staleness)

**Symptom**: Stale data read because `__ldg` uses the non-coherent
read-only cache path (PTX `ld.global.nc`). Writes by the same or
another kernel are not reflected.
**Detection**: Correctness failure when kernel reads data that was
recently written. Texture/RO cache is not invalidated by global
stores during the same launch.
**Fix**: Only use `__ldg` for data truly read-only for the kernel
execution. For data possibly modified by the same kernel, use
default loads (or drop the `__restrict__` qualifier so the compiler
cannot auto-emit `ld.global.nc`).
**Not re-measured on H200** — the probe kernel never writes to the
buffer. Retained as legacy-anecdotal pending a read-after-write
correctness probe.
**Source**: PG §5.4.8.3 (L25083-L25130).

## P2. Cache hints ignored by compiler or hardware

**Symptom**: No performance change despite using a cache-hint
intrinsic.
**Detection**: Profile shows identical cache behavior with and
without the hint. Compiler may reorder or merge loads; hardware may
ignore advisory operators.
**Fix**: Verify with `nvcc -ptx` that the expected
`ld.global.{ca,cg,cs,lu,cv,nc}` instruction is present. On H200 the
templated probe confirms the compiler emits distinct instructions
per template (the wall-clock delta at L2 regime is the evidence —
if all had merged, __ldcg would not be 2.26× slower than default).
**Source**: PG §2.2.3.6; probe 2026-04-23.

## P3. L1 bypass (`__ldcg`) causing L2 pressure — and SM-local reuse loss

**Symptom (legacy, A100)**: "Using `__ldcg` everywhere causes L2
miss rate to increase because more traffic goes through L2."
**Symptom (measured on H200 sm_9.0a, 2026-04-23)**: The wall-clock
cost is concrete and large: **2.26× slowdown** at L2-resident
reuse regime (8 MiB buffer, 16 inner passes). The L2 miss-rate
framing is a red herring on H200 — the actual problem is that
bypassing L1 forces every read to route L2→registers, and L2 BW
per SM is lower than L1 BW per SM.
**Detection**: NCU `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum
= 0` for kernels using `__ldcg`; compare wall-clock to a default
variant on the same data.
**Fix**: Only use `__ldcg` when the buffer is truly not reused
within the SM's residency window. In single-kernel reductions /
normalizations / scans that pass over input multiple times, this
condition is almost never met — use default / `__ldca` / `__ldg`
instead. The legacy framing "save L1 capacity for other data"
requires the "other data" to actually benefit from L1; verify
before reaching for `__ldcg`.
**Source**: PG §5.4.8.3; probe 2026-04-23.

## P4. Streaming hint (`__ldcs`) on reused data

**Symptom (legacy)**: "Using `__ldcs` on data that is actually
accessed multiple times causes repeated cache misses."
**Symptom (measured on H200 sm_9.0a, 2026-04-23)**: **Not observed
in un-contended L2.** With an 8 MiB buffer in a 60 MiB L2 and no
competing traffic, `__ldcs` tags lines evict-first but the tags
never trigger because there is no memory pressure to resolve.
Measured wall-clock is **identical to default** (6043.67 GB/s
both at 16 inner passes).
**Updated fix**: The hazard still exists — but only in **contended
L2** regimes. If you are applying `__ldcs` expecting a "streaming
benefit" in a single-kernel workload, measure first; it is very
likely a null. Single-stream MVPs should default-load.
**Follow-up probe**: `80-experience/hw-probes/cache-hint-contended/`
(open).
**Source**: PG §5.4.8.3; probe 2026-04-23.

## P5. `cudaFuncSetCacheConfig` overridden by runtime

**Symptom**: Requesting `cudaFuncCachePreferL1` but kernel still
gets the default L1/shared split.
**Detection**: Profile shows shared-memory carveout unchanged.
**Fix (legacy)**: Use `cudaFuncSetAttribute` with
`cudaFuncAttributePreferredSharedMemoryCarveout` for more direct
control on newer architectures.
**Note on H200**: Legacy P10 claims the config is entirely
hardware-managed on Ampere+; not re-measured by this probe.
Retained as inferred.
**Source**: PG §3.2.6.

## P6. `const __restrict__` already emits the read-only path — `__ldg` is redundant

**Symptom (legacy)**: "The skill description itself acknowledges
this: 'The compiler may already generate `ld.global.nc` for
`const __restrict__` pointers' — in practice this means `__ldg`
is a no-op on any reasonably modern toolchain."
**Symptom (measured on H200 sm_9.0a, 2026-04-23)**: Confirmed.
`__ldg` vs `default` wall-clock spread is **0.1%** at L2 regime
and **0.4%** at DRAM regime — both within noise. Legacy P6
**upgraded from inferred to measured**.
**Fix**: Write `const __restrict__` on your parameters and let the
compiler emit the right instruction. Explicit `__ldg` is not
wrong but is not optimization either — it's documentation. Use it
when the compiler cannot prove immutability (non-const pointer,
aliased args) or when matching an existing codebase style.
**Source**: PG §5.4.8.3; BP §12.2.5 (L1289-L1330); probe 2026-04-23.

## P7. Simple streaming kernel: default ≈ streaming hint

**Symptom (legacy)**: "On simple streaming kernels the default
load path already behaves like `__ldcg`; the hint only matters
when there is measurable L1 contention from competing data that
would benefit from L1 residency."
**Symptom (measured on H200 sm_9.0a, 2026-04-23)**: Refined.
**At DRAM regime** (256 MiB single-pass), all six variants are
identical at ~1.38 TB/s — consistent with legacy P7.
**At L2 regime** (8 MiB reused), `default` is ~2.26× **faster**
than `__ldcg` because the default uses L1 staging and `__ldcg`
bypasses it. So legacy P7's "default behaves like __ldcg" is
only true in the DRAM-bound regime; in the L2-resident regime the
two are dramatically different.
**Upgraded P7 fix**: Default-load unless you have measured
evidence of L1 contention from competing data. "Saving L1" is
not a free optimization on H200.
**Source**: Level-3 sandbox verification (2026-04-04); probe 2026-04-23.

## P8. Metric shifts from cache hints are misleading without wall-clock gain

**Symptom (legacy)**: "DRAM bytes written dropped ~14.8% with
`__stcs`, suggesting the cache-streaming hint altered write-back
behavior, but this did not translate to any execution time
improvement — metric shifts from cache hints can be misleading
without end-to-end speedup."
**Context on H200 sm_9.0a, 2026-04-23**: Confirmed on the load
side as well. The DRAM regime NCU `dram__bytes_read.sum` is
identical across all six load variants (268 MB per launch,
matches the 256 MiB × 1 pass buffer), yet wall-clock is also
identical. Metric-level "hints are working" evidence is not
evidence of speedup.
**Fix**: Always measure wall-clock or effective BW as the
primary signal. Per-section NCU metric deltas are corroborating
at best; a metric shift with no wall-clock delta means the hint
is either irrelevant or the bottleneck has moved elsewhere.
**Source**: Level-3 sandbox verification (2026-04-04); probe 2026-04-23.

## P9. Store cache hints only matter when stores are the bottleneck

**Symptom (legacy)**: "Store cache hints only matter when stores
are the bottleneck; in load-latency-bound kernels they are
irrelevant and may even slightly degrade performance by bypassing
L2 write-back coalescing."
**Not re-measured on H200** — this probe is read-only.
`80-experience/hw-probes/store-hint/` (open) is the follow-up.
**Fix (legacy)**: Profile first. If the kernel is load-bound,
skip store hints; if store-bound, compare `__stwb` (default) vs
`__stcs` (evict-first) vs `__stcg` (L2-only) vs `__stwt`
(write-through) and pick the winner.
**Source**: Level-3 sandbox verification (2026-04-04).

## P10. `cudaFuncSetCacheConfig` has no observable effect on Ampere+

**Symptom (legacy)**: "On Ampere and later architectures, the
L1/shared memory split is managed automatically by hardware;
`cudaFuncSetCacheConfig` has no observable effect and may give
a false sense of tuning."
**Not re-measured on H200 sm_9.0a**. Retained as inferred. The
probe does not exercise this API, and legacy P5 already covers
the typical symptom (config request ignored).
**Fix**: Don't rely on `cudaFuncSetCacheConfig` for L1 sizing on
sm_90a; if you truly need more shared memory (≥ 48 KB per block)
use `cudaFuncSetAttribute` with
`cudaFuncAttributeMaxDynamicSharedMemorySize`.
**Source**: Level-3 sandbox verification (2026-04-04).

## P11. `__ldlu` benefit invisible in single-kernel microbench

**Symptom (legacy)**: "The `__ldlu` hint did reduce DRAM write
volume (likely fewer dirty L2 evictions), but this benefit is
invisible in runtime for memory-bandwidth-bound kernels that
are already underutilized — the cache pressure relief only
matters when other concurrent kernels or data streams are
competing for cache capacity."
**Consistent with H200 2026-04-23 probe**: The same regime-gap
observed for `__ldcs` (P4) applies to `__ldlu`. Single-stream
microbench cannot show a cross-kernel L2-pressure-relief
benefit; belongs in the contended follow-up probe.
**Fix**: Retain `__ldlu` on known last-use reads (final iteration
of a loop over a temp buffer the kernel is done with) as a low-
cost documentation hint, but do not expect a measurable speedup
in a single-kernel harness.
**Follow-up probe**: `80-experience/hw-probes/cache-hint-contended/`
(open).
**Source**: Level-3 sandbox verification (2026-04-04).

## P12. `__ldcv` is 2.26× slower than default with no compensating benefit (NEW — measured)

**Symptom (measured on H200 sm_9.0a, 2026-04-23)**:
`__ldcv` at L2-resident 16-pass regime runs at 2673 GB/s vs
default's 6044 GB/s — **the same ~44% of default** as `__ldcg`.
**Cause**: `__ldcv` always re-fetches (PTX `ld.global.cv` bypasses
the cache tag check). On H200's L1/L2 hierarchy, the measured
wall-clock behaves as if L1 is bypassed; whether L2 is also
bypassed is not distinguishable by wall-clock (both collapse to
the same BW). Either way, **there is no performance use case for
`__ldcv`**.
**Fix**: Use `__ldcv` only for **correctness** — when polling a
flag that another thread or kernel writes to and you need to see
updated values. Never as a performance knob, never "to force a
fresh read".
**Source**: Discovered by the H200 probe 2026-04-23 (not in
legacy KB).

## P13. Template-load-variant pitfall when mixing const-qualification (NEW — probe-surfaced)

**Symptom**: When switching a load site between default `*p` and
`__ldg(p)` inside the same kernel template, the compiler may emit
*different* cache operators than expected because `const` /
`__restrict__` inference differs between the two paths.
**Detection**: `nvcc -ptx` on each template specialization; confirm
the expected `ld.global.{nc, ca, cg, cs, lu, cv}` appears.
**Fix**: When writing a benchmark that compares variants, keep the
pointer type identical across all templates (the probe's
`stream_read<V>` does this) and rely on the intrinsic dispatch —
not on qualifier changes — to select the cache operator. This
isolates the cache-operator variable from the const-inference
variable.
**Source**: Probe 2026-04-23 methodology note (no symptom
observed, but preventive discipline).
