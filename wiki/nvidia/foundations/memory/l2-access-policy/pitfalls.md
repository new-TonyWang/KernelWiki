---
title: L2 Access Policy — Pitfalls
status: verified
related_skill: wiki/nvidia/foundations/memory/l2-access-policy/skill.md
source:
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L4094-L4170
  excerpt: hitRatio tuning, cudaCtxResetPersistingL2Cache semantics, MIG / MPS caveats.
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1561-L1610
  excerpt: BP §10.2.2 usage guidance + warnings.
experience_refs:
- sources/experience/hw-probes/l2-residency/2026-04-23-l2-residency.md
id: pitfall-l2-access-policy
type: pitfall
vendor: nvidia
---
Legacy P1–P5 came from L3-sandbox verification on pre-H200 hardware (A100 unless noted). P6–P10 are L3-sandbox findings copied from the legacy KB and annotated with the current H200 measurement where the probe touches them. "**Measured on H200 sm_9.0a**" tags indicate pitfalls validated by sources/experience/hw-probes/l2-residency/.

## P1. `hitRatio = 1.0` thrashes silently when `num_bytes > set_aside`

**Symptom (legacy, A100)**: Performance drops ~10 % when the persistent data region exceeds L2 set-aside size with `hitRatio=1.0`. **Symptom (measured on H200 sm_9.0a, 2026-04-23)**: wall-clock is **identical** to no policy (WS=80 MiB / set_aside=37.5 MiB: 1.8694 ms both). The H200 L2 controller degrades gracefully to "nothing net-sticks" rather than evicting into DRAM. This is a **silent null**, not a catastrophic regression — but still the wrong tuning. **Detection**: measure wall-clock with and without the window; if they're equal, the window is a no-op. NCU L2 hit-rate metrics are unreliable here (see probe record — NCU replay artifacts). **Fix**: set `hitRatio = min(1.0, set_aside / WS)`. Measured: this recovers +17.7 % at WS=80 MiB on H200. **Source**: PG §4.13.3 (L4094-L4130); probe 2026-04-23.

## P2. Stale persisting lines reduce L2 for subsequent kernels

**Symptom**: Kernel B runs slower after Kernel A used L2 persistence, even though Kernel B has its own data. **Detection**: L2 hit rate for Kernel B is lower than baseline. Persisting lines from Kernel A still occupy set-aside slots. **Fix**:

```cuda
cudaStreamAttrValue attr = {};
attr.accessPolicyWindow.num_bytes = 0;
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
cudaCtxResetPersistingL2Cache();
```

**Not re-measured on H200** — the current probe is single-kernel so stale carry-over does not occur in the harness. Retained as legacy- anecdotal pending a multi-kernel follow-up probe. **Source**: PG §4.13.5 (reset L2 access to normal).

## P3. MIG / MPS silently disables the set-aside

**Symptom**: `cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, n)` returns `cudaSuccess` but nothing is reserved; subsequent policies never persist. **Detection**: `cudaDeviceGetLimit(&v, cudaLimitPersistingL2CacheSize)` returns 0 after the set call, **or** `nvidia-smi -i 0 -q | grep MIG` confirms MIG mode. **Fix**: L2 set-aside is not available in MIG or on MPS servers you do not administer. Fall back to per-instruction cache hints via the `cache-load-hints` skill (pending). On MPS, the set-aside is a server-start-time option controlled by the administrator. **Source**: PG §4.13.1 (L2 Cache Set-Aside for Persisting Accesses).

## P4. Concurrent streams compete for the set-aside

**Symptom**: Two concurrent streams with persisting windows evict each other's data. **Detection**: Both kernels show lower L2 hit rates than when run alone. **Fix**: Reduce `hitRatio` for each stream (e.g. 0.5 and 0.5 for two equal streams) so the **sum** of pinned footprints fits the set-aside. Alternatively, schedule them serially on one stream. **Not re-measured on H200** — the current probe is single-stream. Follow-up: `sources/experience/hw-probes/l2-residency-contended/` (open). **Source**: PG §4.13.6 (Manage Utilization of L2 Set-Aside Cache).

## P5. `num_bytes` exceeds `accessPolicyMaxWindowSize`

**Symptom**: Values of `num_bytes` beyond the device limit are silently clamped; effective window is smaller than requested. **Detection**: Compare `attr.accessPolicyWindow.num_bytes` against `cudaDeviceProp::accessPolicyMaxWindowSize`. H200: 128 MiB. **Fix**: `num_bytes = std::min(desired, prop.accessPolicyMaxWindowSize);`. For a hot region larger than 128 MiB, window covers only a prefix; the remaining tail uses default caching — this is almost never what the caller intends. **Source**: PG §4.13.7 (Query L2 Cache Properties).

## P6. Applying the policy to already-well-cached data

**Symptom (legacy, pre-H200)**: "Dramatically increases DRAM traffic and hurts performance." **Symptom (measured on H200 sm_9.0a, 2026-04-23)**: Downgraded to a **soft null** — WS=4 MiB and WS=40 MiB cases all land within 0.2 % of no-policy wall-clock (4524/4357/4360 GB/s effective). The H200 L2 controller does **not** punish the attempt; it just wastes a few cycles on policy metadata. Net measured impact: essentially 0. **Fix**: Only apply the window when WS approaches or exceeds the set-aside **and** other memory pressure exists. If the hot region is small enough that normal LRU keeps it resident, skip the window. **Source**: Level-3 sandbox verification (2026-04-05); downgraded on H200 by probe 2026-04-23.

## P7. L2 policy on a bandwidth-saturated kernel

**Symptom (legacy)**: "Kernels already at peak memory bandwidth may see single-kernel latency hurt slightly by policy enforcement overhead." **Symptom (measured on H200 sm_9.0a, 2026-04-23)**: Not observed. The WS=40 MiB case runs at 4357 GB/s regardless of policy — no measurable overhead. Legacy claim retained for completeness but demoted from "measured" to "inferred" on H200. **Fix**: If wall-clock noise floor hides your expected benefit, you do not have a qualifying use case; drop the window. **Source**: Level-3 sandbox verification (2026-04-05).

## P8. Single-kernel microbench cannot demonstrate the "pinning" benefit

**Symptom**: Probe or benchmark shows no wall-clock difference between "policy on" and "policy off" even when WS ≤ set_aside. **Detection**: Your harness has no concurrent workload to create eviction pressure on the hot buffer. LRU already keeps it resident for the duration of a single kernel. **Fix**: Either (a) accept that the window is a pure insurance policy in single-kernel mode (no measurable benefit but no cost either below set-aside), or (b) design a multi-kernel harness that touches competing memory between passes. Follow-up probe `sources/experience/hw-probes/l2-residency-contended/` (open). **Measured on H200 sm_9.0a**: WS=40 MiB soft-null directly confirms this pitfall — legacy P8 upgraded from inferred to measured. **Source**: Level-3 sandbox verification (2026-04-05); probe 2026-04-23.

## P9. `num_bytes` exceeding the reservable set-aside silently no-ops

**Symptom (legacy)**: "API succeeds but no persistence actually occurs." **Measured context on H200 sm_9.0a**: With WS=80 MiB and set_aside=37.5 MiB (H200 cap), `hitRatio=1.0` window ≈ `none` wall-clock — confirming the "silent no-op at hitRatio=1.0 when the region overflows set-aside" variant. The fix is P1's — use tuned hitRatio. **Fix**: Same as P1 (tuned hitRatio). Equivalent to P1 under a different phrasing; kept separate from legacy P1 because legacy P1 focused on the *performance loss* and legacy P9 on the *silent success* — the fix is the same but the diagnostic is different. **Source**: Level-3 sandbox verification (2026-04-05); probe 2026-04-23.

## P10. "This skill is just runtime config — not a kernel optimization"

**Symptom (legacy)**: Verification agent initially categorized the skill as "category error — kernel code doesn't change." **Resolution**: The observation is correct mechanically (kernel source is untouched) but misleading as a verdict. The policy *does* change observable wall-clock bandwidth when applied in the right regime — measured +17.7 % at WS=80 MiB on H200. Treat this skill as a **runtime-level tuning** tool, not a kernel-code optimization; both belong in the perf engineer's toolbox. **Source**: Level-3 sandbox verification (2026-04-05); probe 2026-04-23.

## P11. Forgetting to reserve the set-aside (NEW — discovered during this probe)

**Symptom**: `cudaStreamSetAttribute` with `hitProp=cudaAccessPropertyPersisting` + `hitRatio=1.0` returns `cudaSuccess`, but the policy has zero effect at any WS. **Cause**: The default set-aside on most platforms is **0 bytes** until `cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, n)` is called. Until then, "persisting" has no capacity to persist into. **Detection**: `cudaDeviceGetLimit(&v, cudaLimitPersistingL2CacheSize); assert(v > 0);`. Skipping this check is the most common "my window doesn't do anything" symptom on a correctly-configured H200. **Fix**: Always call `cudaDeviceSetLimit` before the first stream setup. Probe's pattern:

```cuda
cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
size_t set_aside = std::min<size_t>(
    (size_t)(prop.l2CacheSize * 0.75),
    prop.persistingL2CacheMaxSize);
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, set_aside);
```

**Source**: Discovered while writing the probe harness 2026-04-23. PG §4.13.1 documents the reservation step but does not call out the default-zero behavior as a distinct failure mode — this pitfall makes it explicit.
