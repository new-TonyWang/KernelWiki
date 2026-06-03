---
title: L2 Access Policy (persisting window + hitRatio tuning)
status: verified
evidence_level: measured
cuda_version_tested: '12.9'
driver_version_tested: 570.124.06
toolchain: nvcc 12.9.86 + ptxas 12.9
measured_on: H200-SXM
applies_to_pattern_class:
- cuda-core
applies_to_ops:
- reduction
- indexing
- normalization
- elementwise
requires_sm: '>=8.0'
requires_features:
- l2-set-aside
- access-policy-window
single_kernel_useful: false
multi_kernel_useful: true
source:
- path: spec
  anchor: Reference
artifacts:
  code: artifacts/experience/hw-probes/l2-residency/l2_residency_probe.cu
  build: artifacts/experience/hw-probes/l2-residency/build.sh
  introspection: artifacts/experience/hw-probes/l2-residency/device.json
  profile: ''
related_apis:
- cudaStreamSetAttribute
- cudaStreamAttributeAccessPolicyWindow
- cudaAccessPolicyWindow
- cudaAccessPropertyPersisting
- cudaAccessPropertyStreaming
- cudaDeviceSetLimit
- cudaLimitPersistingL2CacheSize
- cudaCtxResetPersistingL2Cache
- cudaGraphKernelNodeSetAttribute
- cudaKernelNodeAttributeAccessPolicyWindow
- cudaLaunchAttributeAccessPolicyWindow
related_skills:
- cache-load-hints
- async-copy
- coalescing
experience_refs:
- sources/experience/hw-probes/l2-residency.md
id: skill-l2-access-policy
type: skill
vendor: nvidia
tags:
- cuda-cpp
applies_to:
- general
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L4094-L4130
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L4060-L4092
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1561-L1581
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L4132-L4168
---
## What

An **L2 access policy window** tells the hardware that a specific region of global memory (identified by `base_ptr` + `num_bytes`) should be treated differently from the rest of the address space when it passes through the L2 cache. The two knobs of interest are:

- **`hitProp = cudaAccessPropertyPersisting`** — tag these lines as *sticky*; when eviction is needed, prefer to evict non-persisting lines instead.
- **`missProp = cudaAccessPropertyStreaming`** — tag misses in the window as streaming (evict-first) so they do not pollute L2 for other workloads.
- **`hitRatio ∈ [0, 1]`** — the fraction of accesses tagged with `hitProp`; the rest fall back to default caching. The purpose is **not** to hit less, but to *shrink the effective persisting footprint* when `num_bytes` exceeds the hardware set-aside.

Two runtime prerequisites:

1. **Reserve a set-aside** via `cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, bytes)`. The maximum is `cudaDeviceProp::persistingL2CacheMaxSize`. On H200 the maximum is **37.5 MiB** (62.5% of the 60 MiB L2).
2. **Apply the window** on a stream, graph node, or launch config. The three entry points are `cudaStreamSetAttribute(..., cudaStreamAttributeAccessPolicyWindow, ...)`, `cudaGraphKernelNodeSetAttribute(..., cudaKernelNodeAttributeAccessPolicyWindow, ...)`, and the `cudaLaunchConfig_t::attrs` array consumed by `cudaLaunchKernelEx`.

The window does **not** change what a kernel's load instructions *emit*; it is a runtime-level policy bit in the L2 controller. Kernel source stays identical. This distinguishes the skill from **`cache-load-hints`**, which operates on individual load/store instructions (`__ldcs`, `__ldca`, `.cg`, `.cs`).

## Why

H200's L2 is **60 MiB with a 37.5 MiB persisting set-aside**. Any working set that repeatedly fits in 37.5 MiB and must survive eviction pressure from *other* concurrent work (other streams, graph nodes, different parts of a graph) is a candidate for the policy. Without competing pressure, normal LRU replacement already keeps the hot working set in L2 for the duration of a single kernel; the policy then has no benefit and may add a small policy-machinery overhead (measured: **negligible at WS = 4 MiB / 40 MiB**).

Where the policy **is** measurably useful is when the hot working set **exceeds** the set-aside. In that regime, `hitRatio = 1.0` silently fails (no speedup — the window tries to pin more than fits, so the hardware rotates pinned lines and nothing net-sticks), but `hitRatio = set_aside / WS` stochastically pins a fitting subset and recovers measurable bandwidth. Measured on H200 at WS = 80 MiB / set_aside = 37.5 MiB: **+17.7% effective BW** vs no policy.

The skill therefore is not "always apply a persisting window on hot data" — it is a specific two-regime tool with a narrow sweet spot.

## When to use

### S1. Reserve set-aside before touching the window

```cuda
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);

size_t set_aside = std::min<size_t>(
    static_cast<size_t>(prop.l2CacheSize * 0.75),
    prop.persistingL2CacheMaxSize);
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, set_aside);
```

Skipping `cudaDeviceSetLimit` leaves set-aside at 0 on most platforms; the window then applies to a zero-byte set-aside and nothing persists. **This is a silent failure** — see pitfall P3.

### S2. Tune `hitRatio` to `set_aside / WS` when WS exceeds set-aside

This is the measured-good regime. When the hot buffer is larger than the set-aside, let hardware pin only the fraction that fits:

```cuda
cudaStreamAttrValue attr = {};
attr.accessPolicyWindow.base_ptr  = hot_data;
attr.accessPolicyWindow.num_bytes = std::min<size_t>(hot_bytes,
                                                     prop.accessPolicyMaxWindowSize);
attr.accessPolicyWindow.hitRatio  = std::min(1.0f,
    static_cast<float>(set_aside) / static_cast<float>(hot_bytes));
attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);

// launch the hot-region kernels on `stream` here
```

Measured result on H200 (WS = 80 MiB, set_aside = 37.5 MiB, hitRatio ≈ 0.469): median launch time drops from 1.870 ms to 1.589 ms, a **+17.7%** speedup. See sources/experience/hw-probes/l2-residency/.

### S3. Reset persisting lines between phases

When a sequence of kernels moves to a different hot region, the stale persisting lines from the previous phase will otherwise occupy L2 and hurt the next phase:

```cuda
cudaStreamAttrValue attr = {};
attr.accessPolicyWindow.num_bytes = 0;    // disable the window
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
cudaCtxResetPersistingL2Cache();          // evict persisting lines
```

This is mandatory between distinct persisting regions; see pitfall P2 (legacy, not re-measured in this probe because the single-kernel harness does not exhibit stale-line carry-over).

### S4. Graph-node and launch-config variants

The same `cudaAccessPolicyWindow` struct is consumed by:

- `cudaGraphKernelNodeSetAttribute(node, cudaKernelNodeAttributeAccessPolicyWindow, &nodeAttr)` — per-node policy inside a CUDA Graph.
- `cudaLaunchConfig_t::attrs[]` with `cudaLaunchAttributeAccessPolicyWindow` — per-launch policy via `cudaLaunchKernelEx`.

Semantics are identical to the stream variant; pick whichever matches the submission model. Not re-measured on H200; retained as inferred from PG §4.13.2.

## When NOT to use

- **When the hot working set already fits L2 and there is no competing pressure.** Measured: WS = 4 MiB, 40 MiB — zero benefit across all three policies tested. The hardware LRU already keeps the buffer hot; adding a window is overhead without gain. Legacy pitfall P6 was "dramatically hurts"; on H200 sm_9.0a it is a soft null rather than a hurt — still not worth the code complexity.
- **When WS > set-aside and you use `hitRatio = 1.0`.** Measured on H200 at WS = 80 MiB: **identical wall-clock to no policy**. The call succeeds, the hot region is tagged, but the hardware cannot pin more than fits. This is a silent null — correct the tuning (S2) or drop the window entirely. Legacy P1.
- **On MIG-partitioned GPUs.** `cudaLimitPersistingL2CacheSize` is a no-op; the call succeeds silently with nothing reserved. Legacy P3.
- **On MPS servers you do not administer.** The set-aside is a server-start-time setting; you cannot adjust it at kernel-launch time per process. Legacy P3 (MPS variant).
- **In single-kernel MVPs** — by construction, there is no competing kernel to create the eviction pressure that this skill resists. Measured: the WS = 40 MiB case is a soft null because nothing else evicts the buffer during the 32-repeat read loop. See the probe's "WS=40 MiB is a soft-null" open question.

## Measured Characteristics

Measured on H200-SXM (sm_9.0a, CUDA 12.9, driver 570.124.06) using sources/experience/hw-probes/l2-residency/ — a `repeat_read_sum` kernel with N_REPEATS=32 inner passes, swept over working-set sizes {4, 40, 80} MiB and policies {none, persist@1.0, persist@tuned}. Full record: sources/experience/hw-probes/l2-residency.md.

### Effective bandwidth by (WS, policy)

| WS     | none       | persist@1.0 | persist@tuned | tuned speedup |
| ------ | ---------: | ----------: | ------------: | ------------: |
|  4 MiB | 4524 GB/s  |  4524 GB/s  |    4534 GB/s  |         1.00× |
| 40 MiB | 4357 GB/s  |  4355 GB/s  |    4360 GB/s  |         1.00× |
| 80 MiB | 1436 GB/s  |  1436 GB/s  |    **1690 GB/s** |     **1.18×** |

H200 hardware limits (`cudaGetDeviceProperties`):

| Field                         | Value     |
| ----------------------------- | --------: |
| `l2CacheSize`                 |   60 MiB  |
| `persistingL2CacheMaxSize`    | 37.5 MiB  |
| `accessPolicyMaxWindowSize`   |  128 MiB  |

Key measured findings:

- **The policy is a null below the set-aside.** Zero measurable benefit for WS = 4 MiB and WS = 40 MiB. The skill's decision tree must check `WS > set_aside` first and only then consider the window.
- **`hitRatio = 1.0` thrashes silently at WS > set-aside.** Same wall-clock as no policy (legacy P1 upgraded to measured on H200; the legacy "dramatic slowdown" is softened to "silent null").
- **`hitRatio = set_aside / WS` gives +17.7% at WS = 80 MiB** on H200. This is the skill's one measurably good recommendation and the probe's load-bearing result.
- **H200's persisting set-aside cap is 37.5 MiB** — only 62.5% of the 60 MiB L2. The fraction is hardware-fixed and not user-tunable above that cap.

The probe's WS = 40 MiB result is a **soft null** because the harness has no competing memory pressure. In a multi-kernel or multi-stream context the policy should still pin this buffer even below the set-aside limit. Follow-up probe `sources/experience/hw-probes/l2-residency-contended/` is open to measure that regime.

## Principles

1. **The policy is a *pinning* tool, not a *caching* tool.** It does not make L2 faster; it tells L2 which lines to keep under pressure. If there is no pressure, there is no benefit.
2. **Tune `hitRatio` to the set-aside, not to the working set.** The measured formula `hitRatio = min(1, set_aside / WS)` is the one actually documented by PG §4.13.3 (L4094-L4130) and the only one that recovers bandwidth when WS exceeds the set-aside.
3. **Reset persisting lines between phases.** Stale pins are worse than no policy (legacy P2). Always disable the window and call `cudaCtxResetPersistingL2Cache` between distinct hot regions.
4. **Graph-node / launch-config variants are equivalent to the stream variant.** Pick the entry point that matches the submission model; don't mix.

## Open questions

- Q1. Does a **concurrent second stream** touching non-window data reveal the window's pinning effect at WS = 40 MiB? The current probe is single-kernel; follow-up `sources/experience/hw-probes/l2-residency-contended/` (open) is the minimal shape to show this.
- Q2. Is the **CUDA-Graph node attribute** exactly equivalent to the stream attribute on H200? PG §4.13.2 says yes; not re-measured.
- Q3. What is the **PTX-level `.L2::cache_hint` + `createpolicy`** behavior relative to the runtime-level window, and do they compose (stack) or override? Not measured in this pass; belongs in the `cache-load-hints` skill (cache-load-hints) follow-up.
- Q4. Does the set-aside reservation **persist across CUDA contexts** on H200, or is it reset at context destruction? PG implies per-context; unverified.

## Legacy references

- `legacy_sandbox_path`: `corpus/nvidia/legacy-optimization/memory/l2-cache-control/skill.md`. Legacy kept five sub-skills (S1 set up window → this S1+S2, S2 tune hitRatio → S2, S3 reset → S3, S4 graph node → S4, S5 query props → folded into S1).
- Legacy pitfalls P1 (thrashing) and P6 (policy on already-cached data) are upgraded from legacy-anecdotal to measured, with the qualitative claim *softened* on H200: both degenerate to silent nulls rather than catastrophic slowdowns. Full audit in `pitfalls.md`.
- **Related but distinct**: `wiki/nvidia/foundations/memory/cache-load-hints/` (pending) operates at the instruction level (`__ldcs`, `__ldca`, `.cg`/`.cs` PTX operators); this skill operates at the runtime-attribute level. The two are complementary — you can apply a persisting window **and** issue `__ldcs` loads inside the kernel, and the policies compose.
