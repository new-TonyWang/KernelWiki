---
title: Tensor-core GEMM Pattern -- Decision Tree
pattern_class: tensor-core
op: gemm
status: draft
hardware:
  device: H200
  sm: 9.0a
source:
- path: wiki/nvidia/techniques/warp-specialization.md
  anchor: Warp-specialized GEMM mainloop (algorithm skeleton)
  excerpt: Plain WS / Pingpong / Cooperative variants; producer/consumer split with mbarrier full/empty pair
- path: wiki/nvidia/foundations/compute/gemm-ptx.md
  anchor: Hopper GEMM via raw PTX (cutlass-free)
  excerpt: TMA-PTX + wgmma-PTX composed into a single GEMM kernel; cutlass-free preprocessor + linked-binary gates
- path: wiki/nvidia/foundations/compute/gemm.md
  anchor: Aligned GEMM on Hopper via cutlass cooperative warp-specialized kernel
  excerpt: Cutlass-API track; aligned (M,N,K) divisible by wgmma atom; fast-path mainloop
- path: sources/experience/kernel-records/2026-04-29-gemm-ws-ptx.md
  anchor: Cutlass-free warp-specialized GEMM (record)
  excerpt: Producer warp + consumer warpgroup composed from tma-ptx + wgmma-ptx + warp-specialization skill
id: routing-gemm-INDEX
type: operator-routing
vendor: nvidia
operator: gemm
architectures:
- sm90
- sm90a
- sm100
languages:
- ptx
- cuda-cpp
- cute-dsl
hardware_features:
- wgmma
- tma
- mbarrier
- cluster
- fp8
- nvfp4
techniques:
- warp-specialization
- persistent-kernel
- pipeline-stages
- kernel-fusion
- shared-memory-optimization
- tma-multicast
- software-exp
kernel_types:
- gemm
- attention
- fused-kernel
- quantization
confidence: inferred
tags:
- wgmma
- tma
- mbarrier
- cluster
- fp8
- nvfp4
- warp-specialization
- persistent-kernel
- pipeline-stages
- kernel-fusion
- shared-memory-optimization
- tma-multicast
- software-exp
- gemm
- attention
- fused-kernel
- quantization
- ptx
- cuda-cpp
- cute-dsl
---
# Tensor-core GEMM Pattern -- Decision Tree

This document guides the kernel-writing agent through a Hopper tensor-core GEMM task that requires a dedicated kernel implementation. It picks between the cutlass-API track (`wiki/nvidia/foundations/compute/gemm/aligned`) and the cutlass-free PTX track (`wiki/nvidia/foundations/compute/gemm-ptx`).

## Scope

This decision tree covers custom-kernel implementation choices only. It starts after the task has been classified as requiring a dedicated kernel implementation.

## Step 1 -- Choose the custom track

Two tracks live under this pattern, distinguished by the dependency policy:

```
Q4. Is cutlass + cute available as a build-time and link-time dependency?
    YES --> CUTLASS-API track. See `wiki/nvidia/foundations/compute/gemm.md`
            (aligned shapes) or `wiki/nvidia/foundations/compute/gemm/non-aligned-tail/skill.md`
            (M / N not multiples of the wgmma atom -- adds a tail-handling block).
            For fused epilogues that need extension beyond the cutlass catalogue,
            see `wiki/nvidia/foundations/compute/gemm-fused/cutlass-epilogue-prologue/skill.md`.
            Continue to Step 2 to pick the warp-spec variant.

    NO  --> CUTLASS-FREE PTX track. The kernel must contain zero
            `cutlass::` / `cute::` symbols at both the preprocessed source
            and linked-binary level. See `wiki/nvidia/foundations/compute/gemm-ptx.md`
            for the single-tile composition; see
            `sources/experience/kernel-records/2026-04-29-gemm-ws-ptx.md`
            for a working warp-specialized extension that adds a producer-warp
            + consumer-warpgroup split and a multi-K-tile pipeline.
            Continue to Step 2 to pick the warp-spec variant -- the
            algorithm skeleton in `wiki/nvidia/techniques/warp-specialization`
            is dependency-agnostic.
```

The cutlass-free track exists for one specific reason: **the kernel must be auditable without pulling cute headers** (compliance / supply-chain audit / building a kernel inside a project that cannot accept cutlass as a dependency). Any time cutlass is allowed, the cutlass-API track is the engineering default -- it saves thousands of lines and gets the cooperative + cluster-multicast variants for free.

## Step 2 -- Pick the warp-specialization variant

The Hopper GEMM mainloop is a producer/consumer pipeline regardless of which track. Three variants from `wiki/nvidia/techniques/warp-specialization.md`:

```
Q5. How large is the M*N grid relative to the H200 SM count (132)?

    Small (problem fits in <= a few SM-fulls of CTAs):
        --> Plain WS. 1 producer warp + 1 consumer warpgroup per CTA.
            Cluster `<1,1,1>`, no persistence. Lowest coordination cost
            and the leader at small problems (measured 2048^3 H200:
            195.4 TFLOPS plain WS vs 186.9 TFLOPS cooperative).

    Medium (uniform per-tile K, room to alternate work):
        --> Pingpong. 1 producer + 2 alternating consumer warpgroups.
            Persistent. Even-k tiles to consumer-A, odd-k to consumer-B.
            Requires UNIFORM K across all tiles (WS pitfall #4).

    Large (problem is many cluster-grids worth; per-tile work is heavy):
        --> Cooperative + cluster-multicast.
            1 producer + 2 cooperating consumers; each consumer takes
            half the wgmma rows of the same tile. REQUIRES cluster
            >= <2,1,1> AND the multicast variant of the bulk-tensor
            mnemonic (`cp.async.bulk.tensor.shared::cluster.global
            .tile.bulk_group.cta_group::2`). Without multicast,
            cooperative is strictly worse than plain WS (pitfall #5).
            Cooperative wins at 8192^3 (292.7 TFLOPS measured).
```

The minimum-viable variant is **plain WS**. The cutlass-free working artifact in `sources/experience/kernel-records/2026-04-29-gemm-ws-ptx/` realizes plain WS at one CTA + 5 warps (4 consumer + 1 producer); pingpong / cooperative are mechanical extensions of that scaffolding.

## Step 3 -- Optimization via ROUTING.md skills

After the basic kernel is correct (cutlass-free preprocessor + linked-binary gate, OR task numeric gate), apply the optimization skills from `ROUTING.md` in priority order. The hot levers for tensor-core GEMM are:

1. **TMA pipeline depth** -- producer's stage count. 2 is the minimum that overlaps; 3-4 the H200 sweet spot; >4 rarely justified.
2. **Multi-warpgroup consumers** -- pingpong / cooperative variants for problems that amortize their setup.
3. **Cluster-multicast TMA** -- cooperative-only; broadcasts one TMA load to two CTAs.
4. **Persistent kernel scheduling** -- pair with pingpong / cooperative; one CTA processes many output tiles serially, hiding launch overhead.

After each optimization, re-benchmark against the task-provided baseline and follow the bottleneck triage in `reasoning/bottleneck-triage.md`.

## Cross-references

- **Skill whitelist for this pattern**: `ROUTING.md`
- **Task packet template**: `TASK-PACKET.md`
- **Bottleneck triage after benchmarking**: `reasoning/bottleneck-triage.md`
- **Algorithm skeleton (variant-independent)**: `wiki/nvidia/techniques/warp-specialization.md`
- **Cutlass-free working kernel record**: `sources/experience/kernel-records/2026-04-29-gemm-ws-ptx/`
