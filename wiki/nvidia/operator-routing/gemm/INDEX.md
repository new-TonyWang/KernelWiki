---
title: Tensor-core GEMM Pattern -- Decision Tree
pattern_class: tensor-core
op: gemm
status: draft
hardware:
  device: H200
  sm: 9.0a
source:
- path: wiki/nvidia/techniques/warp-specialization/skill.md
  anchor: Warp-specialized GEMM mainloop (algorithm skeleton)
  excerpt: Plain WS / Pingpong / Cooperative variants; producer/consumer split with
    mbarrier full/empty pair
- path: wiki/nvidia/foundations/compute/gemm-ptx/skill.md
  anchor: Hopper GEMM via raw PTX (cutlass-free)
  excerpt: TMA-PTX + wgmma-PTX composed into a single GEMM kernel; cutlass-free preprocessor
    + linked-binary gates
- path: wiki/nvidia/foundations/compute/gemm/aligned/skill.md
  anchor: Aligned GEMM on Hopper via cutlass cooperative warp-specialized kernel
  excerpt: Cutlass-API track; aligned (M,N,K) divisible by wgmma atom; fast-path mainloop
- path: sources/experience/kernel-records/2026-04-29-gemm-ws-ptx/README.md
  anchor: Cutlass-free warp-specialized GEMM (record)
  excerpt: Producer warp + consumer warpgroup composed from tma-ptx + wgmma-ptx +
    warp-specialization skill
id: routing-gemm-INDEX
type: operator-routing
vendor: nvidia
operator: gemm
---
# Tensor-core GEMM Pattern -- Decision Tree

This document guides the kernel-writing agent through a Hopper tensor-core GEMM task. The decision tree enforces a **library-first** policy and only proceeds to a custom kernel when measurement shows the library path is insufficient. Within "custom", a second tree picks between the cutlass-API track (`wiki/nvidia/foundations/compute/gemm/aligned`) and the cutlass-free PTX track (`wiki/nvidia/foundations/compute/gemm-ptx`).

## Step 0 -- Try the library first

Before writing any custom CUDA / inline-PTX code, check the production paths.

```
Q0. Is the caller's environment PyTorch-based AND the GEMM is a standard
    matmul (no fused epilogue beyond bias/relu/gelu)?
    YES --> Use torch.matmul / torch.mm / torch.nn.functional.linear.
            Backend dispatch (cuBLAS / cuBLASLt / cutlass) is automatic.
            DONE.
    NO  --> Continue to Q1.

Q1. Is the GEMM a plain D = alpha * A @ B + beta * C with A/B in a dtype
    cuBLAS supports (fp32, fp16, bf16, tf32, fp8, int8) and standard
    row/col-major layout?
    YES --> Use cublasGemmEx / cublasLtMatmul.
            See library-fallback.md.  DONE.
    NO  --> Continue to Q2.

Q2. Does the GEMM need a fused epilogue (bias-add + activation + scale +
    optional D-write) with a layout that cuBLASLt's epilogue catalogue
    covers (CUBLASLT_EPILOGUE_BIAS / RELU / GELU / DRELU_BGRAD / etc.)?
    YES --> Use cublasLtMatmul with the matching CUBLASLT_EPILOGUE_*.
            DONE.
    NO  --> Continue to Q3.

Q3. Does the library path fail to meet performance / fusion / dtype
    requirements after benchmarking on the target shape?
    YES --> Proceed to Step 1 (custom kernel selection).
    NO  --> Re-examine the library path. Most production GEMMs are
            well-served by cuBLASLt. Only proceed when measurement
            shows the gap is not closeable by tuning the library call.
```

**When to skip the library**: the library path is insufficient when:
- The fused epilogue is not in the cuBLASLt catalogue (e.g. fused softmax, fused per-row reduction, custom quantization scale / dequant pattern).
- The dtype combo is not supported (e.g. mxfp8 / nvfp4 with custom scale layout that pre-dates cuBLASLt support on the target driver).
- The kernel must compose with a non-GEMM mainloop (e.g. attention's QK^T followed by softmax followed by PV with shared smem stages).
- The measured cuBLASLt latency exceeds the wgmma-issue floor by more than ~10% at the target shape, leaving room for a hand-tuned schedule.

## Step 1 -- Choose the custom track

Two tracks live under this pattern, distinguished by the dependency policy:

```
Q4. Is cutlass + cute available as a build-time and link-time dependency?
    YES --> CUTLASS-API track. See `wiki/nvidia/foundations/compute/gemm/aligned/skill.md`
            (aligned shapes) or `wiki/nvidia/foundations/compute/gemm/non-aligned-tail/skill.md`
            (M / N not multiples of the wgmma atom -- adds a tail-handling block).
            For fused epilogues that need extension beyond the cutlass catalogue,
            see `wiki/nvidia/foundations/compute/gemm-fused/cutlass-epilogue-prologue/skill.md`.
            Continue to Step 2 to pick the warp-spec variant.

    NO  --> CUTLASS-FREE PTX track. The kernel must contain zero
            `cutlass::` / `cute::` symbols at both the preprocessed source
            and linked-binary level. See `wiki/nvidia/foundations/compute/gemm-ptx/skill.md`
            for the single-tile composition; see
            `sources/experience/kernel-records/2026-04-29-gemm-ws-ptx/README.md`
            for a working warp-specialized extension that adds a producer-warp
            + consumer-warpgroup split and a multi-K-tile pipeline.
            Continue to Step 2 to pick the warp-spec variant -- the
            algorithm skeleton in `wiki/nvidia/techniques/warp-specialization`
            is dependency-agnostic.
```

The cutlass-free track exists for one specific reason: **the kernel must be auditable without pulling cute headers** (compliance / supply-chain audit / building a kernel inside a project that cannot accept cutlass as a dependency). Any time cutlass is allowed, the cutlass-API track is the engineering default -- it saves thousands of lines and gets the cooperative + cluster-multicast variants for free.

## Step 2 -- Pick the warp-specialization variant

The Hopper GEMM mainloop is a producer/consumer pipeline regardless of which track. Three variants from `wiki/nvidia/techniques/warp-specialization/skill.md`:

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

After the basic kernel is correct (cutlass-free preprocessor + linked-binary gate, OR cuBLAS numeric gate), apply the optimization skills from `ROUTING.md` in priority order. The hot levers for tensor-core GEMM are:

1. **TMA pipeline depth** -- producer's stage count. 2 is the minimum that overlaps; 3-4 the H200 sweet spot; >4 rarely justified.
2. **Multi-warpgroup consumers** -- pingpong / cooperative variants for problems that amortize their setup.
3. **Cluster-multicast TMA** -- cooperative-only; broadcasts one TMA load to two CTAs.
4. **Persistent kernel scheduling** -- pair with pingpong / cooperative; one CTA processes many output tiles serially, hiding launch overhead.

After each optimization, re-benchmark against `baseline` (cuBLASLt at the same shape) and follow the bottleneck triage in `reasoning/bottleneck-triage.md`.

## Cross-references

- **Library fallback details**: `library-fallback.md`
- **Skill whitelist for this pattern**: `ROUTING.md`
- **Task packet template**: `TASK-PACKET.md`
- **Bottleneck triage after benchmarking**: `reasoning/bottleneck-triage.md`
- **Algorithm skeleton (variant-independent)**: `wiki/nvidia/techniques/warp-specialization/skill.md`
- **Cutlass-free working kernel record**: `sources/experience/kernel-records/2026-04-29-gemm-ws-ptx/`
