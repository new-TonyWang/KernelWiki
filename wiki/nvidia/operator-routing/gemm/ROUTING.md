---
title: Tensor-core GEMM Pattern -- Skill Routing
pattern_class: tensor-core
op: gemm
status: draft
id: routing-gemm-ROUTING
type: operator-routing
vendor: nvidia
operator: gemm
architectures:
- sm90
- sm90a
languages:
- ptx
- cuda-cpp
- cute-dsl
hardware_features:
- wgmma
- tma
- mbarrier
- cluster
techniques:
- warp-specialization
- persistent-kernel
- pipeline-stages
- vectorized-loads
- register-budgeting
- kernel-fusion
- shared-memory-optimization
- swizzling
- tma-multicast
kernel_types:
- gemm
- fused-kernel
- quantization
confidence: inferred
tags:
- wgmma
- tma
- mbarrier
- cluster
- warp-specialization
- persistent-kernel
- pipeline-stages
- vectorized-loads
- register-budgeting
- kernel-fusion
- shared-memory-optimization
- swizzling
- tma-multicast
- gemm
- fused-kernel
- quantization
- ptx
- cuda-cpp
- cute-dsl
---
# Tensor-core GEMM Pattern -- Skill Whitelist

This file lists the optimization skills applicable to a Hopper tensor-core GEMM kernel, in recommended application order. Only skills with a completed `skill.md` under `wiki/nvidia/foundations/`, `wiki/nvidia/hardware/`, or `wiki/nvidia/techniques/` are listed.

The pattern has two implementation tracks (see `INDEX.md` Step 1):

- **Cutlass-API track** -- consumes cutlass + cute as a header dependency.
- **Cutlass-free PTX track** -- contains zero `cutlass::` / `cute::` symbols at both the preprocessor and linked-binary level.

Each entry below marks which track(s) it applies to. Track-agnostic skills (most algorithm-level patterns) apply to both.

## Applicable skills

### 1. Warp-specialization mainloop (primary, track-agnostic)

- **Skill path**: `wiki/nvidia/techniques/warp-specialization/`
- **Why it matters for GEMM**: hides TMA latency behind wgmma compute by pinning the TMA issuer to a separate warp / warpgroup. The producer/consumer split is the canonical Hopper GEMM mainloop; serial (non-WS) interleaving stalls every iteration on the next TMA tile. Cutlass realizes this as `MainloopSm90TmaGmmaWarpSpecialized*`; the cutlass-free realization composes the producer side from skill 6 and the consumer side from skill 7.
- **When to apply**: every Hopper GEMM. Variant choice (plain / pingpong / cooperative) follows `INDEX.md` Step 2; plain is the safe default below ~a few cluster-grids of CTAs in M*N.
- **Tracks**: both.
- **Measured on H200**: +9.2 % TFLOPS at 2048^3 vs serial baseline (178.9 -> 195.4 TFLOPS), bit-identical correctness; cooperative + multicast climbs to 292.7 TFLOPS at 8192^3.
- **Relevance to bottleneck triage**: Q4 in `reasoning/bottleneck-triage.md` -- "wgmma issue port stalls every iteration on TMA load completion."

### 2. Persistent kernel scheduling (primary, pair with WS)

- **Skill path**: `wiki/nvidia/techniques/persistent-kernel/`
- **Why it matters for GEMM**: a persistent CTA processes many output tiles serially, amortizing kernel-launch cost and stabilizing wgmma issue-port utilization across tile boundaries. Pingpong and cooperative warp-spec variants are persistent by design.
- **When to apply**: pair with pingpong / cooperative (skill 1's medium / large branches). Plain WS does not persist; for small problems the launch-overhead amortization is the wrong trade.
- **Tracks**: both.
- **Relevance to bottleneck triage**: Q4 in `reasoning/bottleneck-triage.md` -- "Per-tile launch overhead visible in nsys timeline as repeated short kernels."

### 3. Aligned GEMM (primary, cutlass-API baseline)

- **Skill path**: `wiki/nvidia/foundations/compute/gemm/aligned/`
- **Why it matters for GEMM**: the cutlass-API canonical kernel for shapes where (M, N, K) are exact multiples of the wgmma atom (M=64, N in {8, 16, 32, 64, 128, 256}, K=8 / 16 / 32 by dtype). No tail / no predicate / no padding -- the fast-path mainloop runs every iteration.
- **When to apply**: every cutlass-API GEMM with aligned shapes; the starting point before reaching for the more specialized skills below.
- **Tracks**: cutlass-API.
- **Relevance to bottleneck triage**: not a triage-driven skill; this is the *baseline* the triage tree's tensor-core branch returns to first.

### 4. GEMM with non-aligned tail (primary, cutlass-API)

- **Skill path**: `wiki/nvidia/foundations/compute/gemm/non-aligned-tail/`
- **Why it matters for GEMM**: extends skill 3 with a tail-handling block for shapes where M or N is not a multiple of the wgmma atom. Adds masking + a smaller atom for the boundary tile so the fast path still runs on the bulk of the grid.
- **When to apply**: cutlass-API GEMM where the caller cannot pad to the alignment boundary upstream (memory budget, layout shared with another op, etc.).
- **Tracks**: cutlass-API.
- **Relevance to bottleneck triage**: Q4 in `reasoning/bottleneck-triage.md` -- "M or N not divisible by atom; current path either pads (wastes bandwidth) or falls back to scalar tail (slow)."

### 5. Cutlass-free GEMM via raw PTX (primary, cutlass-free)

- **Skill path**: `wiki/nvidia/foundations/compute/gemm-ptx/`
- **Why it matters for GEMM**: composes skill 6 (`tma-ptx`) and skill 7 (`wgmma-ptx`) into a single GEMM kernel that contains zero `cutlass::` / `cute::` symbols. Verified by `nvcc -E | grep` and `cuobjdump --dump-elf-symbols | grep`. The single-tile starting point for the cutlass-free track; the warp-specialized extension lives at `sources/experience/kernel-records/2026-04-29-gemm-ws-ptx/`.
- **When to apply**: cutlass-free track only. Required when the project cannot accept cutlass as a build-time or link-time dependency.
- **Tracks**: cutlass-free.
- **Status as of writing**: structural cutlass-free gates pass (0 `cutlass::` / `cute::` symbols); numeric correctness vs cuBLAS at the smallest shape has a residual 497-501 / 512 mismatches traced to the per-thread fragment-store mapping (skill's `pitfalls.md` #1). The WS extension inherits the same residual.
- **Relevance to bottleneck triage**: not a triage-driven skill; this is the *baseline* the triage tree's cutlass-free branch returns to first.

### 6. TMA producer (primary)

- **Skill paths**:
  - cutlass-API: `wiki/nvidia/hardware/tma/`
  - cutlass-free: `wiki/nvidia/hardware/tma-ptx/`
- **Why it matters for GEMM**: the producer warp's load mechanism. `cuTensorMapEncodeTiled` builds the descriptor on the host; `cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes` issues the load on the device; the mbarrier protocol synchronizes producer with consumer. Without TMA, the producer side of skill 1 has nothing to issue and the WS pattern does not apply.
- **When to apply**: every Hopper GEMM; the producer side of skill 1.
- **Tracks**: both -- pick the matching path by track.
- **Specific guidance for GEMM**:
  - Feeding wgmma: `SWIZZLE_128B`, fast-axis = 64 bf16 / 32 fp32 (= 128 B), `box_rows >= 64`, pipeline depth >= 4.
  - mbarrier `expected_tx` must equal `(BOX_M*BOX_K + BOX_K*BOX_N) * sizeof(elem)` (the *sum* of A and B tile bytes for one stage), not just one of the two -- WS pitfall #2.
  - Phase tracking on each mbarrier alternates 0,1,0,1,... per stage cycle: consumer wait phase = `(k / STAGES) & 1`; producer wait phase on bar_empty = `((k - STAGES) / STAGES) & 1` -- WS pitfall #3.
- **Measured on H200** (cutlass-free path): peak 3.72 TB/s = 77.5 % of HBM3e datasheet peak at SWIZZLE_128B + depth=4 + 16 KiB tiles.
- **Relevance to bottleneck triage**: Q4 in `reasoning/bottleneck-triage.md` -- "Stall on `cp.async.bulk.tensor` not overlapped with wgmma compute."

### 7. wgmma consumer (primary)

- **Skill paths**:
  - cutlass-API: `wiki/nvidia/hardware/wgmma/`
  - cutlass-free: `wiki/nvidia/hardware/wgmma-ptx/`
- **Why it matters for GEMM**: the consumer warpgroup's compute mechanism. `wgmma.mma_async.sync.aligned.m64nNkK.<dtype>` is the GEMM-K accumulator step, issued by exactly 128 threads. Without wgmma the consumer side of skill 1 has nothing to issue and the kernel falls back to Ampere mma.sync at far lower throughput.
- **When to apply**: every Hopper GEMM; the consumer side of skill 1.
- **Tracks**: both.
- **Specific guidance for GEMM**:
  - Atom selection: `N=128` maximizes FLOPs per serialized issue (5.32 TFLOPS single-warpgroup); `N=64` minimizes register pressure (4.70 TFLOPS, half the accumulator regs). Dtype is precision-only, not throughput.
  - `wgmma.fence` + `wgmma.commit_group` + `wgmma.wait_group 0` are non-optional in every issue sequence -- wgmma-ptx pitfall #4.
  - Smem descriptor must be 16-byte aligned -- wgmma-ptx pitfall #3.
  - Per-thread fragment count is atom-shape-dependent (4 floats per thread for m64n8k16; 64 for m64n128k8 tf32). Get the accumulator clobber list wrong and ptxas truncates silently -- wgmma-ptx pitfall #5.
  - PTX immediate count varies by dtype family (5 for bf16/fp16, 3 for tf32, 1 for s8) -- wgmma-ptx pitfall #8.
- **Measured on H200**: single-warpgroup serialized plateau ~5.6 TFLOPS bf16/fp16 at any N>=128 (wgmma-ptx pitfall #11). Real GEMM throughput requires multi-warpgroup-per-CTA + multi-accumulator pipelining + multi-CTA scaling, which composes with skills 1, 2, 10, and 11.
- **Relevance to bottleneck triage**: Q4 in `reasoning/bottleneck-triage.md` -- "wgmma single-warpgroup serialized issue plateau visible in ncu as low `sm__inst_executed_pipe_tensor.sustained`."

### 8. Fused epilogue / prologue (secondary, cutlass-API)

- **Skill path**: `wiki/nvidia/foundations/compute/gemm-fused/cutlass-epilogue-prologue/`
- **Why it matters for GEMM**: extends the cutlass GEMM with a custom epilogue (e.g. fused activation, scale, quantize) or prologue (e.g. dequantize from a scaled buffer) when the cuBLASLt epilogue catalogue does not cover the pattern. Avoids the global-memory round-trip of a separate post-GEMM kernel.
- **When to apply**: cutlass-API GEMM where the fused pattern is outside cuBLASLt's `CUBLASLT_EPILOGUE_*` set. Library-first (`INDEX.md` Step 0 Q2) is checked before reaching this skill.
- **Tracks**: cutlass-API.
- **Relevance to bottleneck triage**: Q4 in `reasoning/bottleneck-triage.md` -- "Post-GEMM elementwise kernel re-reads D from DRAM."

### 9. Compiler hints for register / occupancy (secondary)

- **Skill path**: `wiki/nvidia/foundations/compute/compiler-hints/`
- **Why it matters for GEMM**: `__launch_bounds__` controls the register-allocation ceiling, which interacts with wgmma fragment register count and TMA pipeline depth (deeper pipelines = more in-flight smem barriers + descriptors). Setting it incorrectly drops occupancy and serializes wgmma issue.
- **When to apply**: tuning the final kernel after correctness is established. The cutlass-free WS record uses `__launch_bounds__(160, 1)` to bind producer + consumer thread count.
- **Tracks**: both.
- **Relevance to bottleneck triage**: Q4 in `reasoning/bottleneck-triage.md` -- "ncu `achieved_occupancy` low and `register_pressure` high; verify `__launch_bounds__` matches the actual CTA size."

### 10. Instruction-level parallelism via multi-accumulator (secondary)

- **Skill path**: `wiki/nvidia/foundations/compute/ilp/`
- **Why it matters for GEMM**: a single accumulator chain serializes wgmma issues at ~5.6 TFLOPS (wgmma-ptx pitfall #11). Multi-accumulator pipelining splits the K-loop into 2-4 independent chains so consecutive wgmmas do not depend on each other; the issue port stays warm.
- **When to apply**: after warp-spec is in place. Pair with pingpong / cooperative -- the shape that benefits is many-K-tile mainloops where latency hiding inside the K loop matters.
- **Tracks**: both.
- **Relevance to bottleneck triage**: Q4 in `reasoning/bottleneck-triage.md` -- "wgmma issue port underutilized; consecutive `wgmma.wait_group 0` flush-stalls visible in nsys."

### 11. Code-extraction discipline (meta)

- **Skill path**: `wiki/nvidia/foundations/meta/code-extraction/`
- **Why it matters for GEMM**: the cutlass-free track exists by extracting cutlass's PTX wrappers into hand-rolled inline asm. The meta-skill codifies how to cleanly extract a primitive without inheriting the surrounding template machinery.
- **When to apply**: starting any new cutlass-free kernel or porting a cutlass primitive.
- **Tracks**: cutlass-free (primary); cutlass-API (when extending the cutlass codebase, secondary).
- **Relevance to bottleneck triage**: not a triage-driven skill; this is a discipline meta-skill applied at kernel-start time, not at performance-tuning time.

## Skills NOT applicable to tensor-core GEMM

The following skills exist in the KB but are generally not relevant for Hopper tensor-core GEMM kernels:

- **Warp primitives** (`wiki/nvidia/foundations/compute/warp-primitives/`): wgmma is itself the cross-thread reduction primitive for the GEMM K-loop. Adding `__shfl_*` on top of the wgmma fragment is redundant and the m64nNk16 fragment layout is already cross-warp (4 warps) -- a shuffle-based reduce would fight the layout.

- **Reduction skills** (`wiki/nvidia/foundations/sync/atomic-reduction/`): a GEMM kernel's output is a tile, not a scalar; the global-write phase is one store per output element with no contention. Atomic reduction applies to split-K GEMMs (where partial sums from different CTAs are atomically combined), but the split-K combine is itself usually a separate kernel and falls under the reduction pattern, not this one.

- **Memory coalescing** (`wiki/nvidia/foundations/memory/coalescing/`): TMA is the load mechanism on Hopper; coalescing is enforced by the descriptor and the box shape, not by the thread mapping. Skill 6's "fast-axis = 64 bf16" guidance is the equivalent prescription expressed in TMA terms.

- **Bank-conflict avoidance** (`wiki/nvidia/foundations/memory/bank-conflict/`): wgmma's smem descriptor encodes its own swizzle (`SWIZZLE_NONE / 32B / 64B / 128B`); paired with the TMA producer's matching swizzle mode, banks are conflict-free by construction. Hand-rolled `+1` padding does not apply -- wgmma will read the swizzled layout, not a `[N][N+1]` tile.

- **Vectorized access** (`wiki/nvidia/foundations/memory/vectorized-access/`): TMA already issues 128-bit-aligned bulk loads (`cp.async.bulk.tensor`); adding `float4` reinterpretation on top would conflict with the descriptor's element type. Vectorization here is intrinsic to the instruction, not a kernel-level rewrite.

- **Async copy** (`wiki/nvidia/foundations/memory/async-copy/`): `cp.async` (the Ampere primitive) is superseded by TMA on Hopper for tensor-shaped loads. The TMA path is async by design and is what skill 6 covers.

- **Fast math** (`wiki/nvidia/foundations/compute/fast-math/`): GEMM uses the tensor-core MMA path (wgmma); the IEEE-754 vs fast-math distinction does not apply to FMA inside a tensor core. For the epilogue (skill 8) where elementwise activations are added, fast-math may apply -- in that case it is the epilogue kernel's lever, not the GEMM mainloop's.
