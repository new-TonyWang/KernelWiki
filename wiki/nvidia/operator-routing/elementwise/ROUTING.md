---
title: Elementwise Skill Routing
pattern_class: cuda-core
op: elementwise
id: routing-elementwise-ROUTING
type: operator-routing
vendor: nvidia
operator: elementwise
---
# Elementwise -- Skill Whitelist

This file lists the optimization skills applicable to custom elementwise kernels, in priority order. Only skills that exist under `knowledge/30-skill/` are referenced.

---

## Priority 1: coalescing (critical)

- **Skill**: `30-skill/memory/coalescing/`
- **Why**: Elementwise kernels are purely memory-bound. Coalescing is the single most important optimization. A stride-1 access pattern ensures that each warp's 32 loads/stores are serviced in the minimum number of 32-byte transactions.
- **When**: Always. Every elementwise kernel must verify that its thread indexing produces coalesced access. The canonical pattern is `tid = blockIdx.x * blockDim.x + threadIdx.x; data[tid]`.
- **Measured impact**: On H200, stride-32 (non-coalesced) access is 2.85x slower than stride-1 (coalesced) access (see coalescing skill measured characteristics).

---

## Priority 2: bank-conflict (conditional)

- **Skill**: `30-skill/memory/bank-conflict/`
- **Why**: If an elementwise kernel stages data through `__shared__` memory (for example, to reorganize a non-contiguous tensor into a contiguous layout before applying the pointwise function), bank conflicts in the shared memory access can degrade performance.
- **When**: Only when the kernel uses `__shared__` memory. Most pure elementwise kernels do NOT use shared memory (there is no data reuse across threads), so this skill is rarely needed. Apply it only if profiling or code inspection reveals shared memory staging.
- **Measured impact**: On H200, 32-way bank conflicts cause 3.03x latency increase on shared memory loads (see bank-conflict skill measured characteristics).

---

## Priority 3: warp-primitives (rare)

- **Skill**: `30-skill/compute/warp-primitives/`
- **Why**: Pure elementwise kernels have no intra-warp communication needs -- each thread operates independently. However, if an elementwise kernel includes a reduction epilogue (e.g., computing a norm or a sum alongside the elementwise operation), warp shuffles can accelerate that reduction phase.
- **When**: Only when the kernel fuses an elementwise operation with a partial reduction or when a warp-level predicate check (`__any_sync`, `__all_sync`) is used for early-exit logic (e.g., checking for NaN values across a warp).
- **Measured impact**: Warp butterfly reduction takes ~145 cycles for 5 steps on H200 (see warp-primitives skill measured characteristics).

---

### Priority 4: Layout Transform (pre-kernel fix when the input is stride-unfriendly)

- **Skill**: `30-skill/memory/layout-transform/`
- **Why**: coalescing (Priority 1) is a kernel-side fix for the stride-1 case. When the input is a struct-of-fields (AoS) and the elementwise kernel touches only a subset of fields, or when a 2-D tensor's row width is not a multiple of 128 B, the kernel cannot *become* stride-1 from inside — the layout must change before the kernel runs. Sub-skill S1 (AoS → SoA) is the primary relevance; S2 (`cudaMallocPitch`) applies when 2-D rows are misaligned.
- **When**: the upstream data producer supplies AoS but the elementwise kernel reads a single field (or a subset of fields with low locality to each other). For an N-field struct where each elementwise kernel uses k fields, break-even is roughly `k/N × downstream_kernel_count` calls. Measured on H200 (24-B `Particle` struct, kernel reads `.x` only): AoS→SoA conversion amortizes at **3.65 downstream calls** (probe: `80-experience/hw-probes/aos-vs-soa/`).
- **When NOT**: all downstream kernels touch all struct fields (AoS and SoA are equivalent at the HBM level and SoA can be worse via cache fragmentation); single-use data that sees only one downstream kernel; rows already 128-B-aligned (skip S2).
- **Measured impact**: SoA read-one-field is **1.99×** faster than AoS read-one-field on H200; SoA + float4 vectorization is **2.99×** total. The `aos_to_soa_convert` kernel itself runs at 81 % HBM SoL.

### Priority 5: Vectorized Access

- **Skill**: `30-skill/memory/vectorized-access/`
- **Why**: float4 vectorized loads achieve 2.59× bandwidth over scalar loads on H200 (3784 vs 1463 GB/s). For memory-bound elementwise kernels this is the single biggest optimization after coalescing.
- **When**: input arrays are 16-byte aligned and N is divisible by 4. Use `reinterpret_cast<float4*>` or native vector types.

### Priority 6: Warp Divergence (relevant for fused activations / masked writes)

- **Skill**: `30-skill/compute/warp-divergence/`
- **Why it matters for elementwise**: fused kernels that combine a pointwise activation with a conditional branch (ReLU, GELU-select, masked write, clipped gradient) can hit divergence cost if the body is too large for the compiler to predicate. On a compute-bound fused-activation kernel on H200, divergent branching measured **1.82× slower** at any p ∈ (0, 1) vs uniform warps; short-body predicated rewrite is **flat** across p at ~1.07× — elementwise pipelines should always prefer ternary/`fmax`/predicated forms.
- **When to apply**:
  - Rewrite ReLU-style gates as `fmaxf(x, 0.f)` (branchless).
  - Use ternary for small conditional writes: `out[i] = cond ? v : out[i]`.
  - For longer conditional bodies (inline function with > ~7 instructions, or any function call), either split into two kernels by input mask, or accept the predicated 2× cost if p is unpredictable per-lane.
- **When NOT to apply**: elementwise kernels that are memory-bound (the common case: H200 HBM-bound copy-with-activation). On a memory-bound kernel, divergence cost is hidden behind the memory wait — do not churn the code for a metric that won't move. Measured: same kernel shape at N=16M was flat across p on all three variants.
- **Relevance to bottleneck triage**: Q4 in `70-reasoning/bottleneck-triage.md` — "Compute-bound elementwise with divergent conditional on hot path."

### Priority 7: Fast Math Intrinsics

- **Skill**: `30-skill/compute/fast-math/`
- **Why**: `__expf` is 1.27× faster than `expf` on H200 (3318 vs 2604 Gop/s). Relevant for activation functions (gelu, silu, sigmoid) and normalization (rsqrtf).
- **When**: compute involves transcendentals AND relaxed precision is acceptable (not for loss/gradient computation).

### Priority 8: Instruction-Level Parallelism (ILP)

- **Skill**: `30-skill/compute/ilp/`
- **Why**: processing multiple elements per thread via grid-stride loop with unrolling exposes ILP, hiding FMA pipeline latency (4 cycles on H200). 4-acc chains saturate the pipeline.
- **When**: kernel is latency-bound (low occupancy or few instructions per element). Memory-bound elementwise kernels gain less from ILP.

### Priority 9: Compiler Hints

- **Skill**: `30-skill/compute/compiler-hints/`
- **Why**: `__launch_bounds__`, `__restrict__`, `#pragma unroll` guide register allocation and code generation. `__restrict__` enables the compiler to auto-route loads through the read-only cache.
- **When**: tuning phase after functional correctness.

### Priority 10: Register Pressure Management

- **Skill**: `30-skill/memory/register-pressure/`
- **Why**: vectorized access + ILP increase register usage. Forced reduction (--maxrregcount=32) caused 4.84× slowdown from spilling on H200. Monitor with `-Xptxas=-v`.
- **When**: after applying skills 4-6, if occupancy drops below 50%.

### Priority 11: Async Copy (LDGSTS Prefetching)

- **Skill**: `30-skill/memory/async-copy/`
- **Why**: for compute-heavy fused elementwise chains (e.g., bias + gelu + scale), LDGSTS async prefetching to shared memory increases bytes-in-flight without consuming registers.
- **When**: ONLY when the elementwise computation is non-trivial (involves multiple transcendentals or long dependency chains). GTC25-S72683 demonstrated that trivial `a*b` elementwise gets NO benefit (actually 10% regression on H200). The "Stall Long Scoreboard" metric in ncu must be the dominant bottleneck.
- **Caveat**: most pure elementwise kernels are already bandwidth-saturated with coalescing + vectorized access. Async copy adds code complexity for marginal gain in the common case.

### Priority 12: Cache Load Hints (mostly a null on H200; avoid the traps)

- **Skill**: `30-skill/memory/cache-load-hints/`
- **Why it matters for elementwise**: elementwise kernels are the canonical "read-only input, stream through once" shape — the exact pattern the legacy KB suggested `__ldg` for. Measured on H200 sm_9.0a: at DRAM-bound elementwise scale (buffer > 60 MiB L2, single pass), **all six cache-hint variants are within 0.5%** of default. The hint choice is free and irrelevant; time spent picking one is wasted.
- **When to apply**:
  - Use `const __restrict__` on the input pointers and the compiler auto-emits `ld.global.nc` — same result as explicit `__ldg` with less code.
  - For chained / fused elementwise that re-reads the input across multiple passes in one kernel (rare but occurs in e.g. residual-add-then-gelu-then-scale), default-load. **Measured 2.26× slowdown** if `__ldcg` or `__ldcv` is used on L2-resident data with SM-local reuse (skill §S3).
- **When NOT to apply**:
  - Never use `__ldcg` / `__ldcv` on elementwise inputs. The 2.26× L2-reuse penalty hits any chained elementwise kernel where the input is touched by more than one pass.
  - `__ldcs` (streaming / evict-first) is a measured null in un-contended L2 — elementwise MVPs have no competing workload, so no benefit. Default-load.
- **Measured impact on H200**: none in single-kernel elementwise. Benefit only materializes in contended-L2 scenarios not reached by MVP harnesses; see `80-experience/hw-probes/cache-hint/2026-04-23-cache-hint.md`.
- **Relevance to bottleneck triage**: Q4 in `70-reasoning/bottleneck-triage.md` — agent reports "tried `__ldcg` and slower". Correct response: revert to default load and fix pointer qualifiers (`const __restrict__`).

### Priority 13: Half-Precision Math (fp16 / bf16 scalar + packed)

- **Skill**: `30-skill/compute/half-precision-math/`
- **⚠ PRECISION IMPACT — read before applying**: fp16 max finite ≈ **6.55e4** and ULP at 1.0 ≈ **4.88e-4** (~3 decimal digits); bf16 has fp32's range but ULP at 1.0 ≈ **7.81e-3** (~2 decimal digits). Choosing fp16/bf16 is a **correctness decision**, not just a performance decision — see skill's §Precision. Rule: format choice first (range vs precision), packing second.
- **Why it matters for elementwise**: elementwise is the canonical shape that benefits most from half precision — the kernel is memory-bound, so halving the byte footprint gives an immediate **2× effective bandwidth**. The compute-throughput gain (1.58×–1.84× vs fp32, measured) is secondary but real for fused transcendental chains (gelu, silu, activation+scale+add).
- **When to apply**:
  - **Pure elementwise (bandwidth-bound)**: load/store in fp16 or bf16 — halves DRAM traffic. Do the fused compute in whatever precision preserves correctness (skill §S1): if the op is a straight map with no sum (e.g. `out = a * scale + bias`), compute in fp16/bf16 directly. If the op accumulates (attention softmax numerator, reduction-in-elementwise-disguise), **promote to fp32** for the accumulator (skill §S3).
  - **Compute-heavy fused chains (gelu, silu, bias+act+scale+add)**: consider packed `__hfma2`. Measured on H200: only **1.16× over scalar `__hfma`** (not 2× as folklore claims; legacy P14 upgraded to measured). Worth the complexity only when the chain has ≥ 50 FMAs per pack and the alignment is natural.
  - **Precision-sensitive operations**: if any intermediate is compared against a small threshold or fed into a transcendental near a discontinuity, use fp32 regardless of memory cost. See elementwise INDEX.md new dtype decision step.
- **When NOT to apply**:
  - **Any long-running accumulation without fp32 promotion** (softmax denominator over H=4096, dot products, running averages). fp16 overflows at ~6e4; bf16 mantissa error ~0.8% per add. Skill P1, P2.
  - **Pure-streaming memory-bound kernel where packing adds alignment fragility**. The 1.16× compute gain is invisible behind memory stalls (P11). Use scalar `__hadd` / `__hfma`.
  - **Kernels built via `torch.utils.cpp_extension`** without explicit `__hadd` / `__hgt` calls (PyTorch build defines `-D__CUDA_NO_HALF_OPERATORS__`; P13).
- **Measured impact on H200 sm_9.0a**:
  - fp16 scalar = 1.58× fp32 scalar throughput (45 vs 29 TFLOPS).
  - fp16 packed (`__hfma2`) = 1.84× fp32 scalar (1.16× over fp16 scalar).
  - bf16 packed = 12% slower than fp16 packed (46 vs 53 TFLOPS).
  - See `80-experience/hw-probes/half2-throughput/2026-04-23-half2-throughput.md`.
- **Relevance to bottleneck triage**: Q3 in `70-reasoning/bottleneck-triage.md` — elementwise kernel with fp16 input and suspected overflow in a fused reduction-like step. Correct response: promote accumulator to fp32, cast at store.

### Priority 14: Branch Elimination (use intrinsics; do NOT simulate select with arithmetic)

- **Skill**: `30-skill/compute/branch-elimination/`
- **Why it matters for elementwise**: many elementwise kernels contain simple conditionals — ReLU (`max(x, 0)`), clamp, abs, guarded writes (`if (mask[i]) dst[i] = v`). On H200 sm_9.0a the `if/else` and ternary forms of these simple-body conditionals already lower to `FSEL` (SASS predicated-select) — they are branchless at the hardware level. The real win comes from using the **single-op hardware intrinsic** where one exists: `fmaxf` / `fabsf` / `fminf` emit 1 SASS op instead of the 2 ops a hand-written `if/else` produces.
- **When to apply**:
  - Rewrite ReLU as `fmaxf(x, 0.0f)` — measured **1.56× faster** than `(x < 0) ? 0 : x` (one `FMNMX` vs two-op `FSETP + FSEL`).
  - Rewrite `if (x<0) x=-x` as `fabsf(x)` — measured **2.36× faster** (one `LOP3.LUT` vs two-to-three-op branch form). Also faster than the explicit bit-trick `b & 0x7FFFFFFF` (1.51×).
  - Keep `if (cond) acc = v;` and `acc = cond ? v : acc;` as-is for register-level conditional assignment — both emit `FSEL` (1 op) and read more clearly than any rewrite.
- **When NOT to apply**:
  - Do NOT rewrite `if/else` as `a * (1-c) + b * c` — measured **1.27× slower** on H200 (4 FP ops vs 1 FSEL). This is the most common anti-pattern.
  - Do NOT use explicit bit tricks for abs / relu — the compiler synthesises `fabsf` to one `LOP3.LUT` but hand bit tricks emit two LOP3s because the int↔float round-trip prevents fusion.
  - When the `if/else` body is non-trivial (function call, memory store, long expression), a real branch (`BRA`) is emitted and divergence cost becomes real — that is the `warp-divergence` skill's regime, not this one.
- **Measured impact on H200 sm_9.0a**: 1.56× speedup (ReLU via `fmaxf`), 2.36× (abs via `fabsf`), 0.79× (if you arithmetic-simulate a select — slower). See `80-experience/hw-probes/branchless-patterns/2026-04-23-branchless-patterns.md`.
- **Relevance to bottleneck triage**: Q4 in `70-reasoning/bottleneck-triage.md` — elementwise kernel whose hot path includes `(x < 0) ? 0 : x` or `if (cond) dst[i] = ...`. Check SASS: if it is already `FSEL`, the branch doesn't exist; focus elsewhere. If it is `BRA`, simplify the body so predication kicks in.
