---
title: Half-Precision Math (fp16 / bf16 scalar and packed)
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
- normalization
applies_to_dtypes:
- fp16
- bf16
- fp32
requires_sm: '>=7.0'
requires_features:
- fp16-simd
- bf16-simd
single_kernel_useful: true
precision:
  fp16:
    exponent_bits: 5
    mantissa_bits: 10
    max_finite: 65504
    min_normal: 6.1e-05
    eps_relative: 0.000488
    range_decimal: 6.1e-5 to 6.55e4
    accumulation_safe: false
    notes: 'Narrow dynamic range. Safe for activations in inference; dangerous for reductions unless promoted to fp32. Denormals: flushed unless the instruction is a `.noftz` variant (e.g., atom.add.noftz.f16 preserves them).'
  bf16:
    exponent_bits: 8
    mantissa_bits: 7
    max_finite: 3.39e+38
    min_normal: 1.18e-38
    eps_relative: 0.00781
    range_decimal: 1.2e-38 to 3.4e+38 (same as fp32)
    accumulation_safe: false
    notes: 'Training-friendly: covers fp32 dynamic range, so no overflow in typical weight / gradient paths. Precision is coarser than fp16 (~7 bits mantissa vs 10). Still not safe for long reduction chains — promote accumulator to fp32.'
  fp32_accumulator_rule: When input is fp16/bf16 but the operation is a reduction, dot product, softmax normaliser, variance, or any sum of many terms, accumulate in fp32 and only cast back at store time. This is the mixed-precision pattern; skill §S3 documents the canonical shape.
source:
- path: spec
  anchor: Reference
artifacts:
  code: artifacts/experience/hw-probes/half2-throughput/half2_throughput_probe.cu
  build: artifacts/experience/hw-probes/half2-throughput/build.sh
  introspection: artifacts/experience/hw-probes/half2-throughput/device.json
  profile: ''
related_apis:
- __hadd
- __hadd2
- __hmul
- __hmul2
- __hfma
- __hfma2
- __hfma_relu
- __hfma2_relu
- __float2half_rn
- __half2float
- __float2bfloat16_rn
- __bfloat162float
- __halves2half2
- __floats2half2_rn
- __floats2bfloat162_rn
- hexp
- hlog
- hrsqrt
- h2exp
- h2log
- h2rsqrt
- add.f16x2
- fma.rn.f16x2
- fma.rn.bf16x2
- fma.rn.relu.f16
- tanh.approx.f16
- atom.add.noftz.f16
related_skills:
- fast-math
- vectorized-access
- ilp
- register-pressure
- atomic-reduction
experience_refs:
- sources/experience/hw-probes/half2-throughput.md
id: skill-half-precision-math
type: skill
vendor: nvidia
tags:
- cuda-cpp
- pipeline-stages
- vectorized-loads
- cache-policy
- data-reuse
- kernel-fusion
- software-exp
- fused-kernel
- gemm
- attention
- ptx
applies_to:
- general
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1370-L1440
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L25350-L25420
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/parallel-thread-execution/cuda_parallel-thread-execution_index.html.md
  anchor: L11000-L11150
architectures:
- sm90
- sm90a
languages:
- ptx
- cuda-cpp
techniques:
- pipeline-stages
- vectorized-loads
- cache-policy
- data-reuse
- kernel-fusion
- software-exp
kernel_types:
- fused-kernel
- gemm
- attention
confidence: experimental
artifact_dir: artifacts/experience/hw-probes/half2-throughput
---
## What

Half-precision math on H200 sm_9.0a means **fp16** (`__half`, `__half2`) or **bf16** (`__nv_bfloat16`, `__nv_bfloat162`) arithmetic, issued as scalar instructions or as 2-way packed SIMD via the `*.f16x2` / `*.bf16x2` PTX forms (`__hfma2`, `__hadd2`, `__hmul2`). Both formats consume 16 bits of storage per value — half the footprint of fp32 — which **doubles effective memory bandwidth** for pure-load kernels (the dominant benefit for elementwise / normalization kernels that are bandwidth-bound).

The arithmetic-throughput advantage is smaller than folklore suggests. Measured on H200 sm_9.0a:

- Scalar fp16 / bf16 FMA run at **~1.58×** scalar fp32 FMA — nvcc auto-packs the scalar chains into `HFMA2.MMA` / `HFMA2.MMA.BF16_V2` SASS when ILP permits, so scalar fp16/bf16 already gets most of the packing benefit.
- Packed `__hfma2` (fp16) runs at **1.84×** scalar fp32 — only **1.16× more** than scalar fp16. Explicit packing buys the residual register-layout / alignment headroom, not a new hardware path.
- Packed bf16 runs at **1.62×** scalar fp32 — **12% slower than fp16 packed** because `HFMA2.MMA.BF16_V2` has a lower-throughput variant of the half-precision HFMA2 pipe on sm_9.0a.
- bf16 scalar vs bf16 packed is only **2.4%** — within noise; the compiler already auto-packs bf16, and the BF16_V2 pipe's lower ceiling absorbs any alignment residual.

Consequently, the skill's decision rules are not "always pack" — they are "choose the format for **precision and range**, then use packed where packing amortises over enough FMAs to matter".

## Why

Three reasons to move a computation from fp32 to fp16 / bf16 on H200:

1. **Memory bandwidth is halved**. Every load and store moves 2 bytes instead of 4. For memory-bound kernels (the common case in elementwise and in the scaling phase of normalization), this is a **2× effective bandwidth** gain — a much bigger lever than the compute throughput gain.
2. **Compute throughput is 1.58–1.84× vs fp32**. Measurable but smaller than the memory win, and the packing pays only 1.16× on top. Useful for compute-heavy fused chains (activations with transcendentals, fused norm+matmul epilogues).
3. **Tensor Cores consume fp16 / bf16 natively**. A skill outside this MVP's scope, but the format choice here is downstream of a later TC-based workflow.

The cost is **precision loss** — see §Precision below. The decision rule is: **use the format that preserves correctness, then pack where the compute intensity justifies it**.

## Precision (annotation required in every caller)

*This section exists because half precision introduces **correctness risk**, not just performance tuning. Callers (pattern INDEX.md, skill.md chains) must read this before selecting a half format.*

### Range and resolution

| Format | Exp bits | Mantissa bits | Max finite | Min normal | Unit of last place (at 1.0) |
|--------|---------:|--------------:|-----------:|-----------:|----------------------------:|
| fp32   |        8 |            23 |   3.4e+38 |  1.18e-38 |                    1.19e-7 |
| fp16   |    **5** |        **10** | **6.55e+4** | **6.10e-5** |               **4.88e-4** |
| bf16   |    **8** |         **7** | **3.4e+38** | **1.18e-38** |              **7.81e-3** |

- **fp16 has the same mantissa resolution as a short float** — about 3 decimal digits of precision. Any time two values differ by <0.05%, fp16 can't distinguish them.
- **fp16 has a narrow exponent range** — the infamous "65k ceiling". Overflow is trivially easy in any sum of ~1000 elements of magnitude ~100.
- **bf16 has fp32's range** — overflow is a non-issue — but the mantissa is only 7 bits, about **16× coarser** than fp16. Any computation where small increments matter (decay terms, epsilons, running averages near steady state) will lose them.

### Accumulation rule (load-bearing for correctness)

**Never accumulate a sum of more than a few terms in fp16 or bf16.** Promote to fp32 at the point of summation, cast back only at the store. This is enforced by the canonical "mixed-precision reduction" pattern (skill §S3):

```cuda
__half* in_h;
float sum = 0.0f;
for (int i = tid; i < N; i += stride) {
    float v = __half2float(in_h[i]);
    sum = fmaf(v, v, sum);    // fp32 accumulation
}
// ... warp/block reduce in fp32 ...
out_h[tid] = __float2half_rn(sum);   // cast at write
```

Why: in fp16, summing a thousand terms of ~1.0 accumulates rounding error of order `1000 × 4.88e-4 ≈ 0.5` per unit of sum — any sum near or above 2000 already carries a full ULP of error. In bf16 the situation is **worse** (mantissa 7 bits), just without the overflow.

### Denormals

fp16 and bf16 both represent subnormal values. They are **flushed to zero** unless:

- The instruction is a `.noftz` variant (e.g. `atom.add.noftz.f16` — "no flush-to-zero").
- The compiler option / kernel attribute opts out of FTZ.

In practice, flush-to-zero is the default, and subnormals round to 0. For most workloads this is fine; for workloads using very small epsilons (normalisation denominators, attention masks with -inf floors) verify that the epsilon is above the min-normal threshold (fp16: 6.1e-5). See pitfall P4.

### When fp16 / bf16 is WRONG

- **Any sum of > ~100 fp16 terms** without fp32 promotion. Overflow hits at sum ≈ 6.55e4; precision drifts long before that (pitfall P1).
- **Any gradient / backward pass without bf16 or mixed-precision discipline**. Training typically needs bf16's range.
- **Any computation where epsilon < min-normal** (fp16: 6e-5, bf16: 1e-38). Check the epsilon before selecting the format.
- **Any atomic reduction that expects fp32 convergence.** Prefer hierarchical fp32 fan-in over native fp16/bf16 atomics (see §S6 and the `atomic-reduction` skill).
- **Any numerically sensitive operation** (matrix inverse, log-sum-exp with large dynamic range, solving linear systems) — fp16/bf16 mantissas are too coarse.

## When to use

### S1. Use fp16 / bf16 for storage and memory-bound elementwise

For a straight elementwise kernel (`out[i] = activation(in[i])`), the bottleneck is memory; fp16/bf16 halves the bytes moved. This is **2× effective bandwidth** immediately. The compute choice inside the kernel follows S3 (promote to fp32, compute, cast back) — the memory saving is preserved either way.

### S2. Use packed `__hfma2` only for compute-bound FMA chains (fp16 only)

Measured on H200: packed `__hfma2` over `__half2` is **1.16×** scalar `__hfma` on fp16; on bf16 the same explicit-packing gain collapses to **1.02×** (within noise). Packing is justified when:

- The data type is **fp16** (for bf16, explicit packing adds nothing measurable — the compiler already auto-packs scalar into `HFMA2.MMA.BF16_V2`).
- The kernel is **compute-bound** — in memory-bound kernels the 16% is invisible behind memory stalls (pitfall P11).
- The FMA chain is long enough to amortise `half2` pack/unpack overhead. Rule of thumb: ≥ 50 FMAs per pack.
- The data layout allows natural 32-bit alignment (e.g. row-stride even in fp16 elements). Misalignment costs more than the packing win (pitfall P3).

```cuda
// Packed FMA chain (compute-bound epilogue)
half2* in_h2; half2* out_h2;
half2 a = in_h2[i];
half2 b = __floats2half2_rn(scale, scale);
half2 bias2 = __floats2half2_rn(bias, bias);
// One instruction does 2 fp16 FMAs:
half2 y = __hfma2(a, b, bias2);
out_h2[i] = y;
```

If the kernel is memory-bound, skip packing and use scalar `__hfma` — the code is simpler, the alignment risk is gone, and the 1.16× compute gain is invisible behind the memory wall.

### S3. Mixed-precision accumulation pattern (mandatory for reductions)

Any reduction, dot product, norm, softmax numerator, variance accumulator, or running average **must** use fp32 accumulation when the input is fp16 or bf16. See the probe record for the measured rationale; see §Precision above for the numerical reason.

```cuda
__global__ void fp16_dot_with_fp32_acc(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    float* __restrict__ out, int N)
{
    float acc = 0.0f;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < N; i += stride) {
        float fa = __half2float(a[i]);
        float fb = __half2float(b[i]);
        acc = fmaf(fa, fb, acc);
    }
    // Warp / block reduce `acc` in fp32 (see reduction skill).
    // Final atomicAdd(out, acc) in fp32 (see atomic-reduction skill).
}
```

### S4. Half-precision transcendentals (`h2exp`, `hrsqrt`, etc.)

`hexp`, `hlog`, `hrsqrt`, `hsqrt` and their packed `h2*` forms are available in `<cuda_fp16.h>`. On H200 the packed forms **should** map to dedicated instructions per PG §5.4.11.2, but this has not been re-measured by this probe — decomposition into fp32 ops on older arches is a documented risk (pitfall P10). Use them for softmax / normalisation denominators where the input is already fp16 and the precision loss is acceptable (normaliser eps must be ≥ 6e-5 for fp16; see §Precision).

### S5. Fused FMA+ReLU (`__hfma2_relu`)

Maps to PTX `fma.rn.relu.f16x2`. Computes `max(a*b + c, 0)` in a single instruction. Useful for the activation epilogue of fused elementwise chains. Instruction-count reduction is small (~4%) and wall-clock benefit is only visible in compute-bound kernels (pitfall P11). Do not adopt this hint in memory-bound elementwise.

### S6. Native fp16 / bf16 atomicAdd

Available on sm_70+ (fp16) and sm_80+ (bf16). PTX maps to `atom.add.noftz.f16` / `atom.add.noftz.bf16`. `.noftz` preserves denormals through the atomic. Contention throughput is worse than fp32 atomicAdd (pitfall P12); prefer the hierarchical reduction pattern from the `atomic-reduction` skill and atomic only the final per-block partials — that way the atomic is in fp32 and cheap.

## When NOT to use

- **Long reductions / running sums / softmax denominators without fp32 promotion.** fp16 overflow at ~6e4, bf16 mantissa error at ~0.8% per addition — numerically unsound (pitfalls P1, P2).
- **Packed `__hfma2` in memory-bound kernels.** Measured 1.16× over scalar fp16 is invisible behind memory stalls (pitfall P11).
- **Explicit `__hfma2` on `__nv_bfloat162` for compute speedup.** Measured 1.02× over scalar bf16 on H200 — noise-level. The compiler already auto-packs scalar bf16 into `HFMA2.MMA.BF16_V2`; explicit packing is redundant for bf16.
- **bf16 in precision-sensitive downstream consumers.** bf16's ~0.8% eps (vs fp16's ~0.05%) is coarser; any operation whose result is within 1% of a decision threshold will have false crossings.
- **Atomics on a single fp16/bf16 destination under contention.** Use hierarchical fp32 fan-in instead (`atomic-reduction` S1).
- **Kernels built via `framework extension build` that use `__half` via operators.** Framework extension build defines `-D__CUDA_NO_HALF_OPERATORS__` etc. — use explicit `__hadd`, `__hgt`, `__float2half_rn` calls (pitfall P13).

## Measured Characteristics

Measured on H200-SXM (sm_9.0a, CUDA 12.9, driver 570.124.06) using sources/experience/hw-probes/half2-throughput/ — five FMA variants in a 4-chain ILP compute-bound harness. Full record: sources/experience/hw-probes/half2-throughput.md.

### Scalar-equivalent GFLOPS (counts 2 FP ops per packed instruction)

| variant       | scalar GFLOPS | vs fp32 | vs own-scalar |
|---------------|--------------:|--------:|--------------:|
| fp32-scalar   |         28633 |   1.00× |          —    |
| fp16-scalar   |         45351 | **1.58×** |          —    |
| fp16-packed   |         52608 | **1.84×** |    **1.16×** |
| bf16-scalar   |         45381 |   1.59× |          —    |
| bf16-packed   |         46478 |   1.62× |    **1.02×** |

Key measured findings on H200 sm_9.0a:

- **`__hfma2` is 1.16× scalar `__hfma`**, not 2×. The dominant win is scalar fp16 vs scalar fp32 (1.58×, from compiler auto-packing into `HFMA2.MMA`); explicit packing adds only 16% more.
- **bf16 packed is 12% slower than fp16 packed**. Scalar bf16 equals scalar fp16 at 45 TFLOPS; the divergence appears only on the packed path because `HFMA2.MMA.BF16_V2` has lower pipe throughput than `HFMA2.MMA`.
- **bf16 scalar ≈ bf16 packed** (1.02×, within noise). Explicit `__hfma2` on `__nv_bfloat162` adds essentially nothing — auto-packing already lifts scalar bf16 to the BF16_V2 pipe's saturation ceiling.
- **fp32 scalar runs at 28.6 TFLOPS = 43% of H200's 67 TFLOPS peak**, with NCU Compute SOL = 78%. Probe is compute-bound on issue slot but not register-file-saturated; absolute peak would need wider ILP (a separate skill domain).

## Principles

1. **Format choice is about precision + range, not throughput.** The throughput differences (1.16× packing, 1.58× scalar fp16 vs fp32) are small compared to the correctness risk from coarse mantissa or narrow range.
2. **Always accumulate in fp32.** Reductions, softmax, norms, variances — fp32 the accumulator, cast at the store.
3. **Memory bandwidth halving is the real win.** For bandwidth-bound kernels (the common case), fp16/bf16 gives 2× instantly; don't obsess over packing.
4. **Prefer fp16 to bf16 unless range forces bf16.** Measured 12% throughput edge + 16× finer eps.
5. **`.noftz` atomics preserve denormals;** scalar non-atomic fp16 ops flush. Verify your epsilons.

## Open questions

- Q1. **RESOLVED** (SASS audit 2026-04-23): `__hfma2` is only 1.16× scalar `__hfma` because **nvcc auto-packs scalar `__hfma` chains into `HFMA2.MMA` SASS opcodes**. Both the scalar and packed templates produce the same `HFMA2.MMA` instruction mix; "scalar vs packed" at C-source is largely erased at SASS when ILP provides independent pairs. The 1.16× residual is register-layout / alignment cost, not the packing gain itself. See probe record §"PTX / SASS audit".
- Q2. **PARTIALLY RESOLVED** (SASS audit 2026-04-23): bf16 packed is 12% slower than fp16 packed because it emits `HFMA2.MMA.BF16_V2` opcodes — a distinct HFMA2 variant, **not** a fall-through to `FFMA`. Both remain on the half-precision FMA pipe family; the BF16_V2 variant simply has lower measured throughput than the base `HFMA2.MMA` on sm_9.0a hardware. Root cause (pipeline width, multiplier latency, or register port allocation) would need an isolated-instruction microbench to resolve — out of scope here. The compiler hypothesis "bf16 uses FP32 slot" is refuted.
- Q3. On sm_9.0a do `h2exp`, `h2log`, `h2rsqrt` emit single instructions (as PG §5.4.11.2 implies) or decompose into fp32 ops (pitfall P10)? Follow-up probe `sources/experience/hw-probes/half-transcendental/` (open).
- Q4. Is `__hfma2_relu` measurably faster than `__hfma2` + explicit `max(x, 0)` in a compute-bound kernel? ~4% instruction reduction is documented; wall-clock gain in compute-bound kernels is not (pitfall P11). Follow-up probe `sources/experience/hw-probes/half-fma-relu/` (open).
- Q5. Does native `atomicAdd(__half*, __half)` on H200 contention match the "worse than fp32" behavior described in pitfall P12? Not re-measured. Follow-up.

## Legacy references

- `legacy_sandbox_path`: `corpus/nvidia/legacy-optimization/compute/half-precision-math/skill.md`. The legacy skill kept six sub-skills (S1 packed h2, S2 bf16, S3 mixed-precision accumulation, S4 transcendentals, S5 hfma_relu, S6 native atomics). This port:
  - Re-orders by the measured H200 hierarchy: memory-bw-first (§S1), packed-is-marginal (§S2, fp16 only), accumulation-in-fp32 (§S3), transcendentals (§S4), fused FMA+ReLU (§S5), native atomics (§S6).
  - Adds the **Precision** section before any sub-skill — making the correctness-risk annotation the first thing a pattern-level INDEX or ROUTING caller reads.
  - The legacy "2× from packing" claim is corrected by the half2-throughput probe to 1.16× on fp16 / 1.02× on bf16.
- Legacy pitfalls retained in body (rewritten to stand on PG / PTX ISA sources): P1 (fp16 overflow), P2 (bf16 mantissa loss), P3 (packed alignment), P4 (denormal FTZ), P5 (conversion overhead), P6 (bf16 sm_80+ only), P13 (the external framework extension build flags). P10 / P11 / P12 kept as open risks (not re-measured on H200; follow-up probes listed in §Open questions).
- Legacy sandbox-only pitfalls dropped from body (no independent doc source, not re-measured by the half2-throughput probe): "L1 cache hit rate drop with half2" and "BF16 packed changes cache-line reuse" — these were empirical observations without authoritative grounding; they are available in the legacy-knowledge tree if needed but are not carried forward as measured facts.
- Measured pitfalls added by the half2-throughput probe (2026-04-23): P14 (`__hfma2` is 1.16× not 2×), P15 (bf16 packed is 12% slower than fp16 packed), P16 (`<cuda_bf16.h>` overload-resolution gotcha).
- **Related but distinct**:
  - `wiki/nvidia/foundations/compute/fast-math/` covers fp32-specific fast-math approximations (`-use_fast_math` only affects fp32). The two don't compose automatically.
  - `wiki/nvidia/foundations/memory/vectorized-access/`: packed `half2` aligns naturally to 4-byte loads. Use vectorized-access for load/store widening, this skill for the arithmetic inside.
  - `wiki/nvidia/foundations/memory/register-pressure/`: packed formats double accumulator width; check regs-per-thread after adopting §S2.
