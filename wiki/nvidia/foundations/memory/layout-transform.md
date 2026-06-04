---
title: Memory Layout Transformation (AoS ↔ SoA, Pitched, Byte Permute)
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
- transpose
- reduction
requires_sm: '>=7.0'
requires_features:
- coalesced-memory-model
single_kernel_useful: true
source:
- path: spec
  anchor: Reference
artifacts:
  code: artifacts/experience/hw-probes/aos-vs-soa/aos_vs_soa_probe.cu
  build: artifacts/experience/hw-probes/aos-vs-soa/build.sh
  introspection: artifacts/experience/hw-probes/aos-vs-soa/device.json
  profile: ''
related_apis:
- cudaMallocPitch
- cudaMemcpy2D
- cudaMalloc3D
- cublasLtMatrixTransform
- prmt.b32
- cp.async.bulk.tensor
related_skills:
- coalescing
- shared-memory-cache
- vectorized-access
- bank-conflict
id: skill-layout-transform
type: skill
vendor: nvidia
tags:
- cuda-cpp
- tma
- fp8
- pipeline-stages
- vectorized-loads
- cache-policy
- shared-memory-optimization
- swizzling
- fused-kernel
- quantization
- ptx
applies_to:
- general
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L606-L634
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L1379-L1411
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L1413-L1483
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/parallel-thread-execution/cuda_parallel-thread-execution_index.html.md
  anchor: L13528-L13530
architectures:
- sm90
- sm90a
languages:
- ptx
- cuda-cpp
hardware_features:
- tma
- fp8
techniques:
- pipeline-stages
- vectorized-loads
- cache-policy
- shared-memory-optimization
- swizzling
kernel_types:
- fused-kernel
- quantization
confidence: experimental
artifact_dir: artifacts/experience/hw-probes/aos-vs-soa
---
## What

**Layout transformation** is the decision of *how data is organized in memory before a kernel touches it*, separate from *how the kernel accesses that memory*. The two are coupled by the hardware's 32-byte coalesced-transaction rule (PG §2.2.4.1, L1379-L1411): adjacent threads must land in the same 32-byte segment, or the warp issues multiple transactions and wastes bandwidth proportional to the stride.

Four classical choices sit under this umbrella, each trading **when** the cost is paid for **where** the win appears:

1. **AoS → SoA** — convert `struct{x,y,z}[N]` into `{x[N], y[N], z[N]}` so each hot field becomes a stride-1 array.
2. **Pitched 2D allocation** — let `cudaMallocPitch` pad row width so every row starts at a coalescing-aligned address.
3. **In-kernel transpose** — use shared memory to pivot a tile so both the read *and* the write sides are coalesced; mechanism owned by the sibling `shared-memory-cache` skill.
4. **Byte permutation (`prmt.b32`)** — sub-word reordering inside 32-bit registers for tight formats (RGBA↔BGRA, packed int8/fp8).

BP §10.2.1.4 (L606-L634) quantifies the stakes: stride 2 → 50 % bandwidth efficiency; stride 32 → 12.5 %. Any kernel whose natural access pattern has stride > 1 on the fastest-moving dim is paying this tax until a layout transform is applied.

## Why

Unlike coalescing (which is a kernel-side fix) and bank-conflict avoidance (which is a smem-side fix), layout-transform is a **pre-kernel** fix: it changes the data itself so that coalescing becomes automatic. That makes it expensive up front (a conversion kernel or a different allocation API) but free-forever for every subsequent kernel.

The strategic question is **amortization**: if one downstream kernel benefits, the transform is usually not worth it; if many kernels over the lifetime of the buffer benefit, it almost always is. BP §10.2.1.4's closing line — "non-unit-stride global memory accesses should be avoided whenever possible" — is a cost bound, not a hard rule. The skill below is the set of techniques for actually avoiding them, plus the guardrails for when the cost of the transform does *not* amortize (see pitfall P3).

## When to use

### S1. AoS → SoA conversion (primary sub-skill)

When the input is a struct-of-N-fields array and each kernel touches a **subset** of fields (the common case: particle systems where position-only kernels don't touch velocity, or feature-vector pipelines where one pass normalizes only a subset of columns), split the AoS buffer into one flat array per field.

```cuda
struct Particle { float x, y, z, vx, vy, vz; };     // 24 B per element

// Kernel reads only `.x`. In AoS it strides 24 B per thread (wastes
// 20/24 of every L2 transaction); in SoA it strides 4 B (coalesced).
__global__ void move_x_soa(const float* __restrict__ x,
                           float*       __restrict__ x_out,
                           int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x_out[i] = x[i] * 2.0f;
}
```

The one-time conversion kernel:

```cuda
__global__ void aos_to_soa(const Particle* __restrict__ in,
                           float* __restrict__ x, float* __restrict__ y,
                           float* __restrict__ z,
                           /* ...vx,vy,vz */,
                           int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        Particle p = in[i];                 // coalesced 24-B read per thread
        x[i] = p.x;  y[i] = p.y;  z[i] = p.z;   // each store is stride-1
        // ...
    }
}
```

BP §10.2.1.4 (L606-L634) quantifies the upside: the AoS-read-one-field pattern has stride equal to `sizeof(struct) / sizeof(field)`.

### S2. Pitched 2D allocation

When a 2D array's row width in bytes is not a multiple of the coalescing granularity (128 B on current architectures), adjacent rows start at misaligned offsets and every row's first warp pays a coalescing penalty. `cudaMallocPitch` rounds each row up to a well-aligned pitch:

```cuda
float* d_mat;
size_t pitch;                                // returned, in BYTES
cudaMallocPitch(&d_mat, &pitch, cols * sizeof(float), rows);

// Indexing:
float* row_k = (float*)((char*)d_mat + k * pitch);

// Host ↔ device copy MUST use cudaMemcpy2D and the pitch:
cudaMemcpy2D(d_mat, pitch,
             h_data, cols * sizeof(float),  // host pitch (tight)
             cols * sizeof(float), rows,
             cudaMemcpyHostToDevice);
```

Use this when `cols * sizeof(elem) % 128 != 0` and the kernel's access pattern is row-oriented. For row widths that are already 128 B multiples, plain `cudaMalloc` is cheaper — pitched allocation increases total footprint (see pitfall P4).

### S3. In-kernel transpose (pointer; mechanism in shared-memory-cache)

The `[TILE][TILE+1]`-padded shared-memory transpose is the standard out-of-place layout transform when no SoA split is possible (dense matrix transpose, batched axis swap). The **kernel** lives in `wiki/nvidia/foundations/memory/shared-memory-cache/skill.md` §S2; this skill records the *layout decision* — when transposing ahead-of-time beats running every downstream kernel with non-coalesced access. Measured on H200 (4096² fp32): padded tiled transpose hits 1685 GB/s; the per-step cost is ~0.08 ms. If > 3 downstream kernels benefit, the transpose amortizes.

### S4. Byte permutation with `prmt.b32`

When the data is sub-word — e.g. `uchar4` pixels, packed int8 for mixed-precision inference, packed fp8 scales — and the required layout change is a within-register byte reorder, inline-PTX `prmt.b32` avoids round-tripping through memory:

```cuda
// Reverse the 4 bytes inside a 32-bit word (RGBA → ABGR).
__device__ __forceinline__ unsigned int byte_reverse(unsigned int a) {
    unsigned int r;
    asm("prmt.b32 %0, %1, 0, 0x00010203;" : "=r"(r) : "r"(a));
    return r;
}
```

The selector `c` in `prmt.b32 d, a, b, c` is a 4-nibble control picking bytes 0..7 from the concatenation `(a || b)` (PTX ISA L13528). This is a register-only op, faster than any memory-based swizzle when the reorder fits inside 32 bits. For wider reorders, chain multiple `prmt.b32` calls or fall back to S3.

## When NOT to use

- **One-shot data.** If the buffer is read once and discarded, the conversion kernel's own memory traffic matches the traffic you would have paid with the bad layout. Amortization requires ≥ ~3 downstream kernels (derived from the probe: AoS→SoA kernel runs at ~HBM-peak, and so does each SoA kernel; the break-even is roughly (1 + 1) transform + SoA ops vs (stride-N) AoS ops).
- **Kernel hits all fields.** If every downstream kernel touches every field of the struct (the "hot data set = full struct" case), AoS and SoA are equivalent at the HBM level. SoA can then be worse because three separate buffers have three separate cache-line streams with weaker L1 locality.
- **Row width already aligned.** Don't add `cudaMallocPitch` for row widths that are already multiples of 128 B. The pitch padding adds footprint without adding coalescing (pitfall P4).
- **Sub-word permutation that fits a compile-time constant shuffle.** If the compiler already produces a single `prmt.b32` from a C++ expression involving `__byte_perm`, handwritten inline asm adds maintenance cost without perf gain — check SASS before committing.

## Measured Characteristics

Measured on H200-SXM (sm_90a, CUDA 12.9, driver 570.124.06) using sources/experience/hw-probes/aos-vs-soa/ — AoS-vs-SoA probe on `Particle{x,y,z,vx,vy,vz}` (24 B struct), N = 16,777,216 elements, kernel reads one field. Unlocked clock logged at 1980 MHz. Full record: sources/experience/hw-probes/aos-vs-soa.md.

| Kernel                        | Median ms | Useful BW GB/s | DRAM SoL | Warp cyc/issue | Speedup |
| ----------------------------- | --------: | -------------: | -------: | -------------: | ------: |
| `aos_read_one_field`          |    0.1111 |           1208 |   89.16% |         110.77 |   1.00× |
| `soa_read_one_field`          |    0.0558 |           2405 |   43.03% |      **51.34** |   1.99× |
| `soa_vectorized` (float4)     | **0.0372**|       **3613** |   71.31% |         110.66 |**2.99×**|
| `aos_to_soa_convert`          |    0.2017 |           3993 |   81.12% |         101.40 |     n/a |

**Amortization break-even: 3.65 downstream calls** (measured). The conversion kernel costs 0.2017 ms; each downstream SoA kernel saves 0.0553 ms vs keeping the data in AoS.

Key measured findings:

- **Stride penalty is ~2× on H200, not the naive 6×** (struct bytes over field bytes). L2 partially absorbs wasted sectors and the lower instruction count on SoA gives extra headroom. This is a measured correction to the folklore "stride-N = N× slower".
- **DRAM SOL alone is misleading**: the AoS kernel hits 89 % DRAM SOL because HBM is moving ~4.4 TB/s of *bytes*, but only 1.2 TB/s of *useful* bytes (24 % efficiency). Always compare to a stride-1 baseline when investigating a memory-bound kernel (pitfall P5).
- **Vectorization on SoA is free 1.5×** on top of the AoS→SoA transform (float4 loads pack 128 B of useful data per warp transaction).
- **Conversion kernel itself hits 81 % DRAM SOL and 71 % L1/TEX hit rate** — the AoS→SoA split is not a bandwidth waster, it's just a fixed one-pass cost that amortizes after ~4 downstream kernels.
- S2 (`cudaMallocPitch`), S3 (in-kernel transpose; mechanism in sibling `shared-memory-cache` probe), S4 (`prmt.b32`) are **not measured by this probe**. S3 is covered by the sibling skill's 2026-04-21 probe; S2 and S4 remain `inferred` until follow-up probes land.

## Principles

1. **Layout is a *pre-kernel* fix; coalescing is a *kernel* fix.** Both change the same metric (effective bandwidth), but at different points in the pipeline. Use layout transform when the access pattern cannot be changed from inside the kernel (e.g. you must read `.x` only).
2. **Amortize or don't transform.** A layout-transform pays off only when ≥ several downstream kernels use the new layout. One-shot buffers should keep their ingress layout.
3. **SoA splits should be by access-locality, not by struct fields.** If `.x` and `.y` are always accessed together, leave them as `struct{x,y}[N]`. Only split along access boundaries.
4. **`prmt.b32` is register-local.** Sub-word layout changes that can be expressed as byte-permute fit in one instruction; extending the idea to cross-register swaps requires memory ping-pong and becomes a variant of S3.

## Open questions

- Q1. What is the break-even number of downstream SoA kernels vs (1 × AoS→SoA convert + N × AoS kernels) on H200 for the 24 B struct shape of the probe? Planned follow-up probe: `aos-vs-soa/amortization-curve/`.
- Q2. Does `cp.async.bulk.tensor` (TMA with a tensor descriptor, sm_90+) perform an implicit layout transform during copy? If so, pairs of S1 + TMA may collapse into a single-instruction path. Blocked on the `wiki/nvidia/hardware/tma/` entry (pending bucket F bootstrap).
- Q3. On H200, when does pitched allocation's footprint overhead cost more in L2 miss rate than it saves in coalescing? Legacy pitfall P9 claims L2 hit rate dropped 62.7 %→50.1 % on an earlier device; needs re-measurement.

## Legacy references

- `legacy_sandbox_path`: `KernelPilot/optimization/memory/layout-transform/skill.md`. Original 5 sub-skills are routed as: Skill 1 (shared-mem transpose) → lives in sibling `shared-memory-cache` S2 (mechanism); layout decision pointer retained here as S3. Skill 2 (AoS→SoA) → kept as primary S1. Skill 3 (in-place row-to-column reorder) → retained conceptually inside S3 as the in-place variant of the transpose mechanism. Skill 4 (`cudaMallocPitch`) → kept as S2. Skill 5 (`prmt.b32`) → kept as S4.
- Legacy Level-3 sandbox findings P5-P9 are retained in `pitfalls.md`; they were run on pre-H200 hardware and require re-measurement.
