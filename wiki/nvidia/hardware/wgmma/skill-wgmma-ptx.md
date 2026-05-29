---
title: Hopper wgmma via raw PTX (cutlass-free)
status: verified
evidence_level: measured
applies_to_pattern_class:
- tensor-core
- tensor-core/gemm
applies_to_ops:
- gemm
- mma
requires_sm: '>=9.0a'
requires_features:
- wgmma
single_kernel_useful: true
cuda_version_tested: 12.9.86
driver_version_tested: 570.124.06
toolchain: nvcc 12.9 + ptxas 12.9
measured_on: H200-SXM | sm_90a | cuda 12.9.86 | driver 570.124.06
source:
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/parallel-thread-execution/cuda_parallel-thread-execution_index.html.md
  anchor: L28659-L28675
  excerpt: 9.7.15.5. Asynchronous Warpgroup Level Matrix Multiply-Accumulate Operation
    using wgmma.mma_async instruction — The input matrix A of the warpgroup wide MMA
    operations can be either in registers or in the shared memory. The input matrix
    B of the warpgroup wide MMA operations must be in the shared memory. When the
    matrices are in shared memory, their starting addresses must be aligned to 16
    bytes.
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/parallel-thread-execution/cuda_parallel-thread-execution_index.html.md
  anchor: L28451-L28465
  excerpt: wgmma.fence operations to indicate that the register/shared-memory across
    the warpgroup have been written into ... Issue the asynchronous matrix multiply
    and accumulate operations using the wgmma.mma_async operation ... Create a wgmma-group
    and commit all the prior outstanding wgmma.mma_async operations into the group,
    by using wgmma.commit_group ... Wait for the completion of the required wgmma-group.
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/parallel-thread-execution/cuda_parallel-thread-execution_index.html.md
  anchor: L28674-L28695
  excerpt: 9.7.15.5.1.1.1. Matrix Fragments for wgmma.mma_async.m64nNk16 — A warpgroup
    executing wgmma.mma_async.m64nNk16 will compute an MMA operation of shape .m64nNk16
    ... Elements of the matrix are distributed across the threads in a warpgroup so
    each thread of the warpgroup holds a fragment of the matrix.
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/parallel-thread-execution/cuda_parallel-thread-execution_index.html.md
  anchor: L29622-L29640
  excerpt: 9.7.15.5.1.2.2. Matrix Descriptor Format — Matrix descriptor specifies
    the properties of the matrix in shared memory that is a multiplicand in the matrix
    multiply and accumulate operation. It is a 64-bit value contained in a register.
- path: blogs/colfax/cutlass-tutorial-fast-matrix-multiplication-with-wgmma-on-nvidia-hopper-gpus
  anchor: Hopper wgmma walkthrough — descriptor format + sync semantics
- path: '{{CUTLASS_REPO_REF}}/include/cute/arch/mma_sm90_gmma.hpp'
  anchor: cutlass's wgmma inline-PTX wrapper (used as a reference for descriptor construction;
    not included in our binary)
- path: 80-experience/hw-probes/wgmma-ptx/artifacts/wgmma_hello.cu
  anchor: cutlass-free hello-world (single m64n8k16 bf16 atom)
- path: 80-experience/hw-probes/wgmma-ptx/artifacts/wgmma_zoo.cu
  anchor: cutlass-free zoo (11 configs across N-shape, dtype, layout, A-source)
artifacts:
  code: 80-experience/hw-probes/wgmma-ptx/artifacts/wgmma_hello.cu
  build: 80-experience/hw-probes/wgmma-ptx/artifacts/build.sh
  zoo_code: 80-experience/hw-probes/wgmma-ptx/artifacts/wgmma_zoo.cu
  zoo_codegen: 80-experience/hw-probes/wgmma-ptx/artifacts/gen_wgmma_zoo.py
  zoo_build: 80-experience/hw-probes/wgmma-ptx/artifacts/build_zoo.sh
  zoo_run: 80-experience/hw-probes/wgmma-ptx/artifacts/run_zoo.sh
  profile: 80-experience/hw-probes/wgmma-ptx/artifacts/profiles/2026-04-29-wgmma-zoo.csv
related_apis: []
related_skills:
- wgmma
- tma-ptx
- gemm-ptx
id: skill-wgmma-ptx
type: skill
vendor: nvidia
tags:
- cuda-cpp
applies_to:
- general
---
# Hopper wgmma via raw PTX (cutlass-free)

## What it is

A minimal implementation of Hopper's `wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16` instruction issued from inline PTX in a plain CUDA kernel that **does NOT include any cutlass or cute headers**. This is the cutlass-free wgmma primitive layer: it proves the agent can drive Hopper wgmma without depending on cutlass/cute, which is the prerequisite for a cutlass-free GEMM.

Compared to the cutlass-API track (which uses cutlass's `MMA_64x128x8_F32TF32TF32_SS_TN` atom via cute), this skill:

- Constructs the wgmma smem descriptor by hand (bit-layout per the PTX ISA spec).
- Issues `wgmma.fence`, `wgmma.mma_async`, `wgmma.commit_group`, `wgmma.wait_group` via inline PTX.
- Allocates fragments as plain C++ floats; no `cute::Tensor`, no `cute::Layout`, no `cutlass::Array`.
- Verifies the binary has zero `cutlass::` / `cute::` symbols at the preprocessor level (`nvcc -E | grep`) AND at the linked-symbol level (`cuobjdump --dump-elf-symbols | grep`).

## When to use it

- Building cutlass-free kernels (prerequisite for the cutlass-free GEMM).
- Auditing what cutlass's wgmma wrapper (`cute/arch/mma_sm90_gmma.hpp`) does at the PTX level — the wrapper itself is a thin inline-asm layer over the same instructions.
- Educational: understanding how the wgmma descriptor encoding works.

## When NOT to use it

- Production GEMM kernels where cutlass's tile / pipeline / scheduler abstractions would save thousands of lines of code. Cutlass exists for a reason; the cutlass-free track is "extract and rewrite", not "abandon cutlass for production".
- A wgmma issue rate above the **single-warpgroup serialized ~5.6 TFLOPS** ceiling. Real GEMM throughput requires multi-warpgroup-per-CTA + multi-accumulator pipelining + multi-CTA scaling — none of which this skill's primitives provide alone.

## PTX mnemonic inventory (this skill)

| Mnemonic | Purpose |
|---|---|
| `wgmma.fence.sync.aligned` | Barrier between non-wgmma SASS preceding the wgmma issue and the wgmma instructions; flushes outstanding dependent instructions. |
| `wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {acc}, descA, descB, scaleD, scaleA, scaleB, transA, transB` | The wgmma instruction itself. Reads matrices A/B from smem (via descriptors), accumulates into per-thread register fragment `{acc}`. |
| `wgmma.commit_group.sync.aligned` | Closes the current wgmma issue group. |
| `wgmma.wait_group.sync.aligned 0` | Waits for the group to retire so the accumulator fragments are stable for downstream stores. |

For the m64n128k8 tf32 atom (the cutlass-API reference), substitute `m64n128k8.f32.tf32.tf32` and adjust the per-thread fragment layout (64 floats per thread instead of 4).

## Smem descriptor format (uint64_t)

| Bits | Field | Notes |
|---|---|---|
| 0..13 | smem byte address >> 4 | The matrix start address in shared memory, divided by 16 (wgmma requires 16-byte alignment) |
| 14..29 | leading-dim byte offset >> 4 | Stride between rows (or cols, depending on layout) of the matrix, divided by 16 |
| 30..45 | stride-dim byte offset >> 4 | Stride between consecutive 8-row groups (the wgmma fragment unit), divided by 16 |
| 46..48 | matrix base offset | Used for 128B swizzle (set to 0 for no swizzle) |
| 49..51 | reserved | Must be 0 |
| 52..61 | fixed offset | Set to 0 |
| 62..63 | swizzle mode | 00=none, 01=128B, 10=64B, 11=32B |

Construction helper (see `wgmma_hello.cu`):

```cpp
__device__ __forceinline__ uint64_t make_smem_desc(
    const void* smem_ptr, uint32_t ld_bytes, uint32_t sd_bytes, uint32_t swizzle = 0) {
    uint32_t smem_int = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    uint64_t desc = 0;
    desc |= ((uint64_t)(smem_int >> 4)) & 0x3FFF;
    desc |= ((uint64_t)(ld_bytes >> 4) & 0xFFFF) << 14;
    desc |= ((uint64_t)(sd_bytes >> 4) & 0xFFFF) << 30;
    desc |= ((uint64_t)(swizzle & 0x3)) << 62;
    return desc;
}
```

## Atom-shape and dtype family covered

The cutlass-free PTX path covers the full Hopper wgmma family. The 11-config zoo at `80-experience/hw-probes/wgmma-ptx/2026-04-29-wgmma-zoo.md` exercises four orthogonal axes — *N-shape*, *element dtype*, *AB layout*, *A-source location* — all verified cutlass-free.

| Axis | Values measured | Notes |
|---|---|---|
| N-shape (bf16 SS_TN K=16) | 8, 16, 32, 64, 128, 256 | M=64 fixed for all bf16/fp16 atoms |
| dtype (N=64 SS_TN) | bf16 (K=16), fp16 (K=16), tf32 (K=8), s8 (K=32) | f32-acc for floats; s32-acc for int8 |
| layout | SS_TN (transA=0, transB=0), SS_NT (transA=1, transB=1) | NN / TT supported by spec but not measured |
| A-source | SS (smem descriptor), RS (4 × 32-bit regs / thread for K=16) | RS skips A's smem footprint |

PTX immediates count differs by dtype family — get this wrong and ptxas rejects with `Arguments mismatch for instruction 'wgmma.mma_async'`:

| Dtype | Immediates after descA, descB | Form |
|---|---|---|
| `.f32.bf16.bf16`, `.f32.f16.f16`, `.f16.f16.f16` | 5 | `scaleD, scaleA, scaleB, transA, transB` |
| `.f32.tf32.tf32` | 3 | `scaleD, scaleA, scaleB` (A and B are fixed K-major; no trans flags) |
| `.s32.s8.s8`, `.s32.u8.u8` | 1 | `scaleD` only (no scaleA/scaleB; no trans) |

## Measured Characteristics

### Correctness (hello-world)

H200-SXM, sm_90a. The single-instance hello-world kernel issues one m64n8k16 bf16 wgmma with all-1.0 inputs and verifies all 512 outputs equal K = 16 (matched / total = 512 / 512).

### Single-warpgroup throughput (11-config zoo)

H200-SXM, sm_90a. 1 CTA × 128 threads × N_INNER=1024 serialized wgmma issues per launch (accumulator-chained — every iteration depends on the previous, the worst-case pipeline serialization). All-ones inputs, every output = K × N_INNER. 5 warmup + 20 timed launches, median ms. **Every config passes the matched/total correctness gate.** Source `2026-04-29-wgmma-zoo.csv`.

| config            | N | K | TFLOPS (single-warpgroup serialized) |
|-------------------|--:|--:|-------:|
| ss_tn_bf16_n8     |   8 | 16 |   0.74 |
| ss_tn_bf16_n16    |  16 | 16 |   1.44 |
| ss_tn_bf16_n32    |  32 | 16 |   2.77 |
| ss_tn_bf16_n64    |  64 | 16 |   4.70 |
| ss_tn_bf16_n128   | 128 | 16 |   5.32 |
| ss_tn_bf16_n256   | 256 | 16 | **5.63** |
| ss_tn_fp16_n64    |  64 | 16 |   4.69 |
| ss_tn_tf32_n64    |  64 |  8 |   2.35 |
| ss_tn_s8_n64      |  64 | 32 | **9.55** |
| ss_nt_bf16_n64    |  64 | 16 |   4.70 |
| rs_tn_bf16_n64    |  64 | 16 |   4.69 |

**Important scale caveat**: these TFLOPS are *single CTA × 1 warpgroup × full accumulator-dependency chain*. Device peak (~990 TFLOPS bf16 on H200) requires multiplying by ~2 warpgroups × ~4 accumulator chains × 132 SMs ≈ 1000×. The numbers here characterize the per-instance issue cost, not engineering performance.

## Throughput shape — three independent multipliers

1. **N-shape**: doubling N below N=64 ≈ doubles TFLOPS (issue latency floor dominated by per-instance overhead). Above N=64 the instance itself takes more cycles per FMA, so TFLOPS plateaus near 5.6 — adding N just spends more cycles per issue.
2. **Dtype**: at N=64 SS_TN, normalized TFLOPS (`measured × 16 / K`) is ≈ 4.70 for all four dtypes. **Dtype is a precision/range knob, not a throughput knob.** s8 reports 9.55 only because K=32 (2× MACs/instance vs bf16 K=16); tf32 reports 2.35 only because K=8 (½× MACs).
3. **Layout & A-source**: SS_TN, SS_NT, RS_TN are within 0.3 % of each other at N=64 bf16. Trans flags reinterpret the descriptor / register pack; A in registers vs smem changes feed but not issue rate.

## Recommended atom selection (engineering rule of thumb)

| Constraint | Choose |
|---|---|
| Maximize FLOPs per warpgroup serialized issue | **N=128** for bf16/fp16 (5.32 TFLOPS — only 6 % below N=256, half the registers) |
| Minimize register pressure | N=64 (4.70 TFLOPS, 32 fp32 accum / thread vs 64 at N=128) |
| Match cutlass production atom (`MMA_64x128x8_F32TF32TF32_SS_TN`) | tf32 N=128 K=8 (not in the zoo; same instruction family) |
| Minimize accumulator dtype size | s32 + s8 inputs (K=32) — 2× the FLOP density of bf16/fp16 |

## Atomic Usage (PTX)

The raw PTX template for a single wgmma atom (bf16 SS_TN example):

```asm
// Fence: ensure all prior register/smem writes are visible to wgmma
wgmma.fence.sync.aligned;

// Issue: M=64, N=8, K=16, f32 accumulator, bf16 A and B from smem descriptors
wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16
    {%f0, %f1, %f2, %f3},     // 4 f32 accumulators per thread
    descA,                      // uint64_t descriptor for A matrix in smem
    descB,                      // uint64_t descriptor for B matrix in smem
    1,                          // scaleD: 1 = accumulate, 0 = overwrite
    1, 1,                       // scaleA, scaleB: 1 = +1.0
    0, 0;                       // transA=0 (K-major), transB=0 (K-major)

// Commit the issue group
wgmma.commit_group.sync.aligned;

// Wait for the group to retire
wgmma.wait_group.sync.aligned 0;
```

The number of accumulator registers per thread scales with N: N=8 gives 4 floats, N=64 gives 32, N=128 gives 64, N=256 gives 128. The immediate arguments also vary by dtype family (see the table in "Atom-shape and dtype family covered" above).

A complete self-contained atomic skeleton is at [40-hardware-feature/wgmma-ptx/atomic_skeleton.md](atomic_skeleton.md). The skeleton compiles and runs on H200 as a single m64n8k16 bf16 atom, producing the expected output (all 512 elements = K = 16). Build command: `nvcc -gencode=arch=compute_90a,code=sm_90a -o wgmma_hello wgmma_hello.cu` (see `80-experience/hw-probes/wgmma-ptx/artifacts/build.sh`).

## Cross-references

- wgmma reference (cutlass-API path): `40-hardware-feature/wgmma/skill.md` + `80-experience/api-probes/gemm/2026-04-28-wgmma-counters.md`.
- 11-config zoo with full sweep + open questions: `80-experience/hw-probes/wgmma-ptx/2026-04-29-wgmma-zoo.md`.
- TMA-PTX sibling (also cutlass-free): `40-hardware-feature/tma-ptx/skill.md`.
- Cutlass-free GEMM (composes both PTX primitives): `30-skill/compute/gemm-ptx/skill.md`.
- Failure modes: `40-hardware-feature/wgmma-ptx/pitfalls.md`.
