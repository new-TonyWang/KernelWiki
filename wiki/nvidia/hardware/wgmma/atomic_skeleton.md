---
id: skill-wgmma-ptx-atomic_skeleton
type: skill
vendor: nvidia
title: Atomic_Skeleton
tags:
- cuda-cpp
- wgmma
evidence_level: spec
applies_to:
- gemm
source:
- path: spec
  anchor: WGMMA PTX ISA
---
# wgmma Atomic Skeleton (PTX, cutlass-free)

This document extracts the minimal runnable wgmma kernel from `80-experience/hw-probes/wgmma-ptx/artifacts/wgmma_hello.cu`. The kernel issues a single `wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16` instruction with all-ones inputs and verifies that every output equals K=16.

## Build and run

```bash
nvcc -std=c++17 -O3 -gencode=arch=compute_90a,code=sm_90a \
     -lineinfo wgmma_hello.cu -o wgmma_hello
./wgmma_hello
# Expected: D entries matching K=16: 512 / 512
```

Cutlass-free verification:

```bash
nvcc -std=c++17 -O3 -gencode=arch=compute_90a,code=sm_90a -E wgmma_hello.cu \
    | grep -cE 'cutlass::|cute::'
# Expected: 0

cuobjdump --dump-ptx wgmma_hello | grep -c 'wgmma.mma_async'
# Expected: >= 1
```

## Kernel structure

```
1. Cooperative load: 128 threads load A (64x16 bf16 = 2 KB) and B (16x8 bf16 = 256 B) into smem
2. Build smem descriptors: make_smem_desc(ptr, ld_bytes, sd_bytes, swizzle)
3. wgmma.fence.sync.aligned        -- barrier between prior SASS and wgmma issue
4. wgmma.mma_async.sync.aligned    -- the MMA instruction
5. wgmma.commit_group.sync.aligned -- close the issue group
6. wgmma.wait_group.sync.aligned 0 -- wait for retirement
7. Scatter accumulator fragments back to global memory
```

## Smem descriptor construction

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

Bit layout: `[0:13]` = smem byte address >> 4; `[14:29]` = leading-dim offset >> 4; `[30:45]` = stride-dim offset >> 4; `[46:48]` = base offset (0 for no swizzle); `[62:63]` = swizzle mode.

## PTX inline assembly (the core 4-instruction sequence)

```cpp
float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;  // 4 f32 accumulators per thread
uint64_t descA = make_smem_desc(smem_a, 256, 256, 0);
uint64_t descB = make_smem_desc(smem_b, 256, 256, 0);

asm volatile("wgmma.fence.sync.aligned;\n");
asm volatile(
    "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
    "{%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n"
    : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
    : "l"(descA), "l"(descB));
asm volatile("wgmma.commit_group.sync.aligned;\n");
asm volatile("wgmma.wait_group.sync.aligned 0;\n");
```

Immediate arguments for `m64n8k16.f32.bf16.bf16`: `scaleD, scaleA, scaleB, transA, transB` (5 values). For tf32 atoms: 3 values (no trans flags). For s8 atoms: 1 value (scaleD only).

## Accumulator fragment layout

For `m64n8k16`: each of the 128 threads in the warpgroup owns 4 f32 values. The output matrix D is 64x8 = 512 elements. Fragment deposition follows the standard m64xN mapping:
- `row_base = warp_id * 16 + (lane_id >> 2)`
- `col_base = (lane_id & 0x3) * 2`
- Thread writes `d0, d1` at `(row_base, col_base)` and `(row_base, col_base+1)`, and `d2, d3` at `(row_base+8, col_base)` and `(row_base+8, col_base+1)`.

## Measured result

H200-SXM, sm_90a, cuda 12.9.86. All-ones input:
- `D entries matching K=16: 512 / 512` (correctness gate pass)
- Single-warpgroup serialized throughput: 0.74 TFLOPS (see zoo config `ss_tn_bf16_n8`)

## Adapting to other atoms

To switch to `m64n64k16.f32.bf16.bf16`, change the PTX mnemonic and increase the accumulator fragment to 32 floats per thread (64*64/128). To switch to tf32, use `m64n64k8.f32.tf32.tf32` and drop the `transA, transB` immediates.

## Source

Full runnable code: `80-experience/hw-probes/wgmma-ptx/artifacts/wgmma_hello.cu`
Build script: `80-experience/hw-probes/wgmma-ptx/artifacts/build.sh`
Measured record: `80-experience/hw-probes/wgmma-ptx/2026-04-28-wgmma-ptx-hello.md`
