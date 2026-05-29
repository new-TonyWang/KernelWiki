---
id: pitfall-tma
type: pitfall
vendor: nvidia
title: Pitfalls
---
# TMA — pitfalls

## 1. `-arch=sm_90a`, not `-arch=sm_90`

TMA + wgmma encodings live in `sm_90a` (the "a" = architecture-specific extension). Compiling with plain `sm_90` produces `ptxas` errors on `cp.async.bulk.tensor.*`. The canonical build script at `sources/experience/api-probes/gemm/artifacts/build.sh` pins `-arch=sm_90a` for this reason.

## 2. Descriptor lifetime: encoded once, read many times

`CUtensorMap` descriptors are typically encoded host-side via `cuTensorMapEncodeTiled` and copied to a device-side `__constant__` symbol. They must outlive every kernel that references them. cutlass example 48 hides this behind `cute::make_tma_copy(...)`; if you ever drop down to raw PTX, do **not** rebuild the descriptor per-kernel-launch — that doubles host-side cost for no gain.

## 3. Element size + alignment constraints

The TMA engine accepts a fixed set of `(element_size, dimension_count, swizzle_mode)` triples. The most common Hopper combos:

- 1-D / 2-D / 3-D / 4-D / 5-D, element size ∈ {1, 2, 4, 8, 16} bytes.
- Swizzle modes: `none / 32B / 64B / 128B`. The cutlass cooperative GEMM uses `Swizzle<3,4,3>` over a 32-bit unit (= 128B effective stride at 4-byte elements).
- The base address must be aligned to 16 B (some descriptor variants require 64 B).

A descriptor that the API encodes successfully but that the kernel cannot dispatch to almost always indicates a swizzle/alignment mismatch — the runtime won't tell you, the kernel just produces wrong data.

## 4. mbarrier mis-arrival count = silent data-race

`cp.async.bulk.tensor.*` increments an `mbarrier` by exactly one. If the mbarrier was initialized with the wrong arrival count (`mbarrier.init.shared` second argument), wait operations either deadlock or proceed with stale smem. cutlass `cute::TmaInst::copy(...)` plus the `cute::PipelineTmaAsync` machinery hide this; if you write your own mainloop, mirror `MainloopSm90TmaGmmaWarpSpecialized` exactly.

## 5. Don't mix TMA with `cp.async` for the same tile

Both load primitives can target the same smem region, but the synchronization semantics differ — `cp.async`'s commit/wait barrier is independent of mbarrier. Mixing them on the same buffer produces races that look like correctness drift on the second iteration of a multi-stage pipeline.

## 6. Blackwell-only template warnings during compile are harmless

Building the cutlass example on H200 emits warnings of the form

```
sm100_static_tile_scheduler.hpp(53): warning #20012-D: __host__ annotation is ignored on a function …
```

These come from cutlass headers that pull in Blackwell (sm_100) specializations transparently. They are non-fatal on sm_90a and can be silenced with `-diag-suppress 20012`. **Do not** start "cleaning them up" by adding `#ifdef __CUDA_ARCH__` guards inside cutlass headers — that creates a mainenance debt and breaks the upstream pin.

## 7. The single-launch ncu number is not the steady-state number

The first invocation of the example reports ~0.608 ms; under ncu's instrumented launch the same kernel reports ~0.799 ms (a ~30 % overhead from the profiler attaching to the process and inserting its replay buffer). Always compare against an unprofiled baseline; a "regression" detected only under ncu is usually the profiler. The probe record's TFLOPS number is the unprofiled run.

## 8. `smsp__inst_executed_pipe_wgmma.sum` is not available on this driver

Even though the kernel issues wgmma, the `wgmma`-specific pipe counter returns `n/a` under ncu 2025.2.1 + driver 570.124.06. Use `sm__inst_executed_pipe_tensor_op_hmma.sum` as the proxy on this stack — it correctly counts wgmma issues on Hopper.
