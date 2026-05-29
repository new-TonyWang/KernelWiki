---
id: pitfall-wgmma
type: pitfall
vendor: nvidia
title: Pitfalls
---
# wgmma — pitfalls

## 1. `-arch=sm_90a`, not `-arch=sm_90`

wgmma encodings live in the architecture-specific extension `sm_90a`. Plain `sm_90` will not emit wgmma PTX even if the cutlass templates instantiate it; you'll get either a `ptxas` error or, worse, a silently downgraded mma.sync path on some toolchain combinations. The canonical build script at `80-experience/api-probes/gemm/artifacts/build.sh` pins `-arch=sm_90a`.

## 2. **Do not** copy Ampere `mma.sync` / `wmma` experience verbatim

On Ampere a single warp issues `mma.sync.aligned.m16n8k16` (16 threads cooperate). On Hopper a wgmma issue is at the **warpgroup** granularity (128 threads = 4 warps cooperate on a 64-row tile). Carrying Ampere assumptions about register layouts, sync semantics, or accumulator placement to wgmma produces correct-but-slow kernels (or worse, race-condition surprises). Specifically:

- Ampere: `__syncthreads()` between consecutive mmas; Hopper: `wgmma.fence` + `wgmma.commit_group` + `wgmma.wait_group<N>` decouples issue from completion.
- Ampere: accumulator can sit in registers OR shared memory; Hopper: accumulator must be in **registers** of the consumer warpgroup throughout.
- Ampere: warps are independent; Hopper: a warpgroup is the unit of issue and **all 4 warps must stay live** until `wgmma.wait_group` clears.

## 3. SmemLayoutAtom must be wgmma-compatible

A wrong smem layout gives one of two failure modes:

- **Build-time failure**: cutlass's `static_assert` chain rejects the layout (most common; the error message is verbose but pointable).
- **Runtime silent corruption**: the kernel runs, but A or B operands are read from the wrong banks. The cutlass example uses `Swizzle<3,4,3>` over a 32-bit unit; deviating without understanding the SS-form access pattern is a foot-gun. Use `cute::GMMA::ss_op_selector` or the high-level `GMMA::CtaTileMNK` helpers.

## 4. Block size must cover ≥1 warpgroup

128 threads = 1 warpgroup. The cooperative kernel uses 3 warpgroups (1 producer + 2 consumers). A 64-thread block cannot issue wgmma at all; cutlass will refuse to instantiate.

## 5. F32 accumulator only (on Hopper sm_90a)

The `wgmma.mma_async.sync.aligned.m64nNkK.f32.f16.f16.f32` and similar mnemonics all return **F32** accumulator regardless of input dtype on sm_90a. Do not try to declare a `cute::Tensor<float16_t, …>` accumulator and expect wgmma to emit the f16 form — it doesn't on Hopper. (Blackwell sm_100 has different rules; not relevant here.)

## 6. Kernel-name template-mangling explosion at compile time

The mangled name of the cutlass cooperative kernel is several KB of template parameters. nvcc's symbol table can compile them in 30-60 s for a single kernel, and bypassing build caches recompiles them every iteration. Set up `ccache` or use the cutlass profiler to amortize. The single-kernel build in our `build.sh` takes ~25 s on H200.

## 7. ncu's `wgmma`-specific counter is unavailable on driver 570.124.06

`smsp__inst_executed_pipe_wgmma.sum` returns `n/a` on this stack. Use `sm__inst_executed_pipe_tensor_op_hmma.sum` as the proxy — it correctly counts wgmma issues on Hopper. (The "hmma" name is a holdover from Ampere; on Hopper this counter folds wgmma in.)

## 8. Profiler attach overhead distorts kernel time

Cutlass's own timer reports ~0.608 ms for the example GEMM unprofiled, ~0.799 ms under ncu instrumentation (~30 % overhead). When comparing wgmma shape variants, take both numbers from the same instrumentation level — never compare `--metrics …` runs against unprofiled runs.

## 9. Producer/consumer warpgroup deadlocks

If a producer warpgroup increments an mbarrier the consumer is waiting on by **fewer** than the consumer's expected arrival count, the entire CTA hangs forever (no timeout, no error). Mirror cutlass's `Sm90Pipeline` arrival counts exactly; do not "simplify" them.
