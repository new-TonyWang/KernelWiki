---
id: code-cutlass-cute-fmha_skeleton
type: code-walkthrough
vendor: nvidia
title: Fmha_Skeleton
upstream_repo: NVIDIA/cutlass-cute
---
# cutlass 88_hopper_fmha Code Skeleton

A guided tour of the forward-pass code flow, referencing cutlass structs and functions by name (not by line number, per KB rules). All paths are relative to `{{CUTLASS_REPO_REF}}/examples/88_hopper_fmha/`.

## 1. Entry point: `88_hopper_fmha.cu`

The `run()` function:
- Parses CLI options (`Options` struct): `--b` (batch), `--h` (heads), `--q`/`--k` (sequence lengths), `--d` (head dimension).
- Instantiates the kernel type via `FmhaKernelBuilder`:

```cpp
using Kernel = typename cutlass::fmha::kernel::FmhaKernelBuilder<
    cutlass::half_t, cutlass::half_t,  // Element types for Q/K and V/O
    128,                                 // kHeadDim
    /* Options... */
>::Kernel;
```

- Wraps in `cutlass::fmha::device::DeviceUniversal<Kernel>`.
- Calls `device.run(args, workspace, stream)`.

## 2. Kernel builder: `kernel/fmha_kernel_builder.hpp`

`FmhaKernelBuilder<ElementQK, ElementVO, kHeadDim, Options...>`:
- Uses the `find_option_t<Tag, Default, Options...>` mechanism from `fmha_options.hpp`.
- Key compile-time constants resolved here:
  - `kNumMmaWarpGroups` (default: 2; controls consumer warpgroup count; see `find_option_t` in `fmha_collective_tma_warpspecialized.hpp`)
  - `kIsPersistent` (default: false; enables persistent kernel scheduling)
  - `kStagesQ`, `kStagesKV` (pipeline stage counts)
  - `kClusterM` (CTA cluster dimension)
  - `kBlocksPerSM` (max CTAs co-resident per SM)
- Dispatches to `FmhaKernelTmaWarpSpecialized` (the primary forward kernel).

## 3. Forward kernel: `kernel/fmha_kernel_tma_warpspecialized.hpp`

`FmhaKernelTmaWarpSpecialized`:
- `operator()` is the `__global__` kernel entry.
- Allocates shared memory for Q, K, V tiles and pipeline barriers.
- Calls the collective's `mainloop()` to execute the attention computation.
- Calls the collective's `epilogue()` to write output.

## 4. Tile scheduler: `kernel/fmha_tile_scheduler.hpp`

- Manages the mapping from CTA index to (batch, head, Q-tile) work items.
- For persistent kernels, implements work-stealing across the grid.

## 5. Forward collective: `collective/fmha_collective_tma_warpspecialized.hpp`

`FmhaCollectiveTmaWarpSpecialized`:
- This is the core of the FMHA algorithm. It implements a warp-specialized mainloop:
  - **Producer warpgroup**: issues TMA loads for K/V tiles into smem pipeline stages.
  - **Consumer warpgroup(s)**: execute wgmma for S=Q@K^T and O=P@V, plus online softmax.
- Key methods:
  - `load()`: TMA-driven K/V tile loading, managed by mbarrier-based pipeline.
  - `mma()`: wgmma-based matmul, accumulator management, softmax integration.
  - `epilogue()`: final rescale and writeback.

## 6. Softmax: `collective/fmha_collective_softmax.hpp`

- Implements the online softmax algorithm (FlashAttention-2 style).
- Tracks per-row running max and running sum.
- Performs accumulator rescaling when the max updates.
- Uses warp-level reductions (`__shfl_xor_sync`) for rowmax and rowsum across threads.

## 7. Epilogue: `collective/fmha_epilogue.hpp`

- Writes the final attention output from registers to global memory.
- Applies the final `1/l_i` normalization.
- Supports TMA store or direct global store depending on configuration.

## 8. Fusion hooks: `collective/fmha_fusion.hpp`

- Defines the interface for custom attention fusions (e.g., causal mask, ALiBi, custom score modifications).
- The default is `FmhaFusionNone`; `FmhaFusionCausal` implements causal masking.

## Navigation guide

| To understand... | Read... |
|---|---|
| The overall data flow | `88_hopper_fmha.cu` `run()` → `DeviceUniversal::run()` |
| How tiles are configured | `fmha_kernel_builder.hpp` (compile-time option resolution) |
| The mainloop algorithm | `fmha_collective_tma_warpspecialized.hpp` `load()` + `mma()` |
| Online softmax | `fmha_collective_softmax.hpp` |
| TMA pipeline management | `fmha_collective_load.hpp` |
| Tuning knobs | `fmha_options.hpp` (Tag enum) |
