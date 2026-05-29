---
id: code-cutlass-cute-attention-fmha-example-tuning
type: code-walkthrough
vendor: nvidia
title: Tuning
upstream_repo: NVIDIA/cutlass-cute
---
# cutlass 88_hopper_fmha Tuning Parameters

## Compile-time knobs (via `fmha_options.hpp` Tag system)

| Knob | Tag | Default | Allowed values | Effect |
|---|---|---|---|---|
| Head dimension | `kHeadDim` (template param) | 128 | 64, 128, 256 | Determines the K/inner dimension of Q@K^T. Larger D increases smem per tile. |
| MMA warpgroups | `Tag::kNumMmaWarpGroups` | 2 | 1, 2 | Number of consumer warpgroups computing wgmma. Default 2 in the warp-specialized collective. 1 halves compute parallelism but reduces register/smem pressure. |
| Persistent kernel | `Tag::kIsPersistent` | false | true, false | Enables persistent-kernel scheduling (CTAs loop over work items instead of one-shot). Improves occupancy for small batch/head counts. |
| Q pipeline stages | `Tag::kStagesQ` | auto | 1, 2 | Number of smem pipeline stages for Q tiles. 2 enables double-buffering. |
| KV pipeline stages | `Tag::kStagesKV` | auto | 2, 3, 4 | Number of smem pipeline stages for K/V tiles. More stages hide TMA latency. |
| Cluster M | `Tag::kClusterM` | 1 | 1, 2, 4 | CTA cluster size in the M (Q-tile) dimension. Enables TMA multicast across clustered CTAs. |
| Blocks per SM | `Tag::kBlocksPerSM` | auto | 1, 2 | Max CTAs co-resident per SM. Higher increases occupancy but reduces per-CTA smem/register budget. |
| Q loaded separately | `Tag::kLoadsQSeparately` | false | true, false | If true, Q is loaded by a dedicated producer warpgroup; otherwise Q shares the K/V producer. |
| Mainloop locked | `Tag::kIsMainloopLocked` | false | true, false | If true, mainloop uses lock-based synchronization instead of mbarrier. |
| Epilogue locked | `Tag::kIsEpilogueLocked` | false | true, false | Same as above for epilogue. |
| Accumulator QK | `Tag::kAccQK` | f32 | f32, f16 | Accumulator precision for the S=Q@K^T product. f16 saves registers but loses precision. |
| Epilogue kind | `Tag::kEpilogueKind` | default | default, residual | Controls epilogue behavior (e.g., residual connections). |

## Runtime parameters

| Parameter | CLI flag | Default | Effect |
|---|---|---|---|
| Batch size | `--b` | 16 | Number of sequences in the batch. |
| Number of heads | `--h` | 16 | Number of attention heads. |
| Query sequence length | `--q` | 1024 | Length of the Q sequence. |
| Key/Value sequence length | `--k` | 1024 | Length of the K/V sequence. |
| Head dimension | `--d` | 128 | Must match compile-time `kHeadDim`. |
| Attention mask | `--mask` | `no` | Masking mode: `no` (full attention), `causal` (triangular), or `residual`. Parsed in `88_hopper_fmha.cu` main function. |
| Backward pass | `--bwd` | false | Run backward pass in addition to forward. |
| Verification | `--verify` | false | Compare against CPU/GPU reference. |

## Tile shape derivation

The tile shape is derived from compile-time parameters:
- **BLOCK_M** = 64 × `kNumMmaWarpGroups` (64 for 1 warpgroup, 128 for 2)
- **BLOCK_N** = derived from head dimension and smem budget
- **K-dimension** = `kHeadDim` (fixed per tile iteration for Q@K^T)

## Interaction map

```
kNumMmaWarpGroups ─→ BLOCK_M (tile height)
                   ─→ register pressure (more warpgroups = more accumulators)
                   ─→ smem per CTA (Q tile grows with BLOCK_M)

kStagesKV ─────────→ smem per CTA (K/V buffers multiply)
                   ─→ TMA latency hiding (more stages = better hiding)
                   ─→ kBlocksPerSM (more stages = fewer co-resident CTAs)

kClusterM ─────────→ TMA multicast efficiency
                   ─→ grid utilization (cluster must divide grid-M evenly)

kIsPersistent ─────→ kernel scheduling model
                   ─→ useful when grid is smaller than SM count
```

## Source reference

All knobs are defined in `kernel/fmha_options.hpp` as `Tag` enum values and consumed by `kernel/fmha_kernel_builder.hpp` via `find_option_t<Tag, Default, Options...>`.
