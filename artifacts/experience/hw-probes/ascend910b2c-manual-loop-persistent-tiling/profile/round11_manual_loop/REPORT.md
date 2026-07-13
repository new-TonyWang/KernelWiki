# Manual Multi-Tile Loop / Persistent Blocks Experiment

## Question

Can we reduce scalar/control overhead by reducing launch `Block Num` and processing multiple logical tiles inside each Triton program with a manual loop? If yes, can a kernel with launch blocks no more than the vector-core count implement the same work as the high-block-count version?

## Hardware fact used

Project-local KernelWiki page `hw-ascend910b2c` says Ascend 910B2C has 24 AI Core and 48 VEC units total. This experiment therefore uses `48` as the vector-core-count launch limit.

## Setup

- Case: `case_024_bfloat16_15x255x1x1x256x8`
- Elements: `7,833,600`
- Logical tile size: `BLOCK=4096`
- Logical tile count: `ceil(7,833,600 / 4096) = 1913`
- Device: Ascend 910B, `ASCEND_RT_VISIBLE_DEVICES=3`
- Safety log: `profile/round11_manual_loop/run.log`
- Script: `profile/round11_manual_loop/scripts/manual_loop_case024.py`
- Summary CSV: `profile/round11_manual_loop/analysis/summary.csv`

All variants are compared against `loop1` output. The script reported `exact_equal_vs_loop1=True` for every multi-tile and persistent variant. `max_abs_vs_loop1` is `nan` because both equal tensors contain matching Inf values in this workload; exact tensor equality is the stronger evidence here.

## Variants

- `loop1`: one logical tile per Triton program, grid `1913` blocks.
- `loopK`: each Triton program processes `K` contiguous logical tiles via a static `for` loop, grid `ceil(1913 / K)` blocks.
- `persistent_48`: launch exactly `48` programs and have each program process strided tile IDs `pid + i * 48` for `ceil(1913 / 48)=40` iterations.

## Results

| Variant | Launch blocks | Event us | Prof duration us | Scalar us | Scalar ratio | Vec ratio | Exact equal |
|---|---:|---:|---:|---:|---:|---:|---|
| `loop1` | 1913 | 195.120 | 166.186 | 59.787 | 0.364 | 0.342 | yes |
| `loop2` | 957 | 130.570 | 88.764 | 13.870 | 0.161 | 0.644 | yes |
| `loop4` | 479 | 125.490 | 83.384 | 10.622 | 0.132 | 0.682 | yes |
| `loop8` | 240 | 122.100 | 81.404 | 9.326 | 0.119 | 0.699 | yes |
| `loop16` | 120 | 133.230 | 92.863 | 10.147 | 0.110 | 0.712 | yes |
| `loop32` | 60 | 166.690 | 121.085 | 12.211 | 0.101 | 0.721 | yes |
| `loop40` | 48 | 120.120 | 77.403 | 6.710 | 0.090 | 0.728 | yes |
| `persistent_48` | 48 | 119.970 | 78.603 | 7.068 | 0.094 | 0.723 | yes |

## Interpretation

1. Manual multi-tile loop is effective. Moving from `loop1` to `loop8` reduces launch blocks from `1913` to `240`, event time from `195.120 us` to `122.100 us`, and scalar time from `59.787 us` to `9.326 us`.

2. The relationship is not monotonic for every `K`. `loop16` and especially `loop32` regress despite lower block counts, likely because too much serial work per program reduces available parallelism or increases per-program pressure. Thus the best loop factor must be measured, not assumed.

3. The <= vector-core-count experiment succeeds. `loop40` launches `48` blocks, which is equal to the 910B2C VEC total. It remains exact-equal to `loop1` and has the best measured event time in this run (`120.120 us`) with very low scalar time (`6.710 us`).

4. The persistent strided version also succeeds. `persistent_48` launches exactly `48` blocks and processes all `1913` logical tiles by striding over tile IDs. It is exact-equal to `loop1` and almost identical to `loop40`: event `119.970 us`, profiler duration `78.603 us`, scalar `7.068 us`.

## Answer

Yes: reducing `Block Num` by looping over multiple logical tiles inside the kernel substantially reduces scalar/control overhead for `case_024`. Yes: with launch block count constrained to `48` (the 910B2C VEC count), both contiguous-loop (`loop40`) and persistent-strided (`persistent_48`) variants implement the same logical-tile work as the large-block-count version and are exact-equal to it.

## Candidate implication

A production candidate should consider a large-bf16 route using a persistent/looped kernel for case_024-like shapes. It must still pass full 39-case correctness and benchmark gates because this experiment only validates isolated case_024 behavior and timing.
