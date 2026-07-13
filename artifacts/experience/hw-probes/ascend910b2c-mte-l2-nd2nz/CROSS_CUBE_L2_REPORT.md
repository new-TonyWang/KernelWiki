# Cross-cube L2 reuse probe on Ascend 910B2C

## What was tested

I added two multi-cube modes to the existing AscendC/ACLNN MTE benchmark:

- `mode=20`: blockDim=2, two AIC/Cube cores issue `LoadData` GM/L2 -> L0A from the **same addresses**.
- `mode=21`: blockDim=2, two AIC/Cube cores issue `LoadData` GM/L2 -> L0A from offset/disjoint-ish addresses.

Timing uses the same `aclrtEventElapsedTime` host launcher. Each cube performs `loops=4096`, `bytesPerLoop=32768` unless otherwise stated.

## Current measured data

Representative fixed-traffic results from `host_results/cross_cube_l2_raw.txt`:

| working set | warmup | mode0 1 cube | mode20 2 cubes same addr | mode21 2 cubes shifted addr | observation |
|---:|---:|---:|---:|---:|---|
| 16 MiB | cold | 636.99 us | 638.81 us | 643.84 us | 2 cubes finish in almost same wall time as 1 cube |
| 16 MiB | hot | 443.91 us | 441.35 us | 451.45 us | hot L2/SLC state benefits both cubes |
| 64 MiB | cold | 710.33 us | 712.31 us | 703.42 us | no same-address advantage visible |
| 64 MiB | hot | 437.30 us | 458.15 us | 446.23 us | 2-cube wall time close to 1-cube hot |
| 128 MiB | cold | 785.27 us | 787.12 us | 806.51 us | close |
| 128 MiB | hot | 453.37 us | 452.41 us | 450.76 us | close |
| 256 MiB | cold | 811.29 us | 798.35 us | 783.05 us | close |
| 256 MiB | hot | 442.00 us | 463.01 us | 452.86 us | close |

Full-hot capacity-style samples:

| per-cube working set | mode20 same addr | mode21 shifted/disjoint-ish |
|---:|---:|---:|
| 128 MiB | 456.68 us | 455.62 us |
| 160 MiB | 769.68 us | 559.10 us |
| 192 MiB | 1142.26 us | 1088.24 us |
| 224 MiB | 1506.32 us | 1656.64 us |
| 256 MiB | 1916.68 us | 1931.96 us |

## Conclusion

For GM/L2 -> L0A `LoadData`, **L2/SLC hot data is visible to multiple cube cores** in the sense that two cubes reading hot data complete in nearly the same wall time as one cube for the same per-cube loop count. So the cache is not behaving like a strictly private per-cube cache.

However, **simultaneous same-address reads by two cubes did not show a clear additional reuse/multicast advantage** over shifted/disjoint-ish reads. Mode20 and mode21 are usually within measurement noise or within ~5%; sometimes mode20 is slightly faster, sometimes slower. This suggests the bottleneck in this microbenchmark is mainly per-cube MTE/LoadData pipeline and shared L2/SLC service, not duplicate HBM fetches per cube.

Practical answer: **yes, L2/SLC can be reused/shared across cube cores for hot data, but do not expect same-line cross-cube reads to give a strong extra speedup beyond making the data hot in L2/SLC.** For scheduling, it is still worth arranging temporal locality so neighboring cubes consume recently loaded tiles, but the bigger win remains avoiding cold misses and staying under the ~160--192 MiB effective hot capacity observed earlier.

## Caveat / stronger test prepared

A stricter sequential test was prepared in `aclnn_mte_host.cpp`:

- `mode=30`: first warm with single-cube `mode=0`, then time two-cube same-address `mode=20` on the same allocation.
- `mode=31`: first warm with single-cube `mode=0`, then time two-cube disjoint `mode=21`.

This was not run because `npu-smi` later showed another process on NPU 9, and per the safety rule I stopped launching new NPU work.

## Raw data

- `host_results/cross_cube_l2_raw.txt`
- source: `mte_microbench_proj/op_kernel/mte_micro_bench.cpp`
- host launcher: `aclnn_mte_host.cpp`
