---
id: exp-ascend910b2c-mte-l2-nd2nz
title: "Ascend 910B2C MTE/L2/ND2NZ Microbenchmark"
type: experience
vendor: ascend
probe_slug: ascend910b2c-mte-l2-nd2nz
status: verified
kind: local-benchmark
trigger: local-benchmark
api: LoadData/DataCopy/Nd2NzParams
namespace: ascendc
evidence_level: measured
clock_policy: unknown
measured_on:
  device: Ascend 910B2C
  host: ssh 910b
  physical_device: 3
  cann: /usr/local/Ascend/cann-9.0.0
  soc_compile_target: ascend910b
  date: '2026-07-05'
architectures:
- ascend910b2
- ascend910b2c
languages:
- ascendc
techniques:
- tiling-optimization
- load-order-optimization
- data-reuse
- cache-policy
hardware_features:
- ai-core
- cube-unit
- l1-buffer
- l0a
- l0b
- ub
tags:
- ascendc
- ai-core
- cube-unit
- l1-buffer
- l0a
- l0b
confidence: experimental
artifact_dir: artifacts/experience/hw-probes/ascend910b2c-mte-l2-nd2nz
artifacts:
  report: artifacts/experience/hw-probes/ascend910b2c-mte-l2-nd2nz/RESULTS_20260705.md
  cross_cube_report: artifacts/experience/hw-probes/ascend910b2c-mte-l2-nd2nz/CROSS_CUBE_L2_REPORT.md
  host: artifacts/experience/hw-probes/ascend910b2c-mte-l2-nd2nz/aclnn_mte_host.cpp
  kernel: artifacts/experience/hw-probes/ascend910b2c-mte-l2-nd2nz/kernel/mte_micro_bench.cpp
  raw_modes: artifacts/experience/hw-probes/ascend910b2c-mte-l2-nd2nz/host_results/modes_raw.txt
  raw_l2: artifacts/experience/hw-probes/ascend910b2c-mte-l2-nd2nz/host_results/l2_cold_hot_raw.txt
  raw_cross_cube: artifacts/experience/hw-probes/ascend910b2c-mte-l2-nd2nz/host_results/cross_cube_l2_raw.txt
conclusions:
  timing_method: aclrtEventElapsedTime around repeated ACLNN custom-op launches; usually 5 repeats with 1 warmup for main table
  tile_bytes: 32768
  large_shape_input: 8388608 fp16 elements
  single_aic_logical_gbps:
    gm_l2_to_l0a_loaddata: 300.46
    gm_l2_to_l1_raw_datacopy: 148.80
    gm_nd_to_l1_nz_nd2nz: 115.57
    l1_to_l0a: 367.82
    l1_to_l0b: 204.65
    gm_l2_to_l0a_transpose: 303.87
  contention:
    gm_l2_to_l0a_plus_l1_same_aic: mostly serial/shared upstream bandwidth
    l1_to_l0a_plus_l0b_same_aic: partial overlap but still contended
  l2_effective_hot_capacity_mib: 160-192
  cross_cube_l2_reuse: hot L2/SLC data is visible across cube cores, but same-address two-cube reads did not show strong extra multicast advantage
---
# Ascend 910B2C MTE/L2/ND2NZ Microbenchmark

This local benchmark measures single-AIC and two-AIC Cube-side data movement paths on Ascend 910B2C using an AscendC custom op and an ACLNN host launcher. It was designed to answer four practical questions:

1. MTE2 bandwidth from GM/L2 to L0A and to L1, and whether same-core paths contend.
2. MTE1 bandwidth from L1 to L0A and L0B, and whether same-core paths contend.
3. L2/SLC hot/cold and capacity effects on copy pipelines.
4. Large-shape A-transpose movement strategy, including direct L2->L0A transpose vs L1/NZ staging.

## Environment

- Host: `ssh 910b`
- Device: Ascend 910B2C, physical device id 3
- CANN: `/usr/local/Ascend/cann-9.0.0`
- Timing: `aclrtEventElapsedTime` around repeated ACLNN custom-op launches
- Main tile: fp16 32 KiB per loop (`bytesPerLoop=32768`)
- Main large-shape input: 8,388,608 fp16 elements (16 MiB allocation)

## Main measured results

| Path | Mode | Average time | Logical traffic | Logical GB/s | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| GM/L2 -> L0A via `LoadData` | 0 | 446.688 us | 128 MiB | 300.46 | MTE2-style direct Cube feed |
| GM/L2 -> L1 raw via `DataCopy` | 1 | 902.024 us | 128 MiB | 148.80 | L1 staging path |
| GM ND -> L1 NZ via `DataCopy(Nd2NzParams)` | 2 | 1161.284 us | 128 MiB | 115.57 | ND2NZ transform into L1 |
| GM/L2 -> L0A plus GM/L2 -> L1 on same AIC | 3 | 1394.984 us | 256 MiB | 192.43 | Mostly shared/serialized upstream bandwidth |
| L1 -> L0A via `LoadData` | 4 | 364.852 us | 128 MiB | 367.82 | Faster MTE1 direction |
| L1 -> L0B via `LoadData` | 5 | 655.820 us | 128 MiB | 204.65 | ~0.56x L0A bandwidth |
| L1 -> L0A plus L1 -> L0B on same AIC | 6 | 953.852 us | 256 MiB | 281.40 | Partial overlap, still contended |
| GM/L2 -> L0A with `ifTranspose=true` | 7 | 441.676 us | 128 MiB | 303.87 | Best one-use A-transpose path |
| GM ND -> L1 NZ then L1 -> L0A | 8 | 1525.692 us | 256 MiB | 175.96 | Too slow unless A tile reuse amortizes ND2NZ |

## L2/SLC observations

- Cold vs hot GM/L2 -> L0A differs by roughly **2.2x** in this protocol: cold 128 MiB logical traffic is about 0.93-1.21 ms, while hot is about 0.42-0.47 ms for 32 KiB..64 MiB allocations.
- Full-hot capacity-style runs show a cliff starting between **160 MiB and 192 MiB**:

| Hot working set | Loops | Average time | Logical GB/s |
| ---: | ---: | ---: | ---: |
| 128 MiB | 4096 | 437.82 us | 306.55 |
| 160 MiB | 5120 | 546.80 us | 306.82 |
| 192 MiB | 6144 | 866.44 us | 232.37 |
| 224 MiB | 7168 | 1486.34 us | 157.95 |
| 256 MiB | 8192 | 1927.02 us | 139.31 |

Treat the **effective hot capacity for this single-core MTE2->L0A stream as about 160 MiB**, with a transition region up to about 192 MiB. This is a measured behavior, not a public capacity specification.

## Cross-cube L2/SLC reuse

A two-cube follow-up used:

- `mode=20`: blockDim=2, both cubes read the same GM/L2 addresses into L0A.
- `mode=21`: blockDim=2, cubes read shifted/disjoint-ish addresses into L0A.

Representative fixed-traffic results:

| Working set | Warmup | 1 cube mode0 | 2 cubes same addr mode20 | 2 cubes shifted addr mode21 |
| ---: | ---: | ---: | ---: | ---: |
| 16 MiB | cold | 636.99 us | 638.81 us | 643.84 us |
| 16 MiB | hot | 443.91 us | 441.35 us | 451.45 us |
| 128 MiB | hot | 453.37 us | 452.41 us | 450.76 us |
| 256 MiB | hot | 442.00 us | 463.01 us | 452.86 us |

The data indicates that **L2/SLC hot data is visible across cube cores**: two cubes reading hot data finish in nearly the same wall time as one cube for the same per-cube loop count. However, same-address simultaneous reads do **not** show a strong additional multicast/reuse advantage over shifted/disjoint-ish reads.

## A-transpose strategy

For one-use large A tiles, direct `LoadData` into L0A with `ifTranspose=true` is best in this probe (~303.87 GB/s). The two-stage `GM ND -> L1 NZ -> L0A` path is much slower for one use.

Using the measured rates, the two-stage path only becomes attractive when the L1/NZ tile is reused many times. A simple amortization estimate gives a crossover near **16 uses**:

- Reuse < ~16: use direct GM/L2 -> L0A transpose.
- Reuse >= ~16: pretransform/cache A as NZ in L1 and feed L0A from L1 while pipelining future GM->L1 loads.

## Artifacts

The archived benchmark contains the report, host launcher, AscendC kernel, tiling header, op-host tiling code, run scripts, and raw timing data under:

```text
artifacts/experience/hw-probes/ascend910b2c-mte-l2-nd2nz/
```
