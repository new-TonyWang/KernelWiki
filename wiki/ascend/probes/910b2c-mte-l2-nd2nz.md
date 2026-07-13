---
id: exp-ascend910b2c-mte-l2-nd2nz-summary
title: "Ascend 910B2C MTE/L2/ND2NZ Movement Probe"
type: experience
vendor: ascend
probe_slug: ascend910b2c-mte-l2-nd2nz
evidence_level: measured
measured_on:
  device: Ascend 910B2C
  cann: /usr/local/Ascend/cann-9.0.0
  soc_compile_target: ascend910b
  date: '2026-07-05'
source:
- path: sources/experience/hw-probes/ascend910b2c-mte-l2-nd2nz.md
  anchor: Main measured results
architectures:
- ascend910b2
- ascend910b2c
languages:
- ascendc
hardware_features:
- ai-core
- cube-unit
- l1-buffer
- l0a
- l0b
- ub
techniques:
- tiling-optimization
- load-order-optimization
- data-reuse
- cache-policy
tags:
- ascendc
- ai-core
- cube-unit
- l1-buffer
- l0a
- l0b
confidence: experimental
artifact_dir: artifacts/experience/hw-probes/ascend910b2c-mte-l2-nd2nz
---
# Ascend 910B2C MTE/L2/ND2NZ Movement Probe

This page summarizes a local AscendC/ACLNN microbenchmark for 910B2C Cube-side movement paths. The detailed source record is `sources/experience/hw-probes/ascend910b2c-mte-l2-nd2nz.md` (`exp-ascend910b2c-mte-l2-nd2nz`).

## Key measured reference points

All main rows use fp16 32 KiB per loop, `loops=4096`, 16 MiB input allocation unless noted, and `aclrtEventElapsedTime` around repeated ACLNN custom-op launches.

| Path | Mode | Logical GB/s | Average time | Practical interpretation |
| --- | ---: | ---: | ---: | --- |
| GM/L2 -> L0A via `LoadData` | 0 | 300.46 | 446.688 us | Fast direct Cube feed path |
| GM/L2 -> L1 raw via `DataCopy` | 1 | 148.80 | 902.024 us | L1 staging is about half of direct L0A feed |
| GM ND -> L1 NZ via `DataCopy(Nd2NzParams)` | 2 | 115.57 | 1161.284 us | ND2NZ into L1 is slower than raw L1 copy |
| GM/L2 -> L0A + GM/L2 -> L1 on same AIC | 3 | 192.43 | 1394.984 us | Mostly shared/serialized MTE2/upstream path |
| L1 -> L0A via `LoadData` | 4 | 367.82 | 364.852 us | L0A direction is faster |
| L1 -> L0B via `LoadData` | 5 | 204.65 | 655.820 us | L0B direction is ~0.56x L0A |
| L1 -> L0A + L1 -> L0B on same AIC | 6 | 281.40 | 953.852 us | Partial overlap, still contended |
| GM/L2 -> L0A transpose (`ifTranspose=true`) | 7 | 303.87 | 441.676 us | Best one-use A-transpose path |
| GM ND -> L1 NZ then L1 -> L0A | 8 | 175.96 | 1525.692 us | Useful only if L1/NZ reuse amortizes ND2NZ |

## L2/SLC and cache behavior

- Hot vs cold matters: cold GM/L2 -> L0A for 128 MiB logical traffic is roughly 0.93-1.21 ms; hot is roughly 0.42-0.47 ms in this protocol.
- The effective hot-capacity cliff for this single-core MTE2 -> L0A stream starts between **160 MiB and 192 MiB**.
- Cross-cube tests show that L2/SLC hot data is visible across cube cores, but simultaneous same-address two-cube reads did not show a strong extra multicast advantage over shifted/disjoint-ish reads.

## GEMM/A-transpose implications

- For one-use A tiles, prefer direct GM/L2 -> L0A `LoadData(ifTranspose=true)`.
- For A tiles reused about **16 times or more**, the `GM ND -> L1 NZ` cost can be amortized and L1/NZ staging may become attractive.
- Do not assume GM/L2 -> L0A and GM/L2 -> L1 can independently double throughput on the same AIC; the measured combined path is close to serialization.
- L1 -> L0A is substantially faster than L1 -> L0B in this probe; schedule/reuse accordingly when choosing A/B tile residency.

## Archived benchmark

```text
artifacts/experience/hw-probes/ascend910b2c-mte-l2-nd2nz/
```

Important files:

- `RESULTS_20260705.md` — full report.
- `CROSS_CUBE_L2_REPORT.md` — cross-cube L2/SLC reuse follow-up.
- `kernel/mte_micro_bench.cpp` — AscendC kernel modes.
- `aclnn_mte_host.cpp` — ACLNN host launcher and timer.
- `host_results/*.txt` — raw timing data.
