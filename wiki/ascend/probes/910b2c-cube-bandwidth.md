---
id: exp-ascend910b2c-cube-bandwidth
title: "Ascend 910B2C Cube Memory-Path Bandwidth"
type: experience
vendor: ascend
probe_slug: ascend910b2c-cube-bandwidth
evidence_level: measured
measured_on:
  device: Ascend 910B2C
  cann: /usr/local/Ascend/cann-8.5.1
  soc_compile_target: Ascend910B2
  date: '2026-06-22'
source:
- path: sources/experience/hw-probes/ascend910b2c-bandwidth.md
  anchor: Valid Cube-side Results
architectures:
- ascend910b2
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
tags:
- ascendc
- ai-core
- cube-unit
- l1-buffer
- l0a
- l0b
confidence: experimental
artifact_dir: artifacts/experience/hw-probes/ascend910b2c-bandwidth
---
# Ascend 910B2C Cube Memory-Path Bandwidth

This page summarizes a local AscendC bandwidth probe for Ascend 910B2C Cube-side data movement paths. The source record is `sources/experience/hw-probes/ascend910b2c-bandwidth.md` (`exp-ascend910b2c-bandwidth`).

## Key measured reference points

| Path | Scope | GB/s | GiB/s | Notes |
| --- | --- | ---: | ---: | --- |
| GM -> A1/L1 via MTE2 | 24 AIC aggregate | 1540.560 | 1434.758 | 512KiB chunk, 1GiB work bytes, 32 iterations |
| GM -> B1/L1 via MTE2 | 24 AIC aggregate | 1540.526 | 1434.726 | 512KiB chunk, 1GiB work bytes, 32 iterations |
| A1/L1 -> A2/L0A via MTE1 | 24 AIC aggregate | 9806.495 | 9133.010 | 64KiB chunk, 1.5MiB work bytes, 65536 iterations |
| B1/L1 -> B2/L0B via MTE1 | 24 AIC aggregate | 5196.829 | 4839.924 | 64KiB chunk, 1.5MiB work bytes, 65536 iterations |
| GM -> A1/L1 via MTE2 | 1 AIC | 133.175 | 124.029 | 512KiB chunk, 1GiB work bytes, 8 iterations |
| GM -> B1/L1 via MTE2 | 1 AIC | 131.758 | 122.709 | 512KiB chunk, 1GiB work bytes, 8 iterations |
| A1/L1 -> A2/L0A via MTE1 | 1 AIC | 408.733 | 380.662 | 64KiB chunk, 64KiB work bytes, 65536 iterations |
| B1/L1 -> B2/L0B via MTE1 | 1 AIC | 216.593 | 201.718 | 64KiB chunk, 64KiB work bytes, 65536 iterations |

## Practical implications for AscendC Cube kernels

- Treat `GM -> A1/B1` plus `A1/B1 -> A2/B2` as the validated matmul-style movement path for Cube-side staging.
- For aggregate 24-AIC measurements, L1-to-L0A bandwidth is about **1.89x** L1-to-L0B bandwidth in this probe (`9806.495 / 5196.829`).
- Direct `GM -> L0A/L0B` modes in the benchmark source are experimental; host-timed runs produced impossible values and should not be used as bandwidth references.
- The 24-AIC `GM -> A1/B1` aggregate values are about **1.54 TB/s decimal** for this protocol, while single-AIC values are about **132-133 GB/s decimal**.

## Archived benchmark

The archived benchmark includes host launch code, AscendC kernels, tiling/mode headers, run script, and the raw result report under:

```text
artifacts/experience/hw-probes/ascend910b2c-bandwidth/
```

Reproduction command from the original environment:

```bash
cd npu/benchmarks/910b_bandwidth
ASCEND_RT_VISIBLE_DEVICES=3 ./run.sh 1GiB 7 0
```
