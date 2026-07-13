---
id: exp-ascend910b2c-bandwidth
title: "Ascend 910B2C Cube Bandwidth Probe"
type: experience
vendor: ascend
probe_slug: ascend910b2c-bandwidth
status: verified
kind: local-benchmark
trigger: local-benchmark
api: DataCopy/LoadData
namespace: ascendc
evidence_level: measured
clock_policy: unknown
measured_on:
  device: Ascend 910B2C
  host: ssh 910b
  physical_device: 3
  process_device: 0
  cann: /usr/local/Ascend/cann-8.5.1
  soc_compile_target: Ascend910B2
  date: '2026-06-22'
architectures:
- ascend910b2
languages:
- ascendc
techniques:
- tiling-optimization
- load-order-optimization
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
artifact_dir: artifacts/experience/hw-probes/ascend910b2c-bandwidth
artifacts:
  report: artifacts/experience/hw-probes/ascend910b2c-bandwidth/RESULTS_20260622.md
  host: artifacts/experience/hw-probes/ascend910b2c-bandwidth/main.cpp
  cube_kernel: artifacts/experience/hw-probes/ascend910b2c-bandwidth/kernel/cube_bandwidth_kernel.h
  run: artifacts/experience/hw-probes/ascend910b2c-bandwidth/run.sh
conclusions:
  timing_method: host steady_clock around one kernel launch plus aclrtSynchronizeStream; median of 7 repeats
  valid_paths:
    gm_to_a1_l1_mte2_24_aic_gbps: 1540.560
    gm_to_b1_l1_mte2_24_aic_gbps: 1540.526
    l1_to_l0a_mte1_24_aic_gbps: 9806.495
    l1_to_l0b_mte1_24_aic_gbps: 5196.829
    gm_to_a1_l1_mte2_1_aic_gbps: 133.175
    gm_to_b1_l1_mte2_1_aic_gbps: 131.758
    l1_to_l0a_mte1_1_aic_gbps: 408.733
    l1_to_l0b_mte1_1_aic_gbps: 216.593
  invalid_or_unreported_paths:
  - Direct GM -> L0A/L0B modes were present but produced impossible host-timed values and are not reported as valid bandwidth.
---
# Ascend 910B2C Cube Bandwidth Probe

This local benchmark measures AscendC Cube-side data movement bandwidth on an Ascend 910B2C device. It uses a host timer around one kernel launch plus `aclrtSynchronizeStream` and reports the median of 7 repeats.

## Environment

- Host: `ssh 910b`
- CANN: `/usr/local/Ascend/cann-8.5.1`
- Device: physical NPU `3` (`910B2C`), exposed as process device `0` via `ASCEND_RT_VISIBLE_DEVICES=3`
- SoC compile target: `Ascend910B2`
- Timing: host `steady_clock` around one kernel launch plus `aclrtSynchronizeStream`; median of 7 repeats

## Valid Cube-side Results

| Case | Path | Blocks | Active AIC | Chunk | Work bytes | Iterations | GB/s | GiB/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cube_gm_to_a1_l1_mte2_24_aic` | GM -> A1/L1, MTE2 | 24 | 24 | 512KiB | 1GiB | 32 | 1540.560 | 1434.758 |
| `cube_gm_to_b1_l1_mte2_24_aic` | GM -> B1/L1, MTE2 | 24 | 24 | 512KiB | 1GiB | 32 | 1540.526 | 1434.726 |
| `cube_l1_to_l0a_mte1_24_aic` | A1/L1 -> A2/L0A, MTE1 | 24 | 24 | 64KiB | 1.5MiB | 65536 | 9806.495 | 9133.010 |
| `cube_l1_to_l0b_mte1_24_aic` | B1/L1 -> B2/L0B, MTE1 | 24 | 24 | 64KiB | 1.5MiB | 65536 | 5196.829 | 4839.924 |
| `cube_gm_to_a1_l1_mte2_1_aic` | GM -> A1/L1, MTE2 | 1 | 1 | 512KiB | 1GiB | 8 | 133.175 | 124.029 |
| `cube_gm_to_b1_l1_mte2_1_aic` | GM -> B1/L1, MTE2 | 1 | 1 | 512KiB | 1GiB | 8 | 131.758 | 122.709 |
| `cube_l1_to_l0a_mte1_1_aic` | A1/L1 -> A2/L0A, MTE1 | 1 | 1 | 64KiB | 64KiB | 65536 | 408.733 | 380.662 |
| `cube_l1_to_l0b_mte1_1_aic` | B1/L1 -> B2/L0B, MTE1 | 1 | 1 | 64KiB | 64KiB | 65536 | 216.593 | 201.718 |

## Measurement interpretation

- The reliable Cube data path exposed by the AscendC matmul examples is `GM -> A1/B1` followed by `A1/B1 -> A2/B2`.
- Direct `GM -> L0A/L0B` modes exist in the source but host-timed runs produced impossible values, so those numbers are not treated as valid bandwidth evidence.
- The 24-AIC aggregate results are the most useful hardware-level reference points; the 1-AIC results provide a per-core sanity check.

## Reproduce

```bash
cd npu/benchmarks/910b_bandwidth
ASCEND_RT_VISIBLE_DEVICES=3 ./run.sh 1GiB 7 0
```

The archived code and result report are in `artifacts/experience/hw-probes/ascend910b2c-bandwidth/`.
