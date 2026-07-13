---
id: exp-attention-tuning
type: experience
vendor: nvidia
title: Readme
probe_slug: attention-tuning
evidence_level: measured
measured_on:
  device: H200
  sm: sm_90a
  cuda_runtime: '12.8'
  driver: '570'
architectures:
- sm90
- sm90a
kernel_types:
- attention
- flash-attention
confidence: experimental
tags:
- attention
- flash-attention
---
# attention-tuning

Kernel-parameter tuning sweeps for attention operators on H200 (sm_90a).

## Probe records

| Date | Record | Operator | Swept axis |
|------|--------|----------|-----------|
| 2026-05-08 | `2026-05-08-attention-tile-sweep.md` | MVP flash-attention | BLOCK_M x BLOCK_N |
