---
id: hw-foundation-runtime-introspection
title: Hardware Runtime Introspection
type: hardware
vendor: nvidia
architectures:
- sm90
- sm90a
tags:
- cuda-cpp
confidence: source-reported
related: []
sources: []
aliases: []
blackwell_relevance: Runtime introspection concepts apply to both Hopper and Blackwell
---
# Runtime Introspection (MVP subset of booklet 07 §R)

> This directory holds the contract and sample outputs for
> `tools/kp_introspect.py`. The tool is implemented in M2; this README is the
> stub contract it will populate.

## MVP scope

MVP implements **R1-R4 minus NVML**, which means four CLI subcommands, two of which are non-cached:

| Subcommand | Cached? | Purpose |
| --- | --- | --- |
| `device-static`   | by `(gpu_uuid, driver)` | compute-capability / sm_count / shmem limits / reg limits / L2 / memory bus |
| `kernel-static`   | by cubin sha256         | regs/thread / shmem_static / local_mem / ptx_version (from `cuFuncGetAttribute`) |
| `kernel-dynamic`  | no                      | occupancy / limit_factor for a given (block_size, smem) tuple |
| `bundle`          | no                      | aggregate of the above into one json |

**Deferred** (M5+):
- `device-dynamic` with NVML (clock/temp/power, `clock_policy` inference)
- cubin loader fallbacks (cuobjdump)
- multi-GPU bundle
- `regen-spec-md` auto-write

## Schema version

MVP starts at `schema_version: 0.1.0`. Anything under this schema is forward-compatible only — fields may be added, never removed or renamed, until `0.2.0`.

## Example bundle (filled in M3)

`examples/bundle-h200.json` will be produced by running:

```bash
python -m tools.kp_introspect bundle \
    --device 0 \
    --binary artifacts/warp_reduce/probe \
    --symbol warp_reduce_kernel \
    --block 256 \
    --smem 0 \
    --output 00-foundation/hardware-spec/runtime-introspection/bundle-h200.json
```

After M3 this file will be the canonical reference for what a "good" bundle looks like and what every field should contain on H200. The key H200 facts from these JSON examples are also mirrored into `../h200-specs.md` for easier reading.

## Consumed by

- `benchmark-protocol.md` — every measured entry must bind a bundle.
- `api-probing.md` Step 4 — every probe runs `kernel-static` at minimum.
- `bottleneck-triage.md` Q2 — reads `kernel.dynamic.limit_factor` when available.
- `tools/lint_knowledge.py` — verifies `artifacts.introspection` paths resolve to a bundle that contains the declared `measured_on` facts.
