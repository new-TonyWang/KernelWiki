---
id: exp-README
type: experience
vendor: nvidia
title: Readme
probe_slug: README
evidence_level: measured
measured_on:
  device: H200
  sm: sm_90a
  cuda_runtime: '12.8'
  driver: '570'
---
# artifacts/experience/hw-probes/

Hardware microbench probe records. Sibling of `artifacts/experience/api-probes/`. Schema: `../../templates/frontmatter/experience.yaml` (shared with api-probes).

## Two data sources co-exist

```
hw-probes/
├── canonical/                              ← filled by `kp_introspect microbench`
│   ├── compute-latency/
│   │   ├── fma-rn-f32.json
│   │   ├── shfl-sync-bfly.json
│   │   └── ...
│   └── memory-latency/
│       ├── smem-same-bank.json
│       ├── l1.json
│       ├── l2.json
│       └── hbm.json
│
├── <instruction-slug>/                     ← filled by KB-gen agent during skill build
│   └── <YYYY-MM-DD>-<task-slug>.md         ← per-skill probe record
│       (e.g. shfl-sync-bfly/2026-04-15-warp-primitives.md)
│
└── <memory-level-slug>/                    ← filled by KB-gen agent during skill build
    └── <YYYY-MM-DD>-<task-slug>.md         ← per-skill probe record
        (e.g. hbm/2026-04-15-coalescing.md)
```

### `canonical/` — generic baseline (produced by CLI)

Produced by `kp_introspect microbench compute-latency` and `kp_introspect microbench memory-latency` (see booklet 07 §R.4.5.1 and §R.4.5.2). These files are **general-purpose** — one fma, one shuffle, one smem-same-bank read, etc. — and they feed `00-foundation/hardware-spec/h200-specs.md` with a canonical cycle table.

JSON format matches the `microbench` block in the `kp_introspect bundle` schema (see booklet 07 §R.6).

### `<instruction-slug>/` and `<memory-level-slug>/` — skill-specific (produced by agent)

Produced by the KB-gen agent during M4c / M5 while building a skill. Unlike canonical data, these records exercise the **exact pattern** the skill uses: e.g., `shfl.sync.bfly` with the specific mask chosen for warp-reduce, or `ld.shared` with the specific stride pattern used by a bank-conflict skill.

Format: Markdown with `experience.yaml` frontmatter, body sections per `api-probing.md` Step 5 (Summary / Minimal Kernel / Build / Measurement / Introspection / Notes).

## Who reads what

- **`h200-specs.md` (auto-generated section)**: reads `canonical/**` for the reference latency table shown at the top of the hardware spec.
- **`wiki/nvidia/foundations/<family>/<skill>/skill.md` (§ Measured Characteristics)**: reads the skill-specific probe records and links to them as evidence for narrative claims.
- **`lint_knowledge.py`**: validates that every bullet under `## Measured Characteristics` of a skill resolves to an existing probe record under `artifacts/experience/hw-probes/...`.

## Why two sources instead of one

Canonical data is cheap, broad, and boring — it answers "what does this instruction cost on H200, in isolation?". It cannot answer the skill-specific question "what does it cost in **this** access pattern, at **this** warp participation, with **this** predicate?". The agent's skill-specific probe does that. They complement each other:

- If the skill-specific probe disagrees with canonical, that **is** the skill's contribution — the probe record's `## Notes` should explain why.
- If they agree, the skill can still cite the canonical number for pedagogical context.
