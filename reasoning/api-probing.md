# API Probing Protocol (F10, skeleton)

> **Version**: MVP skeleton (~50 lines of executable rules). The full version
> (discrepancy handling, multi-kernel probes, NVML clock verification, source
> corpus formalization) is deferred until after MVP ships.
>
> **Sibling protocol**: `hardware-microbench.md` covers **active** microbench
> probes (cycle-level compute latency, pointer-chasing memory latency,
> whitepaper-driven feature sweeps). **This file** covers **single-API
> semantics** probes. They share benchmark-protocol.md, artifacts schema,
> and frontmatter, but they answer different questions:
> - `api-probing.md`: "what does this API do / how do I call it?"
> - `hardware-microbench.md`: "how fast is this instruction / memory level?"

The purpose of a probe is to turn an API that the knowledge base merely *lists* into an API that the knowledge base *demonstrates*. A probe is successful when, after it finishes, a downstream agent writing a kernel can copy-paste the probe record and get working, measured code.

## When to probe

Trigger a probe when any of these conditions is true:

- A `wiki/nvidia/api-definitions/<ns>/<func>.md` exists but has no `## End-to-End Example` section.
- A `wiki/nvidia/foundations/**/apis.md` lists an API that is absent from `wiki/nvidia/api-definitions/`.
- The gap queue (`sources/experience/api-probes/_gap-queue.md`) has a pending entry pointing at this API.
- A skill build task (M4c/M5) needs an API whose semantics you cannot fully justify from `upstream_scope` grep alone.

## Protocol (6 steps, all mandatory)

### Step 1 — Read upstream first

Before writing any experimental code, query the source corpus:

```bash
python3 -m scripts.source_corpus.cli search '<api-symbol>' --scope cuda-official
python3 -m scripts.source_corpus.cli search '<api-symbol>' --scope source-code/cuda-samples
python3 -m scripts.source_corpus.cli read 'cuda-official/toolkit-docs-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md' --anchor 'L120-L140'
```

Collect every hit as `{path, line_range}` and store it in the probe record's `referenced_in_corpus:` field. If zero hits, mark the probe `kind: undocumented`, write the stub, and stop — ask a human to decide scope.

### Step 2 — Write a minimal kernel

Rules for the probe kernel:

- One `.cu` file, ≤ 80 lines including host.
- Uses only the API under probe plus trivial scaffolding (malloc / H2D / launch / D2H / free).
- A fixed, documented launch config (grid + block + smem) — no auto-tuning here.
- Produces a numerically checkable output on the host.

Store the file under `<target_path>/artifacts/<slug>/probe.cu`.

### Step 3 — Compile

```bash
nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo -o probe probe.cu
```

Record the exact command in `artifacts.build` (either the literal command string or a path to a `build.sh`). On compile failure, record the ptxas/nvcc message verbatim in the probe record and set `status: failed`.

### Step 4 — Run, verify, measure

Follow `benchmark-protocol.md` strictly:
- 5 warmup launches (discarded).
- 20 measurement launches via CUDA events; record median + p10 + p90.
- Reference implementation (torch or hand-written scalar) compared with `max_abs_err` against the tolerance in `benchmark-protocol.md`.
- Run `kp_introspect kernel-static --binary probe --symbol <kernel>` and attach the JSON as `artifacts.introspection`.

### Step 5 — Write the probe record

Path: `sources/experience/api-probes/<YYYY-MM-DD>-<ns>-<func-slug>.md`. Frontmatter: exactly what `templates/frontmatter/experience.yaml` specifies — no extra fields, no missing fields. Body sections, in order:

1. `## Summary` — one paragraph: what was probed, on what hardware, result in one sentence.
2. `## Minimal Kernel` — the `.cu` source verbatim (fenced code block).
3. `## Build` — the exact nvcc command.
4. `## Measurement` — the benchmark table (per `benchmark-protocol.md` Pillar 4).
5. `## Introspection` — one-line summary plus path to the bundle json.
6. `## Notes` — pitfalls, reference hits from Step 1, open questions.

### Step 6 — Back-fill `wiki/nvidia/api-definitions/`

If `wiki/nvidia/api-definitions/<ns>/<func>.md` already exists: append a `## End-to-End Example` section with a link to the probe record, and flip `has_end_to_end_example: true` in the frontmatter.

If it doesn't exist: create it now using `templates/frontmatter/api-raw.yaml`, with the signature extracted from upstream (Step 1) and a link to the probe record under `probed_by:`.

## Failure modes (MVP handling)

| Condition | What to do |
| --- | --- |
| Zero upstream hits | `kind: undocumented`, `status: blocked`, stop, ask human. |
| Compile fails | `status: failed`, keep the probe record with the error message, stop. |
| Correctness fails (max_abs_err exceeds tolerance) | `status: failed`, record the observed error, stop. |
| Latency not interesting (single call < 0.01 ms) | Still record; the probe is for *semantics*, not performance. Set `evidence_level: measured` anyway. |
| Probe passes but contradicts upstream doc | **Out of MVP scope** (discrepancy handling). Record the observation in `## Notes`, `status: verified`, and file an issue. |

## Out of scope (skeleton version)

- Full discrepancy handling and automatic `cuda-official` annotation.
- Multi-kernel probes (cluster-wide, multi-CTA coordination).
- NVML-backed `clock_policy` verification (until M5).
- Cross-run deduplication via a provenance index.
