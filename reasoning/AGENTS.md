# KB-Gen Agent Contract

> This directory is the **knowledge base being built**. Downstream
> kernel-writing agents will later consume it. All content here is produced
> by a **KB-generation agent** that obeys this file.
>
> **Scope (updated 2026-04-28 by `docs/15-代码仓库抽取知识流程_update_en.md`)**:
> CUDA-core single-kernel optimizations for H200 (sm_90a) **plus** tensor-core
> primitives (TMA, wgmma) and fused cuda-core/tensor-core gemm flows extracted
> from the cutlass repository. Multi-kernel system features remain layered
> separately under `90-system-level/`. The previously deferred layers
> `40-hardware-feature/`, `50-classical-algo/`, and `60-code/` (formerly
> `60-cutlass-cute/`) are now in scope; `60-code/` uses a source-driven
> second-level structure (`60-code/<source-repo>/`, e.g. `60-code/cutlass-cute/`).

---

## Step 0 — Read these meta-skills before writing anything

An agent starting any task under this directory MUST first read, in order:

1. `70-reasoning/task-packet.md` — the input contract for a build task.
2. `70-reasoning/api-probing.md` — the F10 protocol for probing an API you do not yet understand.
3. `70-reasoning/hardware-microbench.md` — the protocol for **active microbenchmark probes** (compute-instruction latency, memory pointer chasing, whitepaper-driven feature sweeps). Use this whenever a task needs cycle-level or bandwidth facts that the passive introspection APIs cannot give.
4. `70-reasoning/benchmark-protocol.md` — the N2 protocol for reproducible measurement.
5. `70-reasoning/bottleneck-triage.md` — the decision tree for picking the next skill when a benchmark falls short.
6. `templates/frontmatter/*.yaml` — required frontmatter schemas per layer.
7. `05-source-corpus/RETRIEVAL.md` — the retrieval contract for upstream materials.
8. `GLOBAL_VARIABLES.md` — placeholder variables for local paths or hosted mirrors.

If any of these files is missing, stop and report — do not improvise.

**How to decide between `api-probing.md` and `hardware-microbench.md`**:
- If the task asks *"what does this API do / how do I call it?"* → `api-probing.md`.
- If the task asks *"how fast is this instruction / how many cycles does this memory level cost?"* → `hardware-microbench.md`.
- If the task asks for a hardware-new-feature characterization driven by an architecture whitepaper (TMA, DSMEM cluster size sweep, wgmma throughput, setmaxnreg effect) → `hardware-microbench.md` §Feature sweep. **MVP status (updated 2026-04-28)**: active when the task is part of the cutlass-extraction roadmap (see `docs/15-代码仓库抽取知识流程_update_en.md`). Hardware-feature skills land under `40-hardware-feature/<feature>/`; their measurement records go to `80-experience/hw-probes/<feature>/`.

---

## Layer responsibilities (MVP subset)

| Layer | Purpose | Entry format | Who writes |
| --- | --- | --- | --- |
| `00-foundation/hardware-spec/` | H200 static spec + runtime introspection bundles | `.md` + `.json` | `tools/kp_introspect.py` (M2/M3) + agent to narrate |
| `05-source-corpus/` | Manifest-backed source registry and retrieval contract for upstream docs / blogs / source repos | `MANIFEST.yaml` + `RETRIEVAL.md` + generated indexes | Humans + retrieval tools; **agent reads only** |
| `10-api-raw/<ns>/<func>.md` | Authoritative signature + minimal usage per API | Strict schema: see `templates/frontmatter/api-raw.yaml` | Agent (during skill build or standalone API probe) |
| `20-pattern/cuda-core/<op>/` | Operator-entry four-pack: `INDEX.md` / `ROUTING.md` / `library-fallback.md` / `TASK-PACKET.md` | One directory per operator | Agent (M6) |
| `30-skill/<family>/<skill>/` | Single-kernel optimization technique. MVP: `skill.md` (required) + `pitfalls.md` (required) + `apis.md` (optional, when skill touches APIs). `verified.md` is dropped in MVP. | Schema: `templates/frontmatter/skill.yaml` | Agent (M4c, M5) |
| `40-hardware-feature/<feature>/` | Hardware-new-feature characterization (TMA, wgmma, DSMEM, async-pipeline, setmaxnreg, …). Same four-file convention as `30-skill/`: `skill.md` (required) + `pitfalls.md` (required) + `apis.md` (optional). `evidence_level: measured` requires a `## Measured Characteristics` section pointing at `80-experience/hw-probes/<feature>/...`. | Schema: `templates/frontmatter/hardware-feature.yaml` (extends `skill.yaml` with `requires_sm` ≥ 9.0) | Agent (driven by `docs/15-代码仓库抽取知识流程_update_en.md`) |
| `50-classical-algo/<algo>/` | End-to-end classical optimization algorithms extracted from upstream code (warp-specialization, persistent-kernel, online-softmax, split-k, …). Same four-file convention as `30-skill/`. Algorithm skeleton code lives separately under `60-code/<source-repo>/<algo>/`. | Schema: `templates/frontmatter/classical-algo.yaml` (extends `skill.yaml`) | Agent (driven by `docs/15-代码仓库抽取知识流程_update_en.md`) |
| `60-code/<source-repo>/<topic>/` | Code-repository extraction layer: **library-usage knowledge** for upstream operator libraries (cutlass / cute on Hopper; vllm; …). Per-topic directories carry `README.md` (when-to-use + knob catalogue), `tuning.md` (template-parameter search log + configuration strategy), and optional `<feature>_skeleton.md` (line-numbered tour of the upstream code). Buildable `.cu` / `build.sh` files do **not** live here — the canonical reproducible binaries are under `80-experience/<api-probes|hw-probes>/<topic>/artifacts/`. Exception: `60-code/ptx-gemm/` keeps its `.cu` because the cutlass-free track has no upstream example to point at. Lint exempts the whole subtree from frontmatter requirements. | — | Agent (driven by `docs/15-代码仓库抽取知识流程_update_en.md`) |
| `70-reasoning/` | Meta-skills (handwritten, frozen) | — | Humans only; **agent is forbidden to write here** |
| `80-experience/api-probes/` | F10 API-semantics probe products | Schema: `templates/frontmatter/experience.yaml` | Agent (during API probe) |
| `80-experience/hw-probes/` | Hardware microbench probe products (compute-latency / memory-latency); `skill.md` links here via `## Measured Characteristics` | Schema: `templates/frontmatter/experience.yaml` | Agent (during microbench, see `70-reasoning/hardware-microbench.md`) |
| `90-system-level/<sub-area>/` | Kernel-boundary and multi-kernel concerns (launch overhead, CUDA graphs, stream concurrency, host-device transfer, …) — operator-invariant, not referenced from `20-pattern/` ROUTING. Loaded on demand from `70-reasoning/bottleneck-triage.md` when the symptom matches. See `90-system-level/AGENTS.md` for the layer contract and roadmap. | `skill.md` + `apis.md` + `pitfalls.md` (same four-file convention as `30-skill/`) | Agent |
| `templates/frontmatter/` | YAML schemas for every layer's frontmatter | — | Humans only |

Layer `07-wiki/` remains **out of scope** in MVP; do not create it. `40-hardware-feature/`, `50-classical-algo/`, and `60-code/` were deferred post-MVP at first but became **active** on 2026-04-28 under the cutlass-extraction plan (`docs/15-代码仓库抽取知识流程_update_en.md`); content there must follow the same evidence/source/artifact rules as `30-skill/`. `90-system-level/` is built incrementally — only `launch-overhead/` exists today; other sub-areas stay pending until a task needs them.

---

## Hard constraints (all enforced by `tools/lint_knowledge.py`)

1. **No imagination**. Every factual claim in any generated `.md` must be grounded in one of:
   - (a) a grepped upstream document or source code path (record in `source:`),
   - (b) a runnable code artifact the agent itself built and ran (record in `artifacts.code`/`artifacts.build`),
   - (c) a `kp_introspect` bundle (record in `artifacts.introspection`).

   Any factual claim without grounding → either downgrade `evidence_level` to `inferred` or delete the claim.

2. **Frontmatter is mandatory**. Every generated `.md` starts with a YAML block that satisfies the relevant template under `templates/frontmatter/`. The agent runs `lint_knowledge.py` against its own output before declaring the task done.

3. **Language is English only**. Agent-facing consistency trumps translation comfort. File names, directory names, frontmatter field values, body prose, and code comments are all English. (User-facing READMEs in Chinese are fine — those live outside `knowledge/`.)

4. **Measured means bound**. Any entry with `evidence_level: measured` MUST populate:
   - `artifacts.code` — relative path to the probe or microbenchmark `.cu`
   - `artifacts.build` — the exact `nvcc` command or a `build.sh` path
   - `artifacts.introspection` — a `kp_introspect bundle --output <path>` json
   - `measured_on` — device name + sm + cuda runtime + driver
   Missing any of these → `lint_knowledge.py` rejects the file.

5. **Source provenance**. `source:` lists at least one `{path, anchor}` entry. `path:` is absolute (for upstream `cuda_document/` / `cuda_repo/`) or repo-relative (for `knowledge/` internal links). `anchor:` is either a heading slug or a `Lxxx-Lyyy` line range. An optional `excerpt:` short quote is welcome. When referencing original upstream material, prefer `05-source-corpus/...` paths produced by the retrieval module instead of raw absolute paths.

6. **Never touch `70-reasoning/`**. Meta-skills are frozen. If you find a gap, report it in the final done-report; do not rewrite them.

7. **Target hardware is fixed**. MVP assumes H200 = sm_90a exclusively. Do not add multi-target variants or SM compatibility flags.

8. **Skill contract** (MVP-simplified). When building `30-skill/<family>/<name>/`, the mandatory deliverables are:
   - **`skill.md`** (REQUIRED) — narrative: what the technique is, why it helps, when to use it, when not to, plus a `## Measured Characteristics` section that links to the hw-probe records under `80-experience/hw-probes/<probe-slug>/...md` produced during this task.
   - **`pitfalls.md`** (REQUIRED) — known failure modes and how to detect them.
   - **`apis.md`** (OPTIONAL) — machine-readable list of APIs touched by this skill, each linked to `10-api-raw/`. **Only create this file when the skill actually touches named APIs**. Pure programming-pattern skills (e.g., loop unrolling, tiling) skip it.
   - **`verified.md`** — **DROPPED in MVP**. Its post-MVP semantics ("marks whether this skill is used by a real kernel") are not needed yet; do not create this file.

   **Where measured data lives** (MVP): microbench results from the agent's own probe runs go into `80-experience/hw-probes/<instruction-or-level-slug>/<YYYY-MM-DD>-<task-slug>.md` with the `experience.yaml` frontmatter. `skill.md` then points at those files via `## Measured Characteristics` bullets. **Do not embed a benchmark table inline in `skill.md`** — the probe record is the single source of truth.

9. **Pattern four-pack contract**. When building `20-pattern/cuda-core/<op>/`, you produce exactly:
   - `INDEX.md` — decision tree: library-first → custom.
   - `ROUTING.md` — skill whitelist for this operator (only references skills already present under `30-skill/`).
   - `library-fallback.md` — documented torch / cub / cuBLAS paths with shape ranges.
   - `TASK-PACKET.md` — the operator-specific task packet (refines `70-reasoning/task-packet.md`).

---

## Task lifecycle

```
tasks/<task_id>.yaml
        │
        ▼
Agent loads task, reads 70-reasoning/* meta-skills
        │
        ▼
Agent searches 05-source-corpus via `source_search` / `source_read` for relevant material
        │
        ▼
(optional) Agent invokes kp_introspect for hardware facts
        │
        ▼
(optional) Agent writes a minimal probe .cu, runs nvcc, measures latency per benchmark-protocol.md
        │
        ▼
Agent writes output files under target_path
        │
        ▼
Agent runs tools/lint_knowledge.py on output; fixes errors
        │
        ▼
Agent emits a done-report:
  - touched files
  - measured results (if any)
  - unresolved gaps (for human follow-up)
```

---

## Forbidden behaviors

- Writing into `70-reasoning/`.
- Writing files outside `target_path` (unless the task explicitly says `also_writes:`).
- Reading files outside `upstream_scope` + `knowledge/` + `tools/`.
- Grepping absolute upstream paths (`{{CUDA_REPO_ROOT}}`, repo-specific `*_REPO_REF`, or `{{CLAUDE_RESEARCH_REF}}`) directly. Use relative paths under `05-source-corpus/` via the retrieval module instead.
- Using a baseline that is not recorded in `verified.md`.
- Declaring `status: verified` without `evidence_level: measured` + full `artifacts` block.
- Running `nvcc` with `-arch` other than `sm_90a` (MVP is H200-only).
- Introducing any file outside English (see constraint 3).

---

## When you are stuck

If a task cannot converge (for example, an API under probe is undocumented everywhere in `upstream_scope`), the agent stops, writes a stub with `status: blocked` and a concrete question, and returns a done-report listing the blockers. **Do not guess.**
