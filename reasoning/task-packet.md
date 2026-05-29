# Task Packet Contract (N3)

The KB-gen agent takes exactly one input: a YAML file under `tasks/`. This file is the **task packet**. If any required field is missing or malformed, the agent refuses to start.

## Required fields

```yaml
task_id: 2026-04-15-warp-primitives         # unique, date-prefixed, kebab-case
task_type: build-skill                      # see enum below
target_path: wiki/nvidia/foundations/compute/warp-primitives/   # relative to , ends in / for dir or .md for file
hardware:
  device: H200
  sm: "9.0a"
  cuda_runtime: "12.4"
upstream_scope:                              # absolute paths OR -relative paths under corpus/nvidia/; agent forbidden to grep outside
  - corpus/nvidia/cuda-official/cuda-toolkit-documentation-13.2/   # -relative entry under 05-source-corpus is allowed
  - {{CCCL_REPO_REF}}/cub/                        # absolute path is also allowed
references:                                  # meta-skills the agent must read first
  - reasoning/api-probing.md
  - reasoning/hardware-microbench.md
  - reasoning/benchmark-protocol.md
  - reasoning/bottleneck-triage.md
success_criteria:                            # self-checked before done-report (MVP rules)
  - file_exists: skill.md                    # required
  - file_exists: pitfalls.md                 # required
  - lint_clean: true
  - at_least_one_hw_probe_linked: true       # skill.md's ## Measured Characteristics must link ≥1 record under sources/experience/hw-probes/
  # apis.md is OPTIONAL — only required when the skill touches named APIs.
  #   If this task expects apis.md, add an explicit line:
  #     - file_exists: apis.md
  # verified.md is DROPPED in MVP — do not include it.
```

## `task_type` enum

| Value | Meaning | Typical output |
| --- | --- | --- |
| `build-skill` | Produce a skill four-pack under `wiki/nvidia/foundations/<family>/<skill>/` | `skill.md` (REQUIRED) + `pitfalls.md` (REQUIRED) + `apis.md` (OPTIONAL — when the skill touches named APIs). `verified.md` is dropped in MVP; do not create it. |
| `probe-api` | Run an F10 probe on one API and back-fill `wiki/nvidia/api-definitions/` | `sources/experience/api-probes/...` + updated `wiki/nvidia/api-definitions/...` |
| `build-pattern` | Produce a pattern four-pack under `wiki/nvidia/operator-routing/cuda-core/<op>/` | `INDEX.md` + `ROUTING.md` + `library-fallback.md` + `TASK-PACKET.md` |
| `introspect-hardware` | Run `kp_introspect` and populate `00-foundation/hardware-spec/` | updated `h200-specs.md` + `runtime-introspection/bundle-*.json` |
| `build-hardware-feature` | Produce a hardware-feature four-pack under `wiki/nvidia/hardware/<feature>/` (TMA, wgmma, DSMEM, cluster-launch-control, …). `target_path` is the feature directory; companion outputs (code skeleton, hw-probe record) live outside that directory and MUST be listed under `also_writes:`. | `skill.md` + `pitfalls.md` (under `target_path`); `*.cu` + `build.sh` under `wiki/nvidia/code-walkthroughs/<source-repo>/<feature>/`; `<YYYY-MM-DD>-<slug>.md` under `sources/experience/hw-probes/<feature>/` |
| `extend-hardware-feature` | Append measured material to an existing `wiki/nvidia/hardware/<feature>/skill.md` (typically a shape sweep or follow-up probe) without changing the skill's identity. `target_path` MUST be the existing feature directory; new probe records under `sources/experience/hw-probes/<feature>/` are listed in `also_writes:`. | Append `## Shape-selection guide` (or analogous section) to the existing `skill.md`; new `<YYYY-MM-DD>-<slug>.md` under `sources/experience/hw-probes/<feature>/` |
| `build-classical-algo` | Produce a classical-algorithm four-pack under `wiki/nvidia/techniques/<algo>/` (warp-specialization, persistent-kernel, online-softmax, split-k, …). The algorithm code skeleton lives separately under `wiki/nvidia/code-walkthroughs/<source-repo>/<algo>/`; the A/B ablation record lives under `sources/experience/api-probes/<op>/`. | `skill.md` + `pitfalls.md` (under `target_path`); `*.cu` + `build.sh` under `wiki/nvidia/code-walkthroughs/<source-repo>/<algo>/`; `<YYYY-MM-DD>-<slug>-ablation.md` under `sources/experience/api-probes/<op>/` |

### Rules for the cutlass-extraction task kinds (`build-hardware-feature`, `extend-hardware-feature`, `build-classical-algo`)

These three kinds were added on 2026-04-28 to support `docs/15-代码仓库抽取知识流程_update_en.md`. All three obey the same guard rails:

1. **`target_path` is deterministic** — exactly one directory, no `<slug>` placeholders, no `OR` alternatives. Resolve any landing-path choice to one path *in the task packet*, not at execution time.
2. **All non-`target_path` outputs are listed under `also_writes:`**. The packet enumerates the exact directories the agent is allowed to touch (e.g. `wiki/nvidia/code-walkthroughs/<source-repo>/<topic>/`, `sources/experience/hw-probes/<feature>/`, `sources/experience/api-probes/<op>/`).
3. **`upstream_repo_pin:` is required** — pin the source-repo commit (e.g. `cutlass@f74fea9c`) so the extracted skeleton is reproducible.
4. **`source_anchors:` is required** — enumerate concrete files in the source repo (e.g. `examples/48_hopper_warp_specialized_gemm/48_hopper_warp_specialized_gemm.cu`, `include/cute/atom/copy_traits_sm90_tma.hpp`) so the agent does not browse blindly.
5. **`success_criteria:` paths are concrete** — every `file_exists:` entry resolves to a literal path (no `<slug>`, `<YYYY-MM-DD>`, or `*`). Date and slug placeholders are forbidden because they prevent programmatic checking.
6. **Stage 3.5 (AC-V) verdict is mandatory** before the task is considered complete; the packet's success-criteria block records the metrics that the experiment record must contain.

If a task needs to land in one of two possible directories (e.g. AC-7 of the cutlass plan: `wiki/nvidia/foundations/compute/gemm-fused/` vs. `wiki/nvidia/operator-routing/fused-cuda-tensor-core/`), the task packet must commit to one before execution; the rationale goes in the per-task report.

## Optional fields

```yaml
related_skills: [coalescing, vectorized-access]     # cross-reference hints for ROUTING.md
prior_art:                                           # existing entries to extend or supersede
  - path: wiki/nvidia/foundations/compute/warp-primitives/skill.md
    relation: supersedes | extends | references
deadline: 2026-04-20                                 # informational, soft
notes: |                                             # free-form guidance from the human issuing the task
  Focus on warp shuffle + vote families. Skip match.* for now — MVP out of scope.
```

## Field semantics

- **`target_path`** — directory (trailing `/`) or file (ending `.md`). Agent refuses to write outside this path unless `also_writes:` is explicitly set.
- **`upstream_scope`** — list of absolute paths or ``-relative paths under `corpus/nvidia/`. Agent uses these for `grep_source` / `read_file` on upstream material. Grepping outside this list is forbidden; reading `` and `tools/` is always allowed (for reading meta-skills and invoking the CLI tools).
- **`references`** — meta-skill files the agent must read *before* generating any output. Agent confirms it has read them by listing their paths in the done-report.
- **`success_criteria`** — checked programmatically by `lint_knowledge.py` + a self-review pass. Any failure → the agent reports what is missing and stops (no partial merge).

## Example — `tasks/build-warp-primitives.yaml`

```yaml
task_id: 2026-04-15-warp-primitives
task_type: build-skill
target_path: wiki/nvidia/foundations/compute/warp-primitives/
hardware:
  device: H200
  sm: "9.0a"
  cuda_runtime: "12.4"
upstream_scope:
  - corpus/nvidia/cuda-official/cuda-toolkit-documentation-13.2/programming-guide/
  - corpus/nvidia/cuda-official/cuda-toolkit-documentation-13.2/ptx-isa/
  - {{CCCL_REPO_REF}}/cub/
  - {{CUDA_SAMPLES_REPO_REF}}/
references:
  - reasoning/api-probing.md
  - reasoning/hardware-microbench.md
  - reasoning/benchmark-protocol.md
  - reasoning/bottleneck-triage.md
related_skills: [coalescing, vectorized-access]
success_criteria:
  - file_exists: skill.md
  - file_exists: pitfalls.md
  - file_exists: apis.md               # REQUIRED here because warp-primitives touches __shfl_sync / __ballot_sync etc.
  - lint_clean: true
  - at_least_one_hw_probe_linked: true # skill.md ## Measured Characteristics links ≥1 hw-probe
notes: |
  Cover shfl.sync (idx/up/down/bfly), vote.sync (ballot/all/any),
  and one classical warp-reduce example (block-level sum of 1024 floats).
  Skip match.sync and reduce.sync — deferred.
  Agent writes its own compute-latency probe per hardware-microbench.md
  §Compute latency protocol; canonical baseline from `kp_introspect microbench
  compute-latency` may be cross-referenced but the per-skill probe is the
  authoritative record for the skill-specific pattern.
```

## What happens on agent start

1. Load `tasks/<task_id>.yaml`, validate against this schema. Refuse on error.
2. Resolve `target_path` → absolute path under ``. Refuse if the path escapes ``.
3. Read every file in `references:` sequentially. These become the agent's operational rules.
4. Begin the build according to `task_type`.
5. On finish, run `tools/lint_knowledge.py` over all written files. Fix any errors.
6. Emit a done-report: touched files, measured results, unresolved gaps.
