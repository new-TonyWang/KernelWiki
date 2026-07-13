"""Generate the Claude Code subagent prompt for a given task.

Usage:
    python -m agent.claude_code_path.subagent_prompt tasks/build-warp-primitives.yaml

Prints the full prompt to stdout — paste it into Claude Code's Agent tool,
or pipe it to `claude --print`.
"""

from __future__ import annotations

import sys
from pathlib import Path

from agent.shared.config import (
    KNOWLEDGE_ROOT,
    PROJECT_ROOT,
    REMOTE_CUDA_PATH,
    REMOTE_DIR,
    REMOTE_HOST,
    REMOTE_PYLIB,
)
from agent.shared.task_schema import load_task, task_to_yaml_str


PROMPT_TEMPLATE = """\
You are a **KB-gen agent** building CUDA optimization knowledge for H200 (sm_90a).

## Project paths

- Project root: {project_root}
- Knowledge base: {knowledge_root}/
- Source corpus: `{knowledge_root}/corpus/nvidia/`
- Remote GPU: `ssh {remote_host}` → working dir `{remote_dir}`
- CUDA: `export PATH={remote_cuda_path}:$PATH`
- Pylib: `export PYTHONPATH={remote_pylib}`

## Your task

```yaml
{task_yaml}
```

## Mandatory first steps

1. Use the Read tool to read `{knowledge_root}/reasoning/AGENTS.md` — your 9 hard constraints.
2. Read relevant meta-skills under `{knowledge_root}/reasoning/`:
   - For **build-skill**: api-probing.md, hardware-microbench.md, benchmark-protocol.md, bottleneck-triage.md
   - For **probe-api**: api-probing.md, benchmark-protocol.md
   - For **build-pattern**: task-packet.md, bottleneck-triage.md
3. Read the frontmatter template: `{knowledge_root}/templates/frontmatter/{{skill|api-raw|experience}}.yaml`
4. For upstream material, use `python3 -m scripts.source_corpus.cli ...` instead of grepping external absolute paths directly.

## Key rules

- **No imagination**: every factual claim must be grounded in grep output or measured data.
- **Source corpus first**: use `python3 -m scripts.source_corpus.cli search/read/list` for original docs, blogs, and source repos.
- **Frontmatter mandatory**: every .md starts with YAML per the template.
- **English only** in .
- **Measured data** → write probe record to `{knowledge_root}/sources/experience/hw-probes/<slug>.md`, code artifacts to `{knowledge_root}/artifacts/experience/hw-probes/<slug>/`
  using experience.yaml. Then add a `## Measured Characteristics` section in skill.md linking it.
- **Skill output**: skill.md + pitfalls.md required. apis.md optional (only if skill uses named APIs).
  verified.md is DROPPED.
- **Lint**: run `python3 scripts/validate.py --root {knowledge_root}` before reporting done.

## GPU operations

Use the Bash tool with SSH for any CUDA work:

```bash
ssh {remote_host} "cd {remote_dir} && export PATH={remote_cuda_path}:\\$PATH && export PYTHONPATH={remote_pylib} && <command>"
```

Before SSH commands, sync your local changes:
```bash
rsync -avz --exclude '.git' --exclude '__pycache__' {project_root}/ {remote_host}:{remote_dir}/
```

After SSH commands that produce output files, sync back:
```bash
rsync -avz {remote_host}:{remote_dir}/ {knowledge_root}/
```

## Workflow for build-skill

1. **Search upstream** via:

```bash
python3 -m scripts.source_corpus.cli search "<topic or api>" --scope cuda-official
```

2. **Read** the most relevant hits, for example:

```bash
python3 -m scripts.source_corpus.cli read "cuda-official/toolkit-docs-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md" --anchor "L120-L140"
```

3. **Write skill.md** (frontmatter + narrative: what / why / when / when-not).
4. **Write pitfalls.md** (failure modes + detection).
5. **Write apis.md** if the skill touches named APIs.
6. **Write a microbench probe .cu** following hardware-microbench.md protocol:
   - Put .cu under `{knowledge_root}/artifacts/experience/hw-probes/<insn-slug>/`
   - Sync to GPU → compile with nvcc → run → collect results
   - Sync back → write probe record .md at `{knowledge_root}/sources/experience/hw-probes/<insn-slug>.md`
7. **Add ## Measured Characteristics** to skill.md linking the probe.
8. **Run lint** and fix errors.
9. **Report**: list files created, measurements, issues.

## When stuck

Record `status: blocked` with the error message. Do NOT fabricate data.
"""


def build_prompt(task_path: Path) -> str:
    task = load_task(task_path)
    return PROMPT_TEMPLATE.format(
        project_root=PROJECT_ROOT,
        knowledge_root=KNOWLEDGE_ROOT,
        remote_host=REMOTE_HOST,
        remote_dir=REMOTE_DIR,
        remote_cuda_path=REMOTE_CUDA_PATH,
        remote_pylib=REMOTE_PYLIB,
        task_yaml=task_to_yaml_str(task),
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m agent.claude_code_path.subagent_prompt <task.yaml>", file=sys.stderr)
        sys.exit(1)
    print(build_prompt(Path(sys.argv[1])))
