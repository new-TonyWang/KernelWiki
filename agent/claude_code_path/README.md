# KB-gen Agent — Claude Code Path (M4b)

Uses Claude Code's built-in Agent tool with a generated prompt. No
additional Python dependencies beyond what M1/M2 already installed.

## Usage

### Option 1: Generate prompt → paste into Claude Code

```bash
cd /path/to/kernel-kb-mvp
python -m agent.claude_code_path.subagent_prompt tasks/build-warp-primitives.yaml > /tmp/prompt.md
```

Then in Claude Code, use the Agent tool with the content of `/tmp/prompt.md`
as the `prompt` parameter.

### Option 2: Inline in a Claude Code conversation

In a Claude Code session, ask:

```
Read the file at /path/to/kernel-kb-mvp/tasks/build-warp-primitives.yaml,
then act as a KB-gen agent following the prompt in
agent/claude_code_path/subagent_prompt.py to build the warp-primitives skill.
```

Claude Code will use its native Read/Write/Edit/Bash/Grep/Glob tools to
execute the task.

## How it differs from the OpenAI path

| Aspect | OpenAI path | Claude Code path |
|---|---|---|
| LLM | Any OpenAI-compatible model (via localhost:4000) | Claude (via Claude Code CLI) |
| Tools | Custom function-call tools (grep_source, write_file, ...) | Claude Code built-in (Read, Write, Bash, Grep, ...) |
| Loop control | Python loop in main.py | Claude Code's internal agent loop |
| GPU access | run_on_gpu tool (SSH wrapper) | Bash tool + SSH command |
| Dependencies | openai + pyyaml + click | pyyaml + click (for lint/probe_gap_scan) |

Both paths share the same:
- Task YAML schema (tasks/*.yaml)
- System prompt content (agent/shared/system_prompt.py)
- Knowledge base contract (knowledge/AGENTS.md)
- Meta-skills (knowledge/70-reasoning/*)
- Frontmatter templates (knowledge/templates/frontmatter/*)
- Output structure (knowledge/30-skill/*, knowledge/80-experience/*)
