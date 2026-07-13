# KB-gen Agent — OpenAI API Path (M4a)

Runs the KB-gen agent loop against an OpenAI-compatible API (e.g., LiteLLM
proxy at localhost:4000).

## Prerequisites

```bash
pip install openai pyyaml click
```

## Usage

```bash
# Set the LLM endpoint (default: http://localhost:4000/v1)
export KP_OPENAI_BASE_URL="http://localhost:4000/v1"
export KP_OPENAI_API_KEY="sk-dummy"
export KP_OPENAI_MODEL="gpt-4o"     # or claude-sonnet-4-20250514, etc.

# Run the agent on a task
cd /path/to/kernel-kb-mvp
python -m agent.openai_path.main tasks/build-warp-primitives.yaml
```

## What happens

1. Loads the task YAML and system prompt.
2. Sends them to the LLM via the OpenAI chat completions API.
3. The LLM responds with tool calls (grep_source, read_file, write_file,
   run_bash, run_on_gpu, sync_to_gpu, sync_from_gpu).
4. Each tool call is executed locally (or via SSH for GPU ops).
5. Results are fed back to the LLM.
6. Loop continues until the LLM produces a final text response or
   KP_MAX_AGENT_TURNS (default 50) is reached.
7. The agent's final report is printed to stdout.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| KP_OPENAI_BASE_URL | http://localhost:4000/v1 | LLM API endpoint |
| KP_OPENAI_API_KEY | sk-dummy | API key |
| KP_OPENAI_MODEL | gpt-4o | Model name |
| KP_MAX_AGENT_TURNS | 50 | Max tool-call rounds |
| KP_REMOTE_HOST | h200_ncu | SSH hostname for GPU ops |
| KP_REMOTE_DIR | /inspire/hdd/.../kernel-kb-mvp | Remote working directory |
| KP_PROJECT_ROOT | (auto-detected) | Local project root |
