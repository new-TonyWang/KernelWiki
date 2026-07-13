"""KB-gen agent — OpenAI API path (M4a).

Usage:
    python -m agent.openai_path.main tasks/build-warp-primitives.yaml

Connects to the LLM via OpenAI-compatible API at KP_OPENAI_BASE_URL
(default http://localhost:4000). Runs a tool-call loop until the agent
says "done" or MAX_AGENT_TURNS is reached.

Environment variables:
    KP_OPENAI_BASE_URL   LiteLLM / vLLM endpoint (default: http://localhost:4000)
    KP_OPENAI_API_KEY    API key (default: sk-dummy)
    KP_OPENAI_MODEL      Model name (default: gpt-4o)
    KP_MAX_AGENT_TURNS   Max tool-call rounds (default: 50)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from agent.shared.config import MAX_AGENT_TURNS, OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
from agent.shared.system_prompt import build_system_prompt
from agent.shared.task_schema import load_task, task_to_yaml_str
from agent.shared.tools import TOOL_REGISTRY
from agent.openai_path.tool_defs import TOOL_DEFINITIONS


def create_client():
    try:
        from openai import OpenAI
    except ImportError:
        print(
            "ERROR: openai package not installed. Install with:\n"
            "    pip install openai\n",
            file=sys.stderr,
        )
        sys.exit(1)
    return OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)


def execute_tool_call(name: str, arguments: dict) -> str:
    """Dispatch a tool call to the shared implementation."""
    func = TOOL_REGISTRY.get(name)
    if func is None:
        return f"(error: unknown tool '{name}')"
    try:
        return func(**arguments)
    except Exception as e:
        return f"(error executing {name}: {e})"


def run_agent(task_path: Path) -> None:
    task = load_task(task_path)
    system_prompt = build_system_prompt()
    task_yaml = task_to_yaml_str(task)

    client = create_client()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Execute this task:\n\n```yaml\n{task_yaml}\n```"},
    ]

    print(f"[agent] model={OPENAI_MODEL} base_url={OPENAI_BASE_URL}", file=sys.stderr)
    print(f"[agent] task={task['task_id']} type={task['task_type']}", file=sys.stderr)
    print(f"[agent] target={task['target_path']}", file=sys.stderr)
    print(f"[agent] max_turns={MAX_AGENT_TURNS}", file=sys.stderr)

    for turn in range(1, MAX_AGENT_TURNS + 1):
        print(f"\n--- turn {turn}/{MAX_AGENT_TURNS} ---", file=sys.stderr)

        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                temperature=0.2,
            )
        except Exception as e:
            print(f"[agent] API error: {e}", file=sys.stderr)
            sys.exit(2)

        choice = response.choices[0]
        msg = choice.message

        if choice.finish_reason == "stop" or (msg.content and not msg.tool_calls):
            print(f"\n[agent] finished at turn {turn}", file=sys.stderr)
            if msg.content:
                print("\n=== AGENT FINAL REPORT ===\n")
                print(msg.content)
            break

        if not msg.tool_calls:
            print(f"[agent] no tool_calls and no content at turn {turn}, stopping", file=sys.stderr)
            break

        messages.append(msg)

        for tc in msg.tool_calls:
            fname = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            print(f"  tool: {fname}({_summarize_args(args)})", file=sys.stderr)
            result = execute_tool_call(fname, args)

            result_preview = result[:200] + "..." if len(result) > 200 else result
            print(f"    → {result_preview}", file=sys.stderr)

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })
    else:
        print(f"\n[agent] hit max turns ({MAX_AGENT_TURNS}), stopping", file=sys.stderr)


def _summarize_args(args: dict) -> str:
    """Short one-line summary of tool arguments for logging."""
    parts = []
    for k, v in args.items():
        s = str(v)
        if len(s) > 60:
            s = s[:57] + "..."
        parts.append(f"{k}={s}")
    return ", ".join(parts)


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m agent.openai_path.main <task.yaml>", file=sys.stderr)
        sys.exit(1)
    task_path = Path(sys.argv[1])
    if not task_path.exists():
        print(f"Task file not found: {task_path}", file=sys.stderr)
        sys.exit(1)
    run_agent(task_path)


if __name__ == "__main__":
    main()
