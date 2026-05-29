"""Tool implementations shared by both agent paths.

Each function takes simple typed arguments and returns a string result.
The OpenAI path wraps these as function-call handlers; the Claude Code path
documents them in the subagent prompt (Claude Code has its own built-in tools,
but we keep this module as a reference implementation + for the OpenAI path).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from agent.shared.config import (
    KNOWLEDGE_ROOT,
    PROJECT_ROOT,
    REMOTE_CUDA_PATH,
    REMOTE_DIR,
    REMOTE_HOST,
    REMOTE_PYLIB,
)
try:
    from scripts.source_corpus import (
        build_provenance_index,
        list_sources,
        provenance_back,
        provenance_walk,
        resolve_source,
        source_read,
        source_search,
    )
except ImportError:
    # Graceful degradation if source_corpus package not on sys.path
    def _stub(*a, **kw):
        return "source_corpus package not available"
    build_provenance_index = list_sources = provenance_back = _stub
    provenance_walk = resolve_source = source_read = source_search = _stub


def grep_source(pattern: str, scope: str, glob: str | None = None,
                max_results: int = 50) -> str:
    """Ripgrep a pattern under a scope directory. Returns matching lines."""
    cmd = ["rg", "--no-heading", "--line-number", "-m", str(max_results), pattern, scope]
    if glob:
        cmd.extend(["--glob", glob])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        return "(error: rg (ripgrep) not found on PATH)"
    except subprocess.TimeoutExpired:
        return "(error: grep timed out after 30s)"
    if result.returncode == 1:
        return "(no matches)"
    if result.returncode != 0:
        return f"(error: rg exit {result.returncode}: {result.stderr[:300]})"
    output = result.stdout
    if len(output) > 10000:
        output = output[:10000] + "\n... (truncated)"
    return output


def source_search_tool(
    query: str,
    scope: str | None = None,
    source_types: list[str] | None = None,
    entity_kind: str | None = None,
    top_k: int = 10,
    regex: bool = False,
) -> str:
    """Search the manifest-backed source corpus and return structured JSON."""
    result = source_search(
        query=query,
        scope=scope,
        source_types=source_types,
        entity_kind=entity_kind,
        top_k=top_k,
        regex=regex,
    )
    return json.dumps(result, ensure_ascii=False)


def source_read_tool(
    path: str,
    anchor: str | None = None,
    section: str | None = None,
    max_chars: int = 6000,
    focused_query: str | None = None,
) -> str:
    """Read a source corpus file by anchor or section and return structured JSON."""
    result = source_read(
        path=path,
        anchor=anchor,
        section=section,
        max_chars=max_chars,
        focused_query=focused_query,
    )
    return json.dumps(result, ensure_ascii=False)


def source_list_tool(
    scope: str | None = None,
    source_type: str | None = None,
    keyword: str | None = None,
    year: str | None = None,
    limit: int = 50,
) -> str:
    """List available source corpus entries from MANIFEST.yaml."""
    result = list_sources(
        scope=scope,
        source_type=source_type,
        keyword=keyword,
        year=year,
        limit=limit,
    )
    return json.dumps(result, ensure_ascii=False)


def source_resolve_tool(ref: str) -> str:
    """Resolve a source id or corpus path to manifest metadata / actual path."""
    return json.dumps(resolve_source(ref), ensure_ascii=False)


def provenance_walk_tool(knowledge_path: str) -> str:
    """Return the frontmatter source refs for a knowledge entry as JSON."""
    return json.dumps(provenance_walk(knowledge_path), ensure_ascii=False)


def provenance_back_tool(path: str, anchor: str | None = None) -> str:
    """Return the reverse provenance references for a source path as JSON."""
    return json.dumps(provenance_back(path, anchor=anchor), ensure_ascii=False)


def build_provenance_tool() -> str:
    """Rebuild the provenance-back index under corpus/nvidia/INDEX."""
    return json.dumps(build_provenance_index(), ensure_ascii=False)


def read_file(path: str, max_lines: int = 500) -> str:
    """Read a file and return its contents (truncated if too long)."""
    p = Path(path)
    if not p.exists():
        return f"(file not found: {path})"
    try:
        text = p.read_text()
    except Exception as e:
        return f"(error reading {path}: {e})"
    lines = text.split("\n")
    if len(lines) > max_lines:
        return "\n".join(lines[:max_lines]) + f"\n\n... (truncated at {max_lines}/{len(lines)} lines)"
    return text


def write_file(path: str, content: str) -> str:
    """Write content to a file. Only allows paths under the project root."""
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    p = p.resolve()
    try:
        p.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        return f"(error: write rejected — path outside project root: {path})"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"(wrote {len(content)} bytes to {p})"


def list_dir(path: str) -> str:
    """List directory contents."""
    p = Path(path)
    if not p.is_dir():
        return f"(not a directory: {path})"
    entries = sorted(p.iterdir())
    lines = []
    for e in entries[:100]:
        kind = "d" if e.is_dir() else "f"
        lines.append(f"[{kind}] {e.name}")
    if not lines:
        return "(empty directory)"
    return "\n".join(lines)


def run_bash(command: str, timeout: int = 120) -> str:
    """Execute a shell command locally. Blocked patterns raise an error."""
    blocked = ["rm -rf /", "sudo ", "mkfs", "> /dev/"]
    for b in blocked:
        if b in command:
            return f"(error: blocked command pattern: '{b}')"
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=timeout, cwd=str(PROJECT_ROOT),
        )
    except subprocess.TimeoutExpired:
        return f"(error: command timed out after {timeout}s)"
    output = result.stdout
    if len(output) > 8000:
        output = output[:8000] + "\n... (truncated)"
    if result.returncode != 0:
        output += f"\n(exit code: {result.returncode})"
        if result.stderr:
            output += f"\nstderr: {result.stderr[:3000]}"
    return output if output.strip() else "(no output)"


def run_on_gpu(command: str, timeout: int = 180) -> str:
    """Execute a command on the remote GPU host via SSH.

    Automatically prepends PATH and PYTHONPATH setup for CUDA + pylib.
    The `command` is run inside the remote kernel-kb-mvp directory.
    """
    remote_cmd = (
        f"cd {REMOTE_DIR} && "
        f"export PATH={REMOTE_CUDA_PATH}:$PATH && "
        f"export PYTHONPATH={REMOTE_PYLIB} && "
        f"{command}"
    )
    full = ["ssh", REMOTE_HOST, remote_cmd]
    try:
        result = subprocess.run(full, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return f"(error: SSH command timed out after {timeout}s)"
    output = result.stdout
    if len(output) > 8000:
        output = output[:8000] + "\n... (truncated)"
    if result.returncode not in (0, 139):
        output += f"\n(exit code: {result.returncode})"
        if result.stderr:
            stderr = result.stderr.replace("bash: warning: setlocale: LC_ALL: cannot change locale (en_US.UTF-8)\n", "")
            if stderr.strip():
                output += f"\nstderr: {stderr[:3000]}"
    return output if output.strip() else "(no output)"


def sync_to_gpu() -> str:
    """Rsync local  and tools/ to the remote GPU host."""
    cmd = (
        f"rsync -avz --exclude '.git' --exclude '__pycache__' "
        f"{PROJECT_ROOT}/ {REMOTE_HOST}:{REMOTE_DIR}/"
    )
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return "(error: rsync timed out)"
    return result.stdout[-2000:] if result.stdout else "(no output)"


def sync_from_gpu(remote_subpath: str = "") -> str:
    """Rsync a subpath from the remote GPU host back to local."""
    cmd = (
        f"rsync -avz {REMOTE_HOST}:{REMOTE_DIR}/{remote_subpath} "
        f"{PROJECT_ROOT}/{remote_subpath}"
    )
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return "(error: rsync timed out)"
    return result.stdout[-2000:] if result.stdout else "(no output)"


TOOL_REGISTRY = {
    "grep_source": grep_source,
    "source_search": source_search_tool,
    "source_read": source_read_tool,
    "source_list": source_list_tool,
    "source_resolve": source_resolve_tool,
    "provenance_walk": provenance_walk_tool,
    "provenance_back": provenance_back_tool,
    "build_provenance": build_provenance_tool,
    "read_file": read_file,
    "write_file": write_file,
    "list_dir": list_dir,
    "run_bash": run_bash,
    "run_on_gpu": run_on_gpu,
    "sync_to_gpu": sync_to_gpu,
    "sync_from_gpu": sync_from_gpu,
}
