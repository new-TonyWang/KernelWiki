"""Search, read, list, and resolve services for the source corpus."""

from __future__ import annotations

import subprocess
from pathlib import Path

from agent.shared.config import SOURCE_CORPUS_ROOT

from .query_types import ResponseEnvelope, SourceHit, SourceReadResult
from .reader import heading_before_line, read_by_anchor, read_by_section
from .registry import find_entries, load_manifest, resolve_corpus_path, to_corpus_path


def _search_one_root(query: str, root: Path, regex: bool, top_k: int) -> list[tuple[Path, int, str]]:
    cmd = ["rg", "--follow", "--line-number", "--color", "never", "-m", str(max(top_k * 5, 20))]
    cmd.append(query if regex else "-F")
    if not regex:
        cmd.append(query)
    cmd.append(str(root))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr.strip() or f"rg exit {result.returncode}")

    rows: list[tuple[Path, int, str]] = []
    for line in result.stdout.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        file_part, line_no, text = parts
        try:
            rows.append((Path(file_part), int(line_no), text.strip()))
        except ValueError:
            continue
    return rows


def source_search(
    query: str,
    scope: str | None = None,
    source_types: list[str] | None = None,
    entity_kind: str | None = None,
    top_k: int = 10,
    regex: bool = False,
) -> dict:
    entries = find_entries(scope=scope)
    if source_types:
        wanted = set(source_types)
        entries = [entry for entry in entries if entry.source_type in wanted]
    if not entries:
        return ResponseEnvelope(
            ok=False,
            error_code="SCOPE_NOT_FOUND",
            message=f"no source entries matched scope={scope!r}",
        ).to_dict()

    hits: list[SourceHit] = []
    for entry in entries:
        if not entry.local_path.exists():
            continue
        for abs_path, line_no, text in _search_one_root(query, entry.local_path, regex, top_k):
            title = heading_before_line(abs_path, line_no)
            score = 1.0
            lowered = text.lower()
            if query.lower() in lowered:
                score += 0.5
            if title and query.lower() in title.lower():
                score += 0.5
            if entry.source_type == "cuda-official":
                score += 0.2
            if entry.source_type == "source-code" and entity_kind in {"operator", "api"}:
                score += 0.2
            hits.append(
                SourceHit(
                    source_id=entry.source_id,
                    source_type=entry.source_type,
                    path=to_corpus_path(abs_path, entry),
                    anchor=f"L{line_no}-L{line_no}",
                    title=title,
                    line_start=line_no,
                    line_end=line_no,
                    snippet=text[:400],
                    score=round(score, 3),
                )
            )

    hits.sort(key=lambda item: (-item.score, item.path, item.line_start))
    uniq: list[SourceHit] = []
    seen: set[tuple[str, int]] = set()
    for hit in hits:
        key = (hit.path, hit.line_start)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(hit)
        if len(uniq) >= top_k:
            break

    return ResponseEnvelope(
        ok=True,
        data={
            "query": query,
            "scope": scope or "",
            "entity_kind": entity_kind or "general",
            "total_hits": len(hits),
            "hits": [hit.to_dict() for hit in uniq],
        },
    ).to_dict()


def source_read(
    path: str,
    anchor: str | None = None,
    section: str | None = None,
    max_chars: int = 6000,
    focused_query: str | None = None,
) -> dict:
    resolved = resolve_corpus_path(path)
    if not resolved.exists():
        return ResponseEnvelope(
            ok=False,
            error_code="SOURCE_NOT_FOUND",
            message=f"source file not found: {path}",
        ).to_dict()

    try:
        if anchor:
            line_start, line_end, content, title = read_by_anchor(resolved, anchor)
            effective_anchor = anchor
        elif section:
            line_start, line_end, content, title = read_by_section(resolved, section)
            effective_anchor = section
        else:
            text = resolved.read_text()
            line_start = 1
            line_end = len(text.splitlines())
            content = text
            title = resolved.name
            effective_anchor = "full-file"
    except ValueError as exc:
        return ResponseEnvelope(
            ok=False,
            error_code="ANCHOR_NOT_FOUND",
            message=str(exc),
        ).to_dict()

    if focused_query and focused_query.lower() in content.lower():
        lines = content.splitlines()
        hit_line = next(
            (idx for idx, line in enumerate(lines) if focused_query.lower() in line.lower()),
            0,
        )
        start = max(hit_line - 15, 0)
        end = min(hit_line + 15, len(lines))
        content = "\n".join(lines[start:end])
        line_start += start
        line_end = min(line_start + (end - start) - 1, line_end)

    truncated = len(content) > max_chars
    if truncated:
        content = content[:max_chars] + "\n... (truncated)"

    result = SourceReadResult(
        path=path if path.startswith("05-source-corpus/") else path,
        anchor=effective_anchor,
        title=title,
        content=content,
        truncated=truncated,
        line_start=line_start,
        line_end=line_end,
    )
    return ResponseEnvelope(ok=True, data=result.to_dict()).to_dict()


def list_sources(
    scope: str | None = None,
    source_type: str | None = None,
    keyword: str | None = None,
    year: str | None = None,
    limit: int = 50,
) -> dict:
    del year  # reserved for future manifest enrichment
    entries = find_entries(scope=scope, source_type=source_type)
    items = []
    for entry in entries:
        if keyword:
            haystack = " ".join(
                [entry.source_id, entry.title, entry.logical_root, *entry.aliases, *entry.tags]
            ).lower()
            if keyword.lower() not in haystack:
                continue
        items.append(entry.as_dict())
    items = items[:limit]
    return ResponseEnvelope(ok=True, data={"items": items, "count": len(items)}).to_dict()


def resolve_source(ref: str) -> dict:
    for entry in load_manifest():
        if ref in {entry.source_id, entry.logical_root, *entry.aliases}:
            return ResponseEnvelope(ok=True, data=entry.as_dict()).to_dict()

    resolved = resolve_corpus_path(ref)
    if resolved.exists():
        return ResponseEnvelope(
            ok=True,
            data={
                "input": ref,
                "resolved_path": str(resolved),
                "exists": True,
                "under_source_corpus": str(resolved).startswith(str(SOURCE_CORPUS_ROOT)),
            },
        ).to_dict()

    return ResponseEnvelope(
        ok=False,
        error_code="SOURCE_NOT_FOUND",
        message=f"could not resolve source reference: {ref}",
    ).to_dict()
