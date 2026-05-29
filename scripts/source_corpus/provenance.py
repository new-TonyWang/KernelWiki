"""Provenance helpers for source corpus references."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .registry import REPO_ROOT as KNOWLEDGE_ROOT, SOURCE_CORPUS_INDEX_ROOT

from .query_types import ProvenanceRef, ResponseEnvelope
from .registry import entry_for_corpus_path, resolve_corpus_path

PROVENANCE_BACK_PATH = SOURCE_CORPUS_INDEX_ROOT / "provenance-back.jsonl"
SKIP_DIRS = {"00-foundation", "05-source-corpus", "70-reasoning", "templates"}
SKIP_NAMES = {"AGENTS.md", "README.md"}


def _read_frontmatter(path: Path) -> dict[str, Any]:
    text = path.read_text()
    if not text.startswith("---\n"):
        return {}
    end = text.find("\n---\n", 4)
    if end < 0:
        return {}
    return yaml.safe_load(text[4:end]) or {}


def _iter_knowledge_entries(root: Path) -> list[Path]:
    out: list[Path] = []
    for path in root.rglob("*.md"):
        rel = path.relative_to(root)
        if rel.parts[0] in SKIP_DIRS:
            continue
        if path.name in SKIP_NAMES or path.name.startswith("_"):
            continue
        out.append(path)
    return out


def provenance_walk(knowledge_path: str) -> dict:
    path = Path(knowledge_path)
    if not path.is_absolute():
        path = (KNOWLEDGE_ROOT.parent / path).resolve()
    if not path.exists():
        return ResponseEnvelope(
            ok=False,
            error_code="SOURCE_NOT_FOUND",
            message=f"knowledge file not found: {knowledge_path}",
        ).to_dict()

    fm = _read_frontmatter(path)
    refs = []
    for item in fm.get("source", []) or []:
        source_path = item.get("path", "")
        source_id = ""
        try:
            entry = entry_for_corpus_path(source_path)
            if entry is not None:
                source_id = entry.source_id
        except Exception:
            source_id = ""
        refs.append(
            ProvenanceRef(
                path=source_path,
                anchor=item.get("anchor", ""),
                excerpt=item.get("excerpt", ""),
                source_id=source_id,
                note=item.get("note", ""),
            ).to_dict()
        )
    return ResponseEnvelope(
        ok=True,
        data={"entry": str(path), "sources": refs, "count": len(refs)},
    ).to_dict()


def build_provenance_index(output_path: Path | None = None) -> dict:
    target = output_path or PROVENANCE_BACK_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for entry in _iter_knowledge_entries(KNOWLEDGE_ROOT):
        fm = _read_frontmatter(entry)
        for item in fm.get("source", []) or []:
            src_path = item.get("path", "")
            anchor = item.get("anchor", "")
            if not src_path:
                continue
            rows.append(
                {
                    "source_path": src_path,
                    "resolved_path": str(resolve_corpus_path(src_path)),
                    "anchor": anchor,
                    "referenced_by": str(entry.relative_to(KNOWLEDGE_ROOT.parent)),
                }
            )

    with target.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return ResponseEnvelope(
        ok=True,
        data={"output_path": str(target), "rows": len(rows)},
    ).to_dict()


def provenance_back(path: str, anchor: str | None = None) -> dict:
    if not PROVENANCE_BACK_PATH.exists():
        build_provenance_index()

    resolved_query = str(resolve_corpus_path(path))
    matches = []
    with PROVENANCE_BACK_PATH.open() as f:
        for line in f:
            row = json.loads(line)
            if row["source_path"] != path and row.get("resolved_path") != resolved_query:
                continue
            if anchor and row.get("anchor") != anchor:
                continue
            matches.append(row["referenced_by"])

    return ResponseEnvelope(
        ok=True,
        data={
            "source": {"path": path, "anchor": anchor or ""},
            "referenced_by": matches,
            "count": len(matches),
        },
    ).to_dict()
