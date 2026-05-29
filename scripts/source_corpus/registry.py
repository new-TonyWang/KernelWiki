"""Manifest-backed source corpus registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from agent.shared.config import SOURCE_CORPUS_MANIFEST, SOURCE_CORPUS_ROOT


@dataclass
class SourceEntry:
    source_id: str
    title: str
    source_type: str
    logical_root: str
    local_path: Path
    upstream_url: str = ""
    version: str = ""
    fetched_at: str = ""
    aliases: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def corpus_root(self) -> Path:
        return SOURCE_CORPUS_ROOT / self.logical_root

    def matches_scope(self, scope: str | None) -> bool:
        if not scope:
            return True
        norm = scope.strip().strip("/")
        if not norm:
            return True
        candidates = {
            self.source_id,
            self.source_type,
            self.logical_root,
            f"05-source-corpus/{self.logical_root}",
            *self.aliases,
        }
        if norm in candidates:
            return True
        if self.logical_root.startswith(norm) or self.source_id.startswith(norm):
            return True
        return any(alias.startswith(norm) for alias in self.aliases)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "title": self.title,
            "source_type": self.source_type,
            "logical_root": self.logical_root,
            "local_path": str(self.local_path),
            "upstream_url": self.upstream_url,
            "version": self.version,
            "fetched_at": self.fetched_at,
            "aliases": list(self.aliases),
            "tags": list(self.tags),
        }


def load_manifest(manifest_path: Path | None = None) -> list[SourceEntry]:
    path = manifest_path or SOURCE_CORPUS_MANIFEST
    items = yaml.safe_load(path.read_text()) or []
    entries: list[SourceEntry] = []
    for item in items:
        entries.append(
            SourceEntry(
                source_id=item["source_id"],
                title=item["title"],
                source_type=item["source_type"],
                logical_root=item["logical_root"],
                local_path=Path(item["local_path"]),
                upstream_url=item.get("upstream_url", ""),
                version=item.get("version", ""),
                fetched_at=item.get("fetched_at", ""),
                aliases=list(item.get("aliases", []) or []),
                tags=list(item.get("tags", []) or []),
            )
        )
    return entries


def find_entries(
    scope: str | None = None,
    source_type: str | None = None,
    manifest_path: Path | None = None,
) -> list[SourceEntry]:
    entries = load_manifest(manifest_path)
    out = [entry for entry in entries if entry.matches_scope(scope)]
    if source_type:
        out = [entry for entry in out if entry.source_type == source_type]
    return out


def entry_for_corpus_path(path: str | Path, manifest_path: Path | None = None) -> SourceEntry | None:
    value = str(path)
    norm = value.strip()
    if norm.startswith("05-source-corpus/"):
        norm = norm[len("05-source-corpus/") :]
    for entry in load_manifest(manifest_path):
        if norm == entry.logical_root or norm.startswith(entry.logical_root + "/"):
            return entry
    return None


def resolve_corpus_path(path: str | Path, manifest_path: Path | None = None) -> Path:
    value = str(path)
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve()
    entry = entry_for_corpus_path(value, manifest_path)
    if entry is None:
        return (SOURCE_CORPUS_ROOT / candidate).resolve()
    suffix = value.strip()
    if suffix.startswith("05-source-corpus/"):
        suffix = suffix[len("05-source-corpus/") :]
    rel = Path(suffix).relative_to(entry.logical_root)
    return (entry.local_path / rel).resolve()


def to_corpus_path(abs_path: Path, entry: SourceEntry) -> str:
    rel = abs_path.resolve().relative_to(entry.local_path.resolve())
    joined = Path("05-source-corpus") / entry.logical_root / rel
    return joined.as_posix()
