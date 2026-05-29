"""Manifest-backed source corpus registry.

Reads corpus/MANIFEST.yaml and provides SourceEntry objects with resolved paths.
Supports both the new schema (tier, remote.url, default_ref) and legacy fields.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Resolve paths relative to repo root, not agent config
_SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = _SCRIPT_DIR.parent.parent
SOURCE_CORPUS_ROOT = REPO_ROOT / "corpus"
SOURCE_CORPUS_MANIFEST = SOURCE_CORPUS_ROOT / "MANIFEST.yaml"
SOURCE_CORPUS_INDEX_ROOT = SOURCE_CORPUS_ROOT / "INDEX"

PLACEHOLDER_RE = re.compile(r'\{\{(\w+)\}\}')


@dataclass
class SourceEntry:
    source_id: str
    title: str
    tier: str  # "in-git" or "external"
    local_path: str  # Raw path (may contain {{placeholders}})
    remote_url: str = ""
    default_ref: str = ""
    fetched_at: str = ""
    aliases: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def resolved_path(self, variables: dict[str, str] | None = None) -> Path | None:
        """Resolve local_path, substituting placeholders if needed."""
        path_str = self.local_path
        if PLACEHOLDER_RE.search(path_str):
            if not variables:
                return None
            for m in PLACEHOLDER_RE.finditer(path_str):
                var = m.group(1)
                if var in variables:
                    path_str = path_str.replace(f"{{{{{var}}}}}", variables[var])
            if PLACEHOLDER_RE.search(path_str):
                return None
            return Path(path_str)
        return SOURCE_CORPUS_ROOT / path_str

    def corpus_root(self, variables: dict[str, str] | None = None) -> Path:
        """Return the resolved corpus directory path."""
        resolved = self.resolved_path(variables)
        return resolved if resolved else SOURCE_CORPUS_ROOT / self.local_path

    def matches_scope(self, scope: str | None) -> bool:
        if not scope:
            return True
        norm = scope.strip().strip("/")
        if not norm:
            return True
        candidates = {self.source_id, *self.aliases}
        if norm in candidates:
            return True
        if self.source_id.startswith(norm):
            return True
        return any(alias.startswith(norm) for alias in self.aliases)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "title": self.title,
            "tier": self.tier,
            "local_path": self.local_path,
            "remote_url": self.remote_url,
            "default_ref": self.default_ref,
            "aliases": list(self.aliases),
            "tags": list(self.tags),
        }


def load_manifest(manifest_path: Path | None = None) -> list[SourceEntry]:
    """Load MANIFEST.yaml and return SourceEntry objects."""
    path = manifest_path or SOURCE_CORPUS_MANIFEST
    items = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    entries: list[SourceEntry] = []
    for item in items:
        remote = item.get("remote", {}) or {}
        entries.append(
            SourceEntry(
                source_id=item["source_id"],
                title=item.get("title", ""),
                tier=item.get("tier", "in-git"),
                local_path=item.get("local_path", ""),
                remote_url=remote.get("url", item.get("upstream_url", "")),
                default_ref=item.get("default_ref", ""),
                fetched_at=item.get("fetched_at", ""),
                aliases=list(item.get("aliases", []) or []),
                tags=list(item.get("tags", []) or []),
            )
        )
    return entries


def find_entries(scope: str | None = None, manifest_path: Path | None = None) -> list[SourceEntry]:
    """Load manifest and filter by scope."""
    entries = load_manifest(manifest_path)
    if scope:
        return [e for e in entries if e.matches_scope(scope)]
    return entries


def resolve_corpus_path(path_str: str) -> Path:
    """Resolve a corpus path (tier-1 relative or tier-2 placeholder)."""
    if PLACEHOLDER_RE.search(path_str):
        variables = load_localize_variables()
        result = path_str
        for m in PLACEHOLDER_RE.finditer(path_str):
            var = m.group(1)
            if var in variables:
                result = result.replace(f"{{{{{var}}}}}", variables[var])
        return Path(result)
    return SOURCE_CORPUS_ROOT / path_str


def to_corpus_path(abs_path: Path) -> str | None:
    """Convert an absolute path back to a corpus-relative string."""
    try:
        return str(abs_path.relative_to(SOURCE_CORPUS_ROOT))
    except ValueError:
        return None


def entry_for_corpus_path(abs_path: Path) -> SourceEntry | None:
    """Find the SourceEntry that contains the given absolute path."""
    variables = load_localize_variables()
    for entry in load_manifest():
        resolved = entry.resolved_path(variables)
        if resolved and abs_path.is_relative_to(resolved):
            return entry
    return None


def load_localize_variables() -> dict[str, str]:
    """Load localize.yaml variables for resolving tier-2 paths."""
    config_path = SOURCE_CORPUS_ROOT / "localize.yaml"
    if not config_path.exists():
        return {}
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    vars_ = dict(config.get("vars", {}) or {})
    derived = dict(config.get("derived", {}) or {})
    env_config = config.get("env", {}) or {}
    if env_config.get("inherit", False):
        for key in list(vars_) + list(derived):
            env_val = os.environ.get(key)
            if env_val and key not in vars_ and key not in derived:
                vars_[key] = env_val
    resolved = dict(vars_)
    for key, val in derived.items():
        if isinstance(val, str):
            for m in PLACEHOLDER_RE.finditer(val):
                ref = m.group(1)
                if ref in resolved:
                    val = val.replace(f"{{{{{ref}}}}}", str(resolved[ref]))
        resolved[key] = val
    return resolved
