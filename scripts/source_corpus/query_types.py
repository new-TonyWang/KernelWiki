"""Structured result types for the source corpus service."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SourceHit:
    source_id: str
    source_type: str
    path: str
    anchor: str
    title: str
    line_start: int
    line_end: int
    snippet: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SourceReadResult:
    path: str
    anchor: str
    title: str
    content: str
    truncated: bool
    line_start: int
    line_end: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProvenanceRef:
    path: str
    anchor: str
    excerpt: str = ""
    source_id: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ResponseEnvelope:
    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    error_code: str = ""
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
