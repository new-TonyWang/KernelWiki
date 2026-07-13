"""Helpers for reading sections and anchors from source corpus files."""

from __future__ import annotations

import re
from pathlib import Path


LINE_RANGE_RE = re.compile(r"^L(?P<start>\d+)(?:-L(?P<end>\d+))?$")


def slugify_heading(value: str) -> str:
    text = value.strip().lstrip("#").strip().lower()
    text = re.sub(r"[`*_]+", "", text)
    text = re.sub(r"[^a-z0-9.\- ]+", "", text)
    text = re.sub(r"\s+", "-", text)
    return text.strip("-")


def extract_title(path: Path) -> str:
    try:
        with path.open() as f:
            for line in f:
                if line.startswith("#"):
                    return line.lstrip("#").strip()
    except OSError:
        return path.name
    return path.name


def _line_bounds(lines: list[str], start: int, end: int) -> tuple[int, int, str]:
    start = max(start, 1)
    end = max(start, min(end, len(lines)))
    return start, end, "".join(lines[start - 1 : end])


def read_by_anchor(path: Path, anchor: str) -> tuple[int, int, str, str]:
    lines = path.read_text().splitlines(keepends=True)
    match = LINE_RANGE_RE.match(anchor.strip())
    if match:
        start = int(match.group("start"))
        end = int(match.group("end") or start)
        start, end, content = _line_bounds(lines, start, end)
        return start, end, content, extract_title(path)

    token = anchor.strip().lstrip("#")
    return read_by_section(path, token)


def read_by_section(path: Path, section: str) -> tuple[int, int, str, str]:
    lines = path.read_text().splitlines(keepends=True)
    wanted = slugify_heading(section)
    headings: list[tuple[int, int, str]] = []
    for idx, line in enumerate(lines, start=1):
        if not line.startswith("#"):
            continue
        level = len(line) - len(line.lstrip("#"))
        title = line.lstrip("#").strip()
        headings.append((idx, level, title))

    for i, (start, level, title) in enumerate(headings):
        title_slug = slugify_heading(title)
        if title_slug != wanted and not title_slug.startswith(wanted) and not wanted.startswith(title_slug):
            continue
        end = len(lines)
        for next_start, next_level, _ in headings[i + 1 :]:
            if next_level <= level:
                end = next_start - 1
                break
        _, end, content = _line_bounds(lines, start, end)
        return start, end, content, title

    raise ValueError(f"section not found: {section}")


def heading_before_line(path: Path, line_no: int) -> str:
    lines = path.read_text().splitlines()
    for idx in range(min(line_no - 1, len(lines) - 1), -1, -1):
        line = lines[idx]
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return path.name
