"""CLI entrypoints for source corpus retrieval."""

from __future__ import annotations

import json
from pathlib import Path

import click

from .provenance import build_provenance_index, provenance_back, provenance_walk
from .service import list_sources, resolve_source, source_read, source_search


def _emit(payload: dict) -> None:
    click.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@click.group()
def cli() -> None:
    """Source corpus retrieval CLI."""


@cli.command("search")
@click.argument("query")
@click.option("--scope", default=None, help="Logical source scope, e.g. cuda-official or source-code/cutlass.")
@click.option("--source-type", "source_type", multiple=True, help="Filter by source type.")
@click.option("--entity-kind", default=None, help="api | feature | operator | general")
@click.option("--top-k", default=10, show_default=True, type=int)
@click.option("--regex", is_flag=True, help="Interpret query as regex.")
def search_cmd(query: str, scope: str | None, source_type: tuple[str, ...], entity_kind: str | None, top_k: int, regex: bool) -> None:
    _emit(
        source_search(
            query=query,
            scope=scope,
            source_types=list(source_type) or None,
            entity_kind=entity_kind,
            top_k=top_k,
            regex=regex,
        )
    )


@cli.command("read")
@click.argument("path")
@click.option("--anchor", default=None)
@click.option("--section", default=None)
@click.option("--max-chars", default=6000, show_default=True, type=int)
@click.option("--focused-query", default=None)
def read_cmd(path: str, anchor: str | None, section: str | None, max_chars: int, focused_query: str | None) -> None:
    _emit(
        source_read(
            path=path,
            anchor=anchor,
            section=section,
            max_chars=max_chars,
            focused_query=focused_query,
        )
    )


@cli.command("list")
@click.option("--scope", default=None)
@click.option("--source-type", default=None)
@click.option("--keyword", default=None)
@click.option("--year", default=None)
@click.option("--limit", default=50, show_default=True, type=int)
def list_cmd(scope: str | None, source_type: str | None, keyword: str | None, year: str | None, limit: int) -> None:
    _emit(
        list_sources(
            scope=scope,
            source_type=source_type,
            keyword=keyword,
            year=year,
            limit=limit,
        )
    )


@cli.command("resolve")
@click.argument("ref")
def resolve_cmd(ref: str) -> None:
    _emit(resolve_source(ref))


@cli.command("provenance-walk")
@click.argument("knowledge_path")
def provenance_walk_cmd(knowledge_path: str) -> None:
    _emit(provenance_walk(knowledge_path))


@cli.command("provenance-back")
@click.argument("path")
@click.option("--anchor", default=None)
def provenance_back_cmd(path: str, anchor: str | None) -> None:
    _emit(provenance_back(path, anchor=anchor))


@cli.command("build-provenance")
@click.option("--output", type=click.Path(path_type=Path), default=None)
def build_provenance_cmd(output: Path | None) -> None:
    _emit(build_provenance_index(output_path=output))


if __name__ == "__main__":
    cli()
