#!/usr/bin/env python3
"""Manage KernelWiki MCP HTTP bearer tokens stored in SQLite.

Examples:

  python3 scripts/mcp_token_admin.py --db /var/lib/kernelwiki/tokens.sqlite add laptop
  python3 scripts/mcp_token_admin.py --db /var/lib/kernelwiki/tokens.sqlite list
  python3 scripts/mcp_token_admin.py --db /var/lib/kernelwiki/tokens.sqlite disable 3
  python3 scripts/mcp_token_admin.py --db /var/lib/kernelwiki/tokens.sqlite rotate 3
  python3 scripts/mcp_token_admin.py --db /var/lib/kernelwiki/tokens.sqlite delete 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mcp_token_store import TokenStore  # noqa: E402


DEFAULT_DB = "data/mcp_tokens.sqlite3"


def _json_print(obj: Any) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True))


def _print_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("(no tokens)")
        return
    headers = ["id", "name", "enabled", "use_count", "created_at", "updated_at", "last_used_at", "note"]
    widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            widths[h] = max(widths[h], len(str(row.get(h, ""))))
    print("  ".join(h.ljust(widths[h]) for h in headers))
    print("  ".join("-" * widths[h] for h in headers))
    for row in rows:
        print("  ".join(str(row.get(h, "")).ljust(widths[h]) for h in headers))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CRUD admin for KernelWiki MCP HTTP bearer tokens")
    parser.add_argument(
        "--db",
        default=os.environ.get("MCP_TOKEN_DB", DEFAULT_DB),
        help=f"SQLite token DB path (default: {DEFAULT_DB}; env: MCP_TOKEN_DB)",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON output")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="initialize the token database")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("add", help="create a new token")
    p.add_argument("name", help="unique token name")
    p.add_argument("--token", help="use an explicit token value instead of generating one")
    p.add_argument("--disabled", action="store_true", help="create token disabled")
    p.add_argument("--note", help="optional note")
    p.set_defaults(func=cmd_add)

    p = sub.add_parser("list", help="list tokens without revealing token values")
    p.add_argument("--active-only", action="store_true", help="hide disabled tokens")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("get", help="show one token by id")
    p.add_argument("id", type=int)
    p.set_defaults(func=cmd_get)

    p = sub.add_parser("update", help="update token metadata")
    p.add_argument("id", type=int)
    p.add_argument("--name")
    p.add_argument("--note")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--enable", action="store_true")
    group.add_argument("--disable", action="store_true")
    p.set_defaults(func=cmd_update)

    p = sub.add_parser("enable", help="enable a token")
    p.add_argument("id", type=int)
    p.set_defaults(func=cmd_enable)

    p = sub.add_parser("disable", help="disable a token")
    p.add_argument("id", type=int)
    p.set_defaults(func=cmd_disable)

    p = sub.add_parser("rotate", help="replace a token secret and print the new value once")
    p.add_argument("id", type=int)
    p.set_defaults(func=cmd_rotate)

    p = sub.add_parser("delete", help="delete a token")
    p.add_argument("id", type=int)
    p.add_argument("-y", "--yes", action="store_true", help="do not ask for confirmation")
    p.set_defaults(func=cmd_delete)

    return parser


def _store(args: argparse.Namespace) -> TokenStore:
    return TokenStore(args.db)


def _emit(args: argparse.Namespace, obj: Any, *, table: bool = False) -> None:
    if args.json or not table:
        _json_print(obj)
    else:
        _print_table(obj)


def cmd_init(args: argparse.Namespace) -> int:
    TokenStore(args.db)
    _emit(args, {"ok": True, "db": str(Path(args.db).expanduser().resolve())})
    return 0


def cmd_add(args: argparse.Namespace) -> int:
    row = _store(args).create_token(args.name, token=args.token, enabled=not args.disabled, note=args.note)
    _emit(args, {"ok": True, "token": row})
    if not args.json:
        print("\nSAVE THIS TOKEN NOW. It is stored hashed and cannot be shown again.")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    rows = _store(args).list_tokens(include_disabled=not args.active_only)
    _emit(args, rows, table=True)
    return 0


def cmd_get(args: argparse.Namespace) -> int:
    row = _store(args).get_token(args.id)
    if not row:
        print(f"token id {args.id} not found", file=sys.stderr)
        return 1
    _emit(args, row)
    return 0


def cmd_update(args: argparse.Namespace) -> int:
    enabled = None
    if args.enable:
        enabled = True
    elif args.disable:
        enabled = False
    row = _store(args).update_token(args.id, name=args.name, enabled=enabled, note=args.note)
    _emit(args, {"ok": True, "token": row})
    return 0


def cmd_enable(args: argparse.Namespace) -> int:
    row = _store(args).update_token(args.id, enabled=True)
    _emit(args, {"ok": True, "token": row})
    return 0


def cmd_disable(args: argparse.Namespace) -> int:
    row = _store(args).update_token(args.id, enabled=False)
    _emit(args, {"ok": True, "token": row})
    return 0


def cmd_rotate(args: argparse.Namespace) -> int:
    row = _store(args).rotate_token(args.id)
    _emit(args, {"ok": True, "token": row})
    if not args.json:
        print("\nSAVE THIS TOKEN NOW. It is stored hashed and cannot be shown again.")
    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    if not args.yes:
        answer = input(f"Delete token id {args.id}? Type 'yes' to continue: ")
        if answer != "yes":
            print("aborted")
            return 2
    deleted = _store(args).delete_token(args.id)
    _emit(args, {"ok": deleted, "deleted": deleted, "id": args.id})
    return 0 if deleted else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:
        if getattr(args, "json", False):
            _json_print({"ok": False, "error": type(exc).__name__, "message": str(exc)})
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
