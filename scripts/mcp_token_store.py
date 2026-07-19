#!/usr/bin/env python3
"""SQLite-backed bearer token store for KernelWiki MCP HTTP auth.

The database stores only salted PBKDF2 hashes.  Plain token values are returned
exactly once when a token is created or rotated.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Any


HASH_ITERATIONS = 200_000
TOKEN_BYTES = 32
SCHEMA_VERSION = 1


def now_ts() -> int:
    return int(time.time())


def generate_token() -> str:
    return secrets.token_urlsafe(TOKEN_BYTES)


def _hash_token(token: str, salt: bytes | None = None) -> tuple[str, str]:
    if not isinstance(token, str) or not token:
        raise ValueError("token must be a non-empty string")
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", token.encode("utf-8"), salt, HASH_ITERATIONS)
    return (
        base64.urlsafe_b64encode(salt).decode("ascii"),
        base64.urlsafe_b64encode(digest).decode("ascii"),
    )


def _verify_token(token: str, salt_b64: str, hash_b64: str) -> bool:
    try:
        salt = base64.urlsafe_b64decode(salt_b64.encode("ascii"))
        _, candidate = _hash_token(token, salt=salt)
    except Exception:
        return False
    return hmac.compare_digest(candidate, hash_b64)


class TokenStore:
    """Small SQLite token repository.

    The class opens short-lived connections per operation so a long-running MCP
    server observes token CRUD changes made by the CLI without restart.
    """

    def __init__(self, db_path: str | os.PathLike[str]) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        if self.db_path.parent:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                INSERT OR IGNORE INTO meta(key, value)
                VALUES ('schema_version', '1');

                CREATE TABLE IF NOT EXISTS tokens (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    salt TEXT NOT NULL,
                    token_hash TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    last_used_at INTEGER,
                    use_count INTEGER NOT NULL DEFAULT 0,
                    note TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_tokens_enabled
                    ON tokens(enabled);
                """
            )

    def create_token(
        self,
        name: str,
        *,
        token: str | None = None,
        enabled: bool = True,
        note: str | None = None,
    ) -> dict[str, Any]:
        name = _validate_name(name)
        token = token or generate_token()
        salt, token_hash = _hash_token(token)
        ts = now_ts()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO tokens(name, salt, token_hash, enabled, created_at, updated_at, note)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (name, salt, token_hash, 1 if enabled else 0, ts, ts, note),
            )
            row = conn.execute("SELECT * FROM tokens WHERE id = ?", (cur.lastrowid,)).fetchone()
        result = self._public_row(row)
        result["token"] = token
        return result

    def ensure_bootstrap_token(self, name: str, token: str, *, note: str | None = None) -> bool:
        """Insert a configured bootstrap token if it is not already present.

        Returns True if a new row was inserted, False if the name already
        exists.  This is useful when migrating from MCP_AUTH_TOKEN to a DB.
        """
        name = _validate_name(name)
        salt, token_hash = _hash_token(token)
        ts = now_ts()
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO tokens(name, salt, token_hash, enabled, created_at, updated_at, note)
                    VALUES (?, ?, ?, 1, ?, ?, ?)
                    """,
                    (name, salt, token_hash, ts, ts, note),
                )
            return True
        except sqlite3.IntegrityError:
            return False

    def list_tokens(self, *, include_disabled: bool = True) -> list[dict[str, Any]]:
        sql = "SELECT * FROM tokens"
        params: tuple[Any, ...] = ()
        if not include_disabled:
            sql += " WHERE enabled = 1"
        sql += " ORDER BY id"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._public_row(r) for r in rows]

    def get_token(self, token_id: int | None = None, *, name: str | None = None) -> dict[str, Any] | None:
        if token_id is None and name is None:
            raise ValueError("token_id or name is required")
        if token_id is not None:
            sql, params = "SELECT * FROM tokens WHERE id = ?", (int(token_id),)
        else:
            sql, params = "SELECT * FROM tokens WHERE name = ?", (_validate_name(name or ""),)
        with self._connect() as conn:
            row = conn.execute(sql, params).fetchone()
        return self._public_row(row) if row else None

    def update_token(
        self,
        token_id: int,
        *,
        name: str | None = None,
        enabled: bool | None = None,
        note: str | None = None,
    ) -> dict[str, Any]:
        fields = []
        params: list[Any] = []
        if name is not None:
            fields.append("name = ?")
            params.append(_validate_name(name))
        if enabled is not None:
            fields.append("enabled = ?")
            params.append(1 if enabled else 0)
        if note is not None:
            fields.append("note = ?")
            params.append(note)
        if not fields:
            existing = self.get_token(token_id)
            if not existing:
                raise KeyError(f"token id {token_id} not found")
            return existing
        fields.append("updated_at = ?")
        params.append(now_ts())
        params.append(int(token_id))
        with self._connect() as conn:
            cur = conn.execute(f"UPDATE tokens SET {', '.join(fields)} WHERE id = ?", params)
            if cur.rowcount == 0:
                raise KeyError(f"token id {token_id} not found")
            row = conn.execute("SELECT * FROM tokens WHERE id = ?", (int(token_id),)).fetchone()
        return self._public_row(row)

    def rotate_token(self, token_id: int) -> dict[str, Any]:
        token = generate_token()
        salt, token_hash = _hash_token(token)
        ts = now_ts()
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE tokens
                SET salt = ?, token_hash = ?, updated_at = ?, last_used_at = NULL, use_count = 0
                WHERE id = ?
                """,
                (salt, token_hash, ts, int(token_id)),
            )
            if cur.rowcount == 0:
                raise KeyError(f"token id {token_id} not found")
            row = conn.execute("SELECT * FROM tokens WHERE id = ?", (int(token_id),)).fetchone()
        result = self._public_row(row)
        result["token"] = token
        return result

    def delete_token(self, token_id: int) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM tokens WHERE id = ?", (int(token_id),))
        return cur.rowcount > 0

    def verify_bearer(self, token: str) -> dict[str, Any] | None:
        if not token:
            return None
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM tokens WHERE enabled = 1 ORDER BY id").fetchall()
            matched_id = None
            matched_row = None
            for row in rows:
                if _verify_token(token, row["salt"], row["token_hash"]):
                    matched_id = int(row["id"])
                    matched_row = row
                    break
            if matched_id is None:
                return None
            conn.execute(
                """
                UPDATE tokens
                SET last_used_at = ?, use_count = use_count + 1
                WHERE id = ?
                """,
                (now_ts(), matched_id),
            )
        return self._public_row(matched_row)

    @staticmethod
    def _public_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": int(row["id"]),
            "name": row["name"],
            "enabled": bool(row["enabled"]),
            "created_at": int(row["created_at"]),
            "updated_at": int(row["updated_at"]),
            "last_used_at": int(row["last_used_at"]) if row["last_used_at"] is not None else None,
            "use_count": int(row["use_count"]),
            "note": row["note"],
        }


def _validate_name(name: str) -> str:
    if not isinstance(name, str):
        raise ValueError("name must be a string")
    name = name.strip()
    if not name:
        raise ValueError("name must be non-empty")
    if len(name) > 100:
        raise ValueError("name must be at most 100 characters")
    return name
