#!/usr/bin/env python3
"""Per-client rate limiting and cumulative extraction quota for the MCP HTTP server.

The tool surface is read-only and deterministic, so per-call output caps alone
cannot bound how much of the knowledge base a client walks away with — it just
calls again with different parameters. This module adds the missing dimension:
a budget that accumulates across calls and is charged in bytes actually
returned, which is the quantity "don't let anyone clone the wiki" is really
about.

Two independent limits, both sliding-window:

  - requests per minute — burst control
  - response bytes per hour — the extraction ceiling

Identity is supplied by the caller (token name, or ``anon:<ip>`` when auth is
off), so quota follows the credential rather than the connection.
"""

from __future__ import annotations

import threading
import time
from collections import deque

DEFAULT_RPM = 60
DEFAULT_QUOTA_BYTES = 5_000_000
DEFAULT_QUOTA_WINDOW = 3600


class RateLimiter:
    """Thread-safe sliding-window limiter with a cumulative byte quota."""

    def __init__(self, rpm: int = DEFAULT_RPM,
                 quota_bytes: int = DEFAULT_QUOTA_BYTES,
                 quota_window: int = DEFAULT_QUOTA_WINDOW) -> None:
        self.rpm = max(0, int(rpm))
        self.quota_bytes = max(0, int(quota_bytes))
        self.quota_window = max(1, int(quota_window))
        self._lock = threading.Lock()
        self._requests: dict[str, deque[float]] = {}
        self._spend: dict[str, deque[tuple[float, int]]] = {}

    @staticmethod
    def _evict(window: deque, cutoff: float, keyed: bool = False) -> None:
        while window and (window[0][0] if keyed else window[0]) < cutoff:
            window.popleft()

    def check(self, identity: str, *, enforce_quota: bool = True) -> tuple[bool, int, str]:
        """Return (allowed, retry_after_seconds, reason).

        Charges one request against the per-minute budget when allowed. Pass
        ``enforce_quota=False`` for the pre-authentication check, where only
        the request rate is meaningful because no identity is established yet.
        """
        now = time.monotonic()
        with self._lock:
            if self.rpm:
                reqs = self._requests.setdefault(identity, deque())
                self._evict(reqs, now - 60.0)
                if len(reqs) >= self.rpm:
                    retry = max(1, int(60.0 - (now - reqs[0])) + 1)
                    return False, retry, (
                        f"rate limit exceeded: {self.rpm} requests/minute"
                    )

            if enforce_quota and self.quota_bytes:
                spend = self._spend.setdefault(identity, deque())
                self._evict(spend, now - self.quota_window, keyed=True)
                used = sum(n for _, n in spend)
                if used >= self.quota_bytes:
                    retry = max(1, int(self.quota_window - (now - spend[0][0])) + 1)
                    return False, retry, (
                        f"extraction quota exhausted: {used:,}/{self.quota_bytes:,} "
                        f"bytes in the last {self.quota_window}s"
                    )

            if self.rpm:
                self._requests[identity].append(now)
        return True, 0, ""

    def charge(self, identity: str, nbytes: int) -> None:
        """Charge response bytes against the cumulative quota."""
        if not self.quota_bytes or nbytes <= 0:
            return
        now = time.monotonic()
        with self._lock:
            spend = self._spend.setdefault(identity, deque())
            spend.append((now, int(nbytes)))
            self._evict(spend, now - self.quota_window, keyed=True)

    def usage(self, identity: str) -> dict[str, int]:
        """Current window usage for *identity* (for audit lines)."""
        now = time.monotonic()
        with self._lock:
            reqs = self._requests.get(identity)
            spend = self._spend.get(identity)
            if reqs is not None:
                self._evict(reqs, now - 60.0)
            if spend is not None:
                self._evict(spend, now - self.quota_window, keyed=True)
            return {
                "requests_last_min": len(reqs) if reqs else 0,
                "bytes_in_window": sum(n for _, n in spend) if spend else 0,
                "quota_bytes": self.quota_bytes,
            }
