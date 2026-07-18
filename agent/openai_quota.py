"""Fail-closed daily token guard for OpenAI shared-traffic usage.

This guard is intentionally separate from pricing/accounting: Hermes' normal
usage accounting is post-response, while this ledger must reserve an upper
bound before a request is sent. It only applies to direct ``openai-api``
routes. Codex OAuth and other providers are never counted here.
"""
from __future__ import annotations

import math
import os
import sqlite3
import time
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from agent.model_metadata import estimate_messages_tokens_rough

PREMIUM_MODELS = frozenset({
    "gpt-5.4", "gpt-5.2", "gpt-5.1", "gpt-5.1-codex", "gpt-5",
    "gpt-5-codex", "gpt-5-chat-latest", "gpt-4.1", "gpt-4o", "o1", "o3",
})
MINI_MODELS = frozenset({
    "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.1-codex-mini", "gpt-5-mini",
    "gpt-5-nano", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o-mini", "o1-mini",
    "o3-mini", "o4-mini", "codex-mini-latest",
})
DEFAULT_LIMITS = {"premium": 250_000, "mini": 2_500_000}
_ACTIVE: ContextVar[Optional["Reservation"]] = ContextVar("openai_quota_reservation", default=None)


class OpenAIQuotaExceeded(RuntimeError):
    """Raised before an OpenAI API request when free capacity is unavailable."""

    status_code = 402


@dataclass(frozen=True)
class Reservation:
    ledger_path: str
    day: str
    bucket: str
    tokens: int
    reservation_id: str


def _base_model(model: Optional[str]) -> str:
    value = (model or "").strip().lower()
    if "/" in value:
        value = value.rsplit("/", 1)[-1]
    # OpenAI model snapshots retain the entitlement tier by base prefix.
    for candidate in sorted(PREMIUM_MODELS | MINI_MODELS, key=len, reverse=True):
        if value == candidate or value.startswith(candidate + "-"):
            return candidate
    return value


def quota_bucket(model: Optional[str]) -> Optional[str]:
    base = _base_model(model)
    if base in MINI_MODELS:
        return "mini"
    if base in PREMIUM_MODELS:
        return "premium"
    return None


def is_direct_openai(provider: str, base_url: str = "") -> bool:
    label = (provider or "").strip().lower()
    if label in {"openai-api", "openai", "openai_api"}:
        return True
    return "api.openai.com" in (base_url or "").lower()


def _profile_home() -> Path:
    try:
        from hermes_cli.config import get_hermes_home
        return Path(get_hermes_home())
    except Exception:
        return Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))


def _policy() -> dict[str, Any]:
    policy = {
        "enabled": True,
        "safety_margin": 0.10,
        "prompt_multiplier": 1.25,
        "fixed_overhead_tokens": 512,
        "premium_daily_tokens": DEFAULT_LIMITS["premium"],
        "mini_daily_tokens": DEFAULT_LIMITS["mini"],
    }
    try:
        from hermes_cli.config import load_config
        configured = load_config().get("openai_free_usage") or {}
        if isinstance(configured, dict):
            policy.update(configured)
    except Exception:
        pass
    return policy


def _limits() -> dict[str, int]:
    policy = _policy()
    margin = min(max(float(policy.get("safety_margin", 0.10)), 0.0), 0.50)
    return {
        "premium": max(0, math.floor(int(policy.get("premium_daily_tokens", 250_000)) * (1 - margin))),
        "mini": max(0, math.floor(int(policy.get("mini_daily_tokens", 2_500_000)) * (1 - margin))),
    }


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _usage_tokens(response: Any, provider: str = "openai-api") -> int:
    usage = getattr(response, "usage", None)
    if not usage:
        return 0
    try:
        from agent.usage_pricing import normalize_usage
        return max(0, int(normalize_usage(usage, provider=provider).total_tokens))
    except Exception:
        prompt = int(getattr(usage, "prompt_tokens", 0) or getattr(usage, "input_tokens", 0) or 0)
        output = int(getattr(usage, "completion_tokens", 0) or getattr(usage, "output_tokens", 0) or 0)
        return max(0, prompt + output)


class OpenAIQuotaLedger:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path or (_profile_home() / "openai_free_usage.sqlite3"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS daily_quota (
                day TEXT NOT NULL,
                bucket TEXT NOT NULL,
                committed_tokens INTEGER NOT NULL DEFAULT 0,
                reserved_tokens INTEGER NOT NULL DEFAULT 0,
                blocked INTEGER NOT NULL DEFAULT 0,
                updated_at REAL NOT NULL,
                PRIMARY KEY(day, bucket)
            )""")

    def reserve(self, model: Optional[str], messages: list, max_tokens: Optional[int], tools: Optional[list]) -> Reservation:
        if not _policy().get("enabled", True):
            raise OpenAIQuotaExceeded("OpenAI free-usage guard is disabled; refusing direct API traffic")
        if tools:
            raise OpenAIQuotaExceeded("OpenAI shared-traffic allowance is not used for tool calls")
        bucket = quota_bucket(model)
        if not bucket:
            raise OpenAIQuotaExceeded(f"OpenAI model {model!r} is not in the eligible free-usage allowlist")
        policy = _policy()
        prompt = estimate_messages_tokens_rough(messages or [])
        multiplier = max(1.0, float(policy.get("prompt_multiplier", 1.25)))
        overhead = max(0, int(policy.get("fixed_overhead_tokens", 512)))
        output_cap = max(1, int(max_tokens or 4096))
        estimate = max(1, math.ceil(prompt * multiplier) + output_cap + overhead)
        day = _today()
        limits = _limits()
        reservation_id = f"{os.getpid()}-{time.time_ns()}"
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            now = time.time()
            conn.execute("INSERT OR IGNORE INTO daily_quota(day,bucket,updated_at) VALUES(?,?,?)", (day, bucket, now))
            row = conn.execute("SELECT * FROM daily_quota WHERE day=? AND bucket=?", (day, bucket)).fetchone()
            if row["blocked"] or row["committed_tokens"] + row["reserved_tokens"] + estimate > limits[bucket]:
                conn.execute("COMMIT")
                raise OpenAIQuotaExceeded(
                    f"OpenAI free {bucket} quota reserved ({row['committed_tokens'] + row['reserved_tokens']:,}/"
                    f"{limits[bucket]:,} safe tokens); using non-OpenAI fallback"
                )
            conn.execute("UPDATE daily_quota SET reserved_tokens=reserved_tokens+?, updated_at=? WHERE day=? AND bucket=?", (estimate, now, day, bucket))
            conn.execute("COMMIT")
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            raise
        finally:
            conn.close()
        reservation = Reservation(str(self.path), day, bucket, estimate, reservation_id)
        _ACTIVE.set(reservation)
        return reservation

    def finalize(self, reservation: Reservation, response: Any) -> None:
        actual = _usage_tokens(response)
        # Missing usage is treated as the full reservation, never as zero.
        actual = actual or reservation.tokens
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT committed_tokens,reserved_tokens FROM daily_quota WHERE day=? AND bucket=?", (reservation.day, reservation.bucket)).fetchone()
            if row:
                committed = row["committed_tokens"] + actual
                reserved = max(0, row["reserved_tokens"] - reservation.tokens)
                blocked = 1 if committed > _limits()[reservation.bucket] else 0
                conn.execute("UPDATE daily_quota SET committed_tokens=?,reserved_tokens=?,blocked=?,updated_at=? WHERE day=? AND bucket=?", (committed, reserved, blocked, time.time(), reservation.day, reservation.bucket))
            conn.execute("COMMIT")
        finally:
            conn.close()
        if _ACTIVE.get() == reservation:
            _ACTIVE.set(None)

    def cancel_current(self) -> None:
        reservation = _ACTIVE.get()
        if not reservation:
            return
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("UPDATE daily_quota SET reserved_tokens=max(0,reserved_tokens-?),updated_at=? WHERE day=? AND bucket=?", (reservation.tokens, time.time(), reservation.day, reservation.bucket))
            conn.execute("COMMIT")
        finally:
            conn.close()
        _ACTIVE.set(None)

    def finalize_current(self, response: Any) -> None:
        reservation = _ACTIVE.get()
        if reservation:
            self.finalize(reservation, response)

    def snapshot(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT day,bucket,committed_tokens,reserved_tokens,blocked,updated_at FROM daily_quota ORDER BY day DESC,bucket").fetchall()
            return [dict(row) for row in rows]


_LEDGER: Optional[OpenAIQuotaLedger] = None

def ledger() -> OpenAIQuotaLedger:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = OpenAIQuotaLedger()
    return _LEDGER


def preflight(provider: str, model: Optional[str], messages: list, max_tokens: Optional[int], tools: Optional[list], base_url: str = "") -> Optional[Reservation]:
    if not is_direct_openai(provider, base_url):
        return None
    return ledger().reserve(model, messages, max_tokens, tools)


def finalize_response(response: Any) -> None:
    ledger().finalize_current(response)


def cancel_current() -> None:
    ledger().cancel_current()


def quota_snapshot() -> list[dict[str, Any]]:
    return ledger().snapshot()
