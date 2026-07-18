from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.openai_quota import (
    OpenAIQuotaExceeded,
    OpenAIQuotaLedger,
    quota_bucket,
    preflight,
)


def test_model_buckets_match_shared_traffic_allowlist():
    assert quota_bucket("gpt-5.4") == "premium"
    assert quota_bucket("gpt-5.4-mini") == "mini"
    assert quota_bucket("gpt-5.4-mini-2026-01-01") == "mini"
    assert quota_bucket("gpt-3.5-turbo") is None


def test_reservation_reconciles_actual_usage(tmp_path: Path):
    ledger = OpenAIQuotaLedger(tmp_path / "quota.sqlite3")
    policy = {
        "enabled": True,
        "safety_margin": 0,
        "prompt_multiplier": 1,
        "fixed_overhead_tokens": 0,
        "premium_daily_tokens": 100,
        "mini_daily_tokens": 100,
    }
    with patch("agent.openai_quota._policy", return_value=policy):
        reservation = ledger.reserve("gpt-5.4-mini", [{"role": "user", "content": "x"}], 10, None)
        assert reservation.tokens > 10
        assert reservation.tokens < 25
        ledger.finalize(reservation, SimpleNamespace(usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2)))
    row = ledger.snapshot()[0]
    assert row["committed_tokens"] == 5
    assert row["reserved_tokens"] == 0


def test_concurrent_reservations_cannot_cross_limit(tmp_path: Path):
    ledger = OpenAIQuotaLedger(tmp_path / "quota.sqlite3")
    policy = {
        "enabled": True,
        "safety_margin": 0,
        "prompt_multiplier": 1,
        "fixed_overhead_tokens": 0,
        "premium_daily_tokens": 20,
        "mini_daily_tokens": 20,
    }

    def reserve_one():
        try:
            return ledger.reserve("gpt-5.4-mini", [{"role": "user", "content": "x"}], 10, None)
        except OpenAIQuotaExceeded:
            return None

    with patch("agent.openai_quota._policy", return_value=policy):
        with ThreadPoolExecutor(max_workers=2) as pool:
            reservations = list(pool.map(lambda _: reserve_one(), range(2)))
    assert sum(r is not None for r in reservations) == 1


def test_tools_and_unlisted_models_fail_closed(tmp_path: Path):
    ledger = OpenAIQuotaLedger(tmp_path / "quota.sqlite3")
    with patch("agent.openai_quota._policy", return_value={"enabled": True}):
        with pytest.raises(OpenAIQuotaExceeded):
            ledger.reserve("gpt-5.4-mini", [], 10, [{"type": "function"}])
        with pytest.raises(OpenAIQuotaExceeded):
            ledger.reserve("gpt-3.5-turbo", [], 10, None)


def test_preflight_ignores_codex_and_other_providers(tmp_path: Path):
    with patch("agent.openai_quota._LEDGER", OpenAIQuotaLedger(tmp_path / "quota.sqlite3")):
        assert preflight("openai-codex", "gpt-5.4", [], 10, None) is None
        assert preflight("openrouter", "gpt-5.4", [], 10, None) is None
