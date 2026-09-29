"""暴露面测试：这一组的每个用例都对应一种"看起来配好了其实在裸奔"的配置。"""

from __future__ import annotations

import pytest

from enterprise.auth import (
    UPSTREAM_DEFAULT_TOKEN_SECRET,
    BearerGuard,
    DailyBudget,
    ExposureError,
    Severity,
    TokenBucketRateLimiter,
    audit_exposure,
    enforce_exposure,
    is_loopback,
    path_is_whitelisted,
)


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("127.0.0.1", True),
        ("::1", True),
        ("localhost", True),
        ("0.0.0.0", False),  # 最常见的"误以为是本地"的写法
        ("::", False),
        ("10.0.0.7", False),
        ("example.com", False),
        ("", False),
        (None, False),
    ],
)
def test_is_loopback(host: str | None, expected: bool) -> None:
    assert is_loopback(host) is expected


def test_public_host_without_api_key_is_a_blocking_error() -> None:
    findings = audit_exposure(host="0.0.0.0", api_key=None, whitelist_paths="/health")
    codes = [finding.code for finding in findings]
    assert "public_without_api_key" in codes
    assert all(finding.severity is Severity.ERROR for finding in findings)
    with pytest.raises(ExposureError):
        enforce_exposure(host="0.0.0.0", api_key=None, whitelist_paths="/health")


def test_upstream_default_whitelist_is_rejected_on_public_host() -> None:
    """复现上游默认配置 + 公网绑定的组合：这才是真正会出事的那个默认值。"""
    findings = audit_exposure(
        host="0.0.0.0",
        api_key="x" * 40,
        whitelist_paths="/health,/api/*",  # 上游 WHITELIST_PATHS 的默认值
    )
    assert [f.code for f in findings] == ["ollama_paths_whitelisted_on_public_host"]


def test_loopback_host_passes_even_with_upstream_defaults() -> None:
    assert audit_exposure(host="127.0.0.1", api_key=None, whitelist_paths="/health,/api/*") == []


def test_default_token_secret_is_rejected_when_accounts_are_configured() -> None:
    findings = audit_exposure(
        host="127.0.0.1",
        api_key="x" * 40,
        whitelist_paths="/health",
        auth_accounts="admin:hash",
        token_secret=UPSTREAM_DEFAULT_TOKEN_SECRET,
    )
    assert [f.code for f in findings] == ["default_token_secret"]


def test_short_api_key_warns_but_does_not_block() -> None:
    findings = audit_exposure(host="0.0.0.0", api_key="short", whitelist_paths="/health")
    assert [f.code for f in findings] == ["short_api_key"]
    assert findings[0].severity is Severity.WARNING
    enforce_exposure(host="0.0.0.0", api_key="short", whitelist_paths="/health")  # 不抛异常


def test_wildcard_cors_is_flagged_on_public_host() -> None:
    codes = [
        f.code
        for f in audit_exposure(host="0.0.0.0", api_key="y" * 40, whitelist_paths="/health", cors_origins="*")
    ]
    assert "wildcard_cors" in codes


def test_path_matching_supports_prefix_pattern() -> None:
    assert path_is_whitelisted("/health", ["/health", "/api/*"])
    assert path_is_whitelisted("/api/tags", ["/api/*"])
    assert not path_is_whitelisted("/documents", ["/health", "/api/*"])


# --- 鉴权 ---


def _payload(path: str, **headers: str) -> dict[str, object]:
    return {"path": path, "headers": headers}


def test_bearer_guard_blocks_missing_credential() -> None:
    guard = BearerGuard("k" * 32)
    assert guard.check(_payload("/v1/query")).rejected
    assert guard.check(_payload("/v1/query", authorization="Bearer wrong")).rejected


def test_bearer_guard_accepts_bearer_and_api_key_header() -> None:
    guard = BearerGuard("k" * 32)
    assert guard.check(_payload("/v1/query", authorization="Bearer " + "k" * 32)).allowed
    assert guard.check(_payload("/v1/query", **{"x-api-key": "k" * 32})).allowed
    assert guard.check(_payload("/v1/query", Authorization="bearer " + "k" * 32)).allowed


def test_bearer_guard_whitelists_only_health_by_default() -> None:
    guard = BearerGuard("k" * 32)
    assert guard.check(_payload("/health")).allowed
    assert guard.check(_payload("/api/tags")).rejected  # 上游默认放行的那条，这里必须拦住


def test_unset_key_refuses_anonymous_by_default() -> None:
    assert BearerGuard(None).check(_payload("/v1/query")).rejected
    assert BearerGuard(None, allow_anonymous_when_unset=True).check(_payload("/v1/query")).allowed


# --- 限流与预算 ---


def test_rate_limiter_allows_burst_then_blocks() -> None:
    now = [0.0]
    limiter = TokenBucketRateLimiter(rate_per_minute=60, burst=3, clock=lambda: now[0])
    assert [limiter.allow("c1") for _ in range(4)] == [True, True, True, False]
    now[0] += 1.0  # 60/min = 每秒 1 个
    assert limiter.allow("c1") is True
    assert limiter.allow("c2") is True  # 按客户端分桶，互不影响


def test_rate_limiter_disabled_when_rate_is_zero() -> None:
    limiter = TokenBucketRateLimiter(rate_per_minute=0, burst=1)
    assert all(limiter.allow("c") for _ in range(100))


def test_daily_budget_blocks_after_limit_and_rolls_over() -> None:
    clock = [0.0]
    budget = DailyBudget(daily_calls=2, clock=lambda: clock[0])
    assert budget.consume() is True
    assert budget.consume() is True
    assert budget.consume() is False
    assert budget.snapshot()["remaining"] == 0
    clock[0] += 86400
    assert budget.consume() is True
    assert budget.snapshot()["used"] == 1


def test_daily_budget_disabled_by_default() -> None:
    budget = DailyBudget(daily_calls=0)
    assert budget.enabled is False
    assert all(budget.consume() for _ in range(50))
