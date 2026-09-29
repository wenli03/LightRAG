"""计量测试：口径是"只报确定的事实"，不报猜出来的 token，也不报写死的钱。"""

from __future__ import annotations

from enterprise.metering import LAYERS, ROLES, MeteringLedger, track


def test_roles_and_layers_match_upstream_and_chain() -> None:
    import pytest

    roles_module = pytest.importorskip("lightrag.llm_roles")

    # 上游 ROLES 是 RoleSpec 对象而不是字符串：这条断言写错过一次，被这条测试当场抓住。
    # 它要守的不变量是「角色名一致」——名字一变，账目会静默记到不存在的角色上，指标看起来正常但全是错的。
    upstream_names = tuple(getattr(spec, "name", spec) for spec in roles_module.ROLES)
    assert upstream_names == ROLES
    assert LAYERS == ("primary", "secondary", "local", "deterministic")


def test_snapshot_aggregates_by_role_and_layer() -> None:
    ledger = MeteringLedger()
    ledger.record(role="extract", layer="primary", tokens_in=1000, tokens_out=200, latency_ms=900.0)
    ledger.record(role="query", layer="primary", tokens_in=300, tokens_out=40, latency_ms=100.0)
    ledger.record(role="query", layer="deterministic", ok=False, latency_ms=5.0, reason="no_evidence")

    snapshot = ledger.snapshot()["by_role_layer"]
    assert snapshot["extract/primary"]["calls"] == 1
    assert snapshot["query/primary"]["tokens_in"] == 300
    assert snapshot["query/deterministic"]["failures"] == 1
    assert snapshot["query/deterministic"]["latency_avg_ms"] == 5.0


def test_missing_token_usage_is_recorded_as_none_not_guessed() -> None:
    ledger = MeteringLedger()
    record = ledger.record(role="query", layer="local", latency_ms=12.0)
    assert record.tokens_in is None
    assert ledger.snapshot()["by_role_layer"]["query/local"]["tokens_in"] == 0


def test_cost_requires_explicit_unit_prices() -> None:
    ledger = MeteringLedger()
    ledger.record(role="query", layer="primary", tokens_in=1_000_000, tokens_out=0, latency_ms=1.0)
    cost = ledger.cost()
    assert cost["priced"] is False
    assert "未配置" in cost["reason"], "没有单价时必须明说没有，而不是给一个 0"


def test_cost_computes_when_prices_are_injected() -> None:
    ledger = MeteringLedger(unit_prices={"query/primary": (2.0, 8.0)})
    ledger.record(role="query", layer="primary", tokens_in=1_000_000, tokens_out=500_000, latency_ms=1.0)
    cost = ledger.cost()
    assert cost["priced"] is True
    assert cost["by_role_layer"]["query/primary"] == 6.0


def test_prometheus_text_exposes_counters() -> None:
    ledger = MeteringLedger()
    ledger.record(role="query", layer="primary", tokens_in=10, tokens_out=2, latency_ms=3.0)
    text = ledger.prometheus_text()
    assert 'lightrag_enterprise_calls_total{role="query",layer="primary"} 1' in text
    assert "lightrag_enterprise_tokens_out_total" in text
    assert text.endswith("\n")


def test_ledger_is_bounded_and_reports_drops() -> None:
    ledger = MeteringLedger(max_records=3)
    for _ in range(5):
        ledger.record(role="query", layer="primary", latency_ms=1.0)
    snapshot = ledger.snapshot()
    assert len(ledger.records()) == 3
    assert snapshot["dropped_records"] == 2
    assert snapshot["by_role_layer"]["query/primary"]["calls"] == 5, "聚合值不能因为明细被截断而失真"


def test_track_context_manager_settles_once_and_does_not_swallow() -> None:
    ledger = MeteringLedger()
    with ledger.track(role="query", layer="primary") as timing:
        timing.settle(tokens_in=5)
    assert ledger.snapshot()["by_role_layer"]["query/primary"]["calls"] == 1

    try:
        with ledger.track(role="extract", layer="primary"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert ledger.snapshot()["by_role_layer"]["extract/primary"]["failures"] == 1


def test_reset_clears_everything() -> None:
    ledger = MeteringLedger()
    ledger.record(role="query", layer="primary", latency_ms=1.0)
    ledger.reset()
    assert ledger.snapshot()["by_role_layer"] == {}


def test_record_type_alias_available() -> None:
    assert track is not None
