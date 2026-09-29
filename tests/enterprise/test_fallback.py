"""降级链测试。重点是把"可预期失败不重试"这条行为差异钉死——它是最容易被改回去的一条。"""

from __future__ import annotations

import pytest

from enterprise.contracts import Reason
from enterprise.fallback import (
    FallbackOutcome,
    LayerSpec,
    LayeredFallback,
    NoLayerAvailableError,
    classify_failure,
)
from enterprise.metering import MeteringLedger
from enterprise.offline import DeterministicExtractive

class Boom(Exception):
    pass


def flaky(failures: list[object], *, calls: list[int] | None = None):
    """第 N 次调用抛出 ``failures[N]``；元素为 None 表示成功。"""

    async def _call(prompt: str, **kwargs: object) -> str:
        if calls is not None:
            calls.append(1)
        index = len(calls) - 1 if calls is not None else 0
        outcome = failures[index] if index < len(failures) else None
        if outcome is not None:
            raise outcome
        return f"ok:{prompt}"

    return _call


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Error 402: insufficient balance", Reason.EXPECTED_PROVIDER_FAILURE),
        ("余额不足，请充值", Reason.EXPECTED_PROVIDER_FAILURE),
        ("Content policy violation detected", Reason.EXPECTED_PROVIDER_FAILURE),
        ("Request timed out after 30s", Reason.RETRYABLE_PROVIDER_FAILURE),
        ("Connection reset by peer", Reason.RETRYABLE_PROVIDER_FAILURE),
        ("something nobody has seen before", Reason.RETRYABLE_PROVIDER_FAILURE),
    ],
)
def test_classify_failure(message: str, expected: Reason) -> None:
    assert classify_failure(Boom(message)) is expected


async def test_primary_layer_used_when_healthy() -> None:
    chain = LayeredFallback([LayerSpec("primary", flaky([])), LayerSpec("deterministic", lambda p, **k: "tail")])
    outcome = await chain.generate("q")
    assert outcome.text == "ok:q"
    assert outcome.layer == "primary"
    assert outcome.degraded is False
    assert outcome.attempts == 1


async def test_expected_failure_does_not_retry() -> None:
    calls: list[int] = []
    chain = LayeredFallback(
        [LayerSpec("primary", flaky([Boom("insufficient balance")], calls=calls)), LayerSpec("deterministic", lambda p, **k: "tail")],
        retries_per_layer=3,
    )
    outcome = await chain.generate("q")
    assert len(calls) == 1, "余额不足是可预期失败，重试只会再烧一次额度"
    assert outcome.layer == "deterministic"
    assert outcome.degraded is True


async def test_retryable_failure_retries_within_layer() -> None:
    calls: list[int] = []
    chain = LayeredFallback(
        [LayerSpec("primary", flaky([Boom("timeout"), None], calls=calls)), LayerSpec("deterministic", lambda p, **k: "tail")],
        retries_per_layer=3,
    )
    outcome = await chain.generate("q")
    assert len(calls) == 2
    assert outcome.layer == "primary"
    assert outcome.attempts == 2


async def test_unconfigured_layer_is_skipped_without_attempt() -> None:
    chain = LayeredFallback([LayerSpec("primary", None, note="未配置 Key"), LayerSpec("deterministic", lambda p, **k: "tail")])
    outcome = await chain.generate("q")
    assert outcome.layer == "deterministic"
    assert outcome.attempts == 1
    assert chain.describe()[0]["configured"] is False


async def test_all_layers_failing_raises_with_reason_chain() -> None:
    chain = LayeredFallback([LayerSpec("primary", flaky([Boom("insufficient balance")]))], retries_per_layer=0)
    with pytest.raises(NoLayerAvailableError) as excinfo:
        await chain.generate("q")
    assert "primary" in str(excinfo.value)
    assert excinfo.value.failures[0][1].startswith("expected_provider_failure")


async def test_deterministic_tail_works_without_any_model() -> None:
    """断网、无 Key、无余额时的最后一道保障：链路必须还能产出带引用的回答。"""
    context = [
        {"reference_id": "c1", "file_path": "02.md", "content": "图谱里的噪声是全局的。一条没有来源的关系会被后续所有多跳查询引用。"},
        {"reference_id": "c2", "file_path": "06.md", "content": "审计日志只保留参数指纹，不保留参数原文。"},
    ]
    chain = LayeredFallback(
        [LayerSpec("primary", None), LayerSpec("deterministic", DeterministicExtractive())],
    )
    outcome = await chain.generate("没有来源的关系为什么会被多跳查询引用", context=context)
    assert outcome.layer == "deterministic"
    assert "无模型模式" in outcome.text
    assert "多跳查询" in outcome.text
    assert "[c1]" in outcome.text
    assert "[c2]" not in outcome.text, "摘录器只应摘与问题有词元重叠的句子所在片段"


async def test_deterministic_tail_refuses_when_context_empty() -> None:
    chain = LayeredFallback([LayerSpec("deterministic", DeterministicExtractive())])
    outcome = await chain.generate("任何问题", context=[])
    assert "不作答" in outcome.text


async def test_metering_records_every_attempt() -> None:
    ledger = MeteringLedger()
    calls: list[int] = []
    chain = LayeredFallback(
        [LayerSpec("primary", flaky([Boom("timeout"), None], calls=calls)), LayerSpec("deterministic", lambda p, **k: "tail")],
        metering=ledger,
        retries_per_layer=3,
        backoff_seconds=0.0,
    )
    outcome = await chain.generate("q")
    snapshot = ledger.snapshot()
    assert outcome.layer == "primary"
    assert snapshot["by_role_layer"]["query/primary"]["calls"] == 2
    assert snapshot["by_role_layer"]["query/primary"]["failures"] == 1


async def test_empty_layer_list_rejected() -> None:
    with pytest.raises(ValueError):
        LayeredFallback([])


async def test_outcome_serialises() -> None:
    outcome = FallbackOutcome(text="x", layer="deterministic", attempts=2, latency_ms=1.5, degraded=True)
    assert outcome.to_dict()["reason"] == "ok"
    assert outcome.to_dict()["degraded"] is True
