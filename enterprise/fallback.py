"""四级降级链：主模型 → 备模型 → 本地模型 → 确定性检索直出。

链尾是**确定性**的，这是整套设计里最重要的一条：**演示不依赖余额、不依赖网络、
不依赖第三方的可用性**。任何一层的失败都只是"这一层不能用"，而不是"这个产品现在不能用"。

失败要分两类，而且分类**必须产生行为差异**，否则分类只是装饰：

* **可预期失败**（余额不足、内容拦截）：重试只会再烧一次时间和额度，所以**不重试**，直接下一层；
* **可重试失败**（超时、连接抖动、5xx）：层内退避重试有限的次数，再降层。

层内重试次数默认 1 次（不是 0 也不是 5）：0 会把一次网络抖动放大成一次降级，
5 会在真挂的时候把响应时间拖成 5 倍——两个都是"看起来没问题"的坏结果。
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from .contracts import Reason
from .metering import MeteringLedger

# 只做关键词匹配的判断，因此**必须承认它是启发式**：provider 的报错文案随时会改，
# 改了就退化成"全部按可重试处理"，不会变成"错误地停止降级"。这个方向的不对称是有意选的。
EXPECTED_FAILURE_MARKERS: tuple[str, ...] = (
    "insufficient balance",
    "insufficient_quota",
    "insufficient quota",
    "billing",
    "余额不足",
    "余额不够",
    "账户余额",
    "content policy",
    "content_filter",
    "content moderation",
    "content blocked",
    "safety system",
    "risk control",
    "风控",
    "违规",
)

RETRYABLE_MARKERS: tuple[str, ...] = (
    "timeout",
    "timed out",
    "connection",
    "temporarily unavailable",
    "rate limit",
    "too many requests",
    "429",
    "500",
    "502",
    "503",
    "504",
    "超时",
)


def classify_failure(exc: BaseException) -> Reason:
    message = f"{type(exc).__name__}: {exc}".lower()
    if any(marker in message for marker in EXPECTED_FAILURE_MARKERS):
        return Reason.EXPECTED_PROVIDER_FAILURE
    if any(marker in message for marker in RETRYABLE_MARKERS):
        return Reason.RETRYABLE_PROVIDER_FAILURE
    # 认不出来的一律当成"可重试"：宁可多花一次重试，也不要因为不认识报错而提前放弃一个其实能成的模型。
    return Reason.RETRYABLE_PROVIDER_FAILURE


@dataclasses.dataclass(frozen=True)
class LayerSpec:
    """一层降级。``func=None`` 表示这一层没配置（例如没给备模型 Key），会被直接跳过。"""

    name: str
    func: Callable[..., Any] | None
    role: str = "query"
    note: str = ""

    @property
    def configured(self) -> bool:
        return self.func is not None


@dataclasses.dataclass(frozen=True)
class FallbackOutcome:
    text: str
    layer: str
    attempts: int
    latency_ms: float
    degraded: bool
    reason: Reason = Reason.OK

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "attempts": self.attempts,
            "latency_ms": self.latency_ms,
            "degraded": self.degraded,
            "reason": self.reason.value,
        }


class NoLayerAvailableError(RuntimeError):
    """所有层都失败。带上每一层的失败原因，否则排障等于猜。"""

    def __init__(self, failures: Sequence[tuple[str, str]]) -> None:
        self.failures = list(failures)
        detail = "；".join(f"{name}: {reason}" for name, reason in self.failures) or "（没有配置任何层）"
        super().__init__(f"降级链全部失败 → {detail}")


class LayeredFallback:
    def __init__(
        self,
        layers: Sequence[LayerSpec],
        *,
        metering: MeteringLedger | None = None,
        retries_per_layer: int = 1,
        backoff_seconds: float = 0.4,
        sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if not layers:
            raise ValueError("降级链至少需要一层")
        self._layers = list(layers)
        self._metering = metering
        self._retries = max(0, retries_per_layer)
        self._backoff = max(0.0, backoff_seconds)
        self._sleep = sleeper

    @property
    def layers(self) -> list[LayerSpec]:
        return list(self._layers)

    def describe(self) -> list[dict[str, Any]]:
        return [
            {"name": layer.name, "role": layer.role, "configured": layer.configured, "note": layer.note}
            for layer in self._layers
        ]

    async def generate(
        self,
        prompt: str,
        *,
        role: str | None = None,
        context: Sequence[Mapping[str, Any]] | None = None,
        question: str | None = None,
    ) -> FallbackOutcome:
        failures: list[tuple[str, str]] = []
        attempts = 0
        started = time.monotonic()
        effective_role = role or self._layers[0].role

        for index, layer in enumerate(self._layers):
            if not layer.configured:
                failures.append((layer.name, "not_configured"))
                continue

            layer_started = time.monotonic()
            tries = 0
            while True:
                tries += 1
                attempts += 1
                call_started = time.monotonic()
                try:
                    produced = await _invoke(layer.func, prompt, role=layer.role, context=context, question=question)
                    text, usage = _split_usage(produced)
                    call_ms = (time.monotonic() - call_started) * 1000.0
                    if self._metering is not None:
                        self._metering.record(
                            role=layer.role,
                            layer=layer.name,
                            ok=True,
                            latency_ms=call_ms,
                            tokens_in=usage.get("tokens_in"),
                            tokens_out=usage.get("tokens_out"),
                            reason=None,
                        )
                    return FallbackOutcome(
                        text=text,
                        layer=layer.name,
                        attempts=attempts,
                        latency_ms=round((time.monotonic() - layer_started) * 1000.0, 3),
                        degraded=index > 0,
                        reason=Reason.OK,
                    )
                except Exception as exc:  # noqa: BLE001 - 降级链就是要吞掉所有异常
                    reason = classify_failure(exc)
                    failures.append((layer.name, f"{reason.value}: {type(exc).__name__}: {exc}"))
                    if self._metering is not None:
                        self._metering.record(
                            role=layer.role,
                            layer=layer.name,
                            ok=False,
                            latency_ms=(time.monotonic() - call_started) * 1000.0,
                            reason=reason.value,
                        )
                    expected = reason is Reason.EXPECTED_PROVIDER_FAILURE
                    if expected or tries > self._retries:
                        break
                    if self._backoff:
                        await self._sleep(self._backoff * tries)

        raise NoLayerAvailableError(failures)


async def _invoke(
    func: Callable[..., Any] | None,
    prompt: str,
    *,
    role: str,
    context: Sequence[Mapping[str, Any]] | None,
    question: str | None,
) -> Any:
    """调用一层。支持同步与异步函数，并按签名裁剪关键字参数。

    按签名裁剪是刻意的：确定性层需要 ``context``，远程模型层只需要 ``prompt``，
    如果强行统一传全部参数，"写一个只关心 prompt 的模型函数"就会变成一件要读文档的事。
    """
    if func is None:
        raise RuntimeError("layer func is None")
    kwargs: dict[str, Any] = {"role": role, "context": context, "question": question or prompt}
    params = _accepted_keywords(func)
    if params is not None:
        kwargs = {key: value for key, value in kwargs.items() if key in params}
    result = func(prompt, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


def _split_usage(produced: Any) -> tuple[str, Mapping[str, Any]]:
    """允许一层返回 ``"文本"`` 或 ``("文本", {"tokens_in":..., "tokens_out":...})``。

    后者是给远程模型层用的：provider 会把接口返回的真实用量带出来。
    之所以要求 provider 显式回传而不是在这里估算，是因为**估算出来的 token 数会让人误以为可以拿它算钱**，
    而账目一旦掺入估算值，就无法再区分"实测"和"猜的"。
    """
    if isinstance(produced, tuple) and len(produced) == 2 and isinstance(produced[1], Mapping):
        text = "" if produced[0] is None else str(produced[0])
        return text, dict(produced[1])
    return ("" if produced is None else str(produced)), {}


def _accepted_keywords(func: Callable[..., Any]) -> set[str] | None:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):  # pragma: no cover - 内建函数等
        return None
    params = signature.parameters
    if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return None  # **kwargs：全部传过去
    return {name for name, param in params.items() if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)}


__all__ = [
    "EXPECTED_FAILURE_MARKERS",
    "FallbackOutcome",
    "LayerSpec",
    "LayeredFallback",
    "NoLayerAvailableError",
    "classify_failure",
]
