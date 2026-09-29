"""治理层统一契约。

本模块只定义**数据形状与协议**，不含业务实现，也**不 import lightrag、不 import 任何第三方库**。
这样做的代价是多写几行协议，收益是：治理层可以脱离上游重依赖单独跑单测，
也就意味着"护栏有没有真的拦住"这件事可以在 CI 里被证伪，而不是靠人肉点页面。
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any, Protocol, runtime_checkable

# 与上游 lightrag/constants.py 的 GRAPH_FIELD_SEP 同值。这里不 import 上游是为了保持零依赖，
# 两处一旦漂移，tests/test_contracts.py::test_source_sep_matches_upstream 会直接失败。
SOURCE_SEP = "<SEP>"


class Reason(str, Enum):
    """拒绝 / 降级的原因码。

    必须是可枚举的稳定字符串：它同时进审计日志、Prometheus 指标和人工待办，
    自由文本会让下游没法聚合，也没法写断言。
    """

    OK = "ok"

    # —— 图谱写入护栏 ——
    EMPTY_ID = "empty_id"
    NO_SOURCE = "no_source"
    TYPE_NOT_ALLOWED = "type_not_allowed"
    EMPTY_DESCRIPTION = "empty_description"
    TOO_MANY_SOURCES = "too_many_sources"

    # —— 回答护栏 ——
    NO_EVIDENCE = "no_evidence"
    LOW_COVERAGE = "low_coverage"
    MISSING_CITATION = "missing_citation"

    # —— 暴露面 ——
    UNAUTHORIZED = "unauthorized"
    RATE_LIMITED = "rate_limited"
    BUDGET_EXCEEDED = "budget_exceeded"

    # —— 模型降级 ——
    EXPECTED_PROVIDER_FAILURE = "expected_provider_failure"
    RETRYABLE_PROVIDER_FAILURE = "retryable_provider_failure"
    NO_LAYER_AVAILABLE = "no_layer_available"

    def __str__(self) -> str:  # pragma: no cover - 仅为日志可读性
        return self.value


@dataclasses.dataclass(frozen=True)
class GuardResult:
    """护栏判定的唯一返回类型。

    ``detail`` 放**结构化证据**（命中条数、覆盖率、阈值、角色、耗时），
    而不是一句人话——因为它要被断言、被聚合、被画进看板。
    """

    allowed: bool
    reason: Reason = Reason.OK
    detail: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def rejected(self) -> bool:
        return not self.allowed

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason.value,
            "detail": dict(self.detail),
        }


ALLOW = GuardResult(allowed=True)


def deny(reason: Reason, **detail: Any) -> GuardResult:
    """构造一个"拒绝"结果。detail 直接展开成关键字参数，调用点更短。"""
    return GuardResult(allowed=False, reason=reason, detail=detail)


@runtime_checkable
class Guard(Protocol):
    """任何"先校验、后放行"的组件都实现它。

    统一成 ``check(payload) -> GuardResult`` 的好处是：鉴权、图谱护栏、证据门槛
    可以在同一个测试夹具里被同等对待，漏测哪一个一眼能看出来。
    """

    def check(self, payload: Mapping[str, Any]) -> GuardResult: ...


@runtime_checkable
class Metering(Protocol):
    def record(self, **fields: Any) -> Any: ...

    def snapshot(self) -> dict[str, Any]: ...


class ContractError(ValueError):
    """调用方违反契约（不是被护栏拒绝）。

    刻意与"被拒绝"区分开：调用方传错了形状应该炸掉，
    而被护栏拒绝是**正常业务路径**，必须返回 GuardResult 而不是抛异常。
    """


def require_mapping(payload: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ContractError(f"{name} 必须是映射，实际为 {type(payload).__name__}")
    return payload


def fingerprint(payload: Any) -> str:
    """对任意 JSON 可序列化对象取稳定指纹（16 位十六进制）。

    审计日志只留指纹与计数、**不留原文**：审计要能追责，但不能变成第二个泄密面
    ——工具参数里经常带着客户数据与内部单号。
    """
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def count_sources(value: Any, *, sep: str = SOURCE_SEP) -> int:
    """数一个 ``source_id`` 字段里到底挂了几处来源。

    上游把多个 chunk 标识用 ``<SEP>`` 拼成一个字符串（``operate.py`` 里
    ``GRAPH_FIELD_SEP.join(...)``），所以"有没有来源"不能只判非空，
    还要判**拆开之后是不是真有东西**——``"<SEP><SEP>"`` 是非空字符串，但零来源。
    """
    if value is None:
        return 0
    if isinstance(value, str):
        return len([part for part in value.split(sep) if part.strip()])
    if isinstance(value, Sequence):
        return len([part for part in value if str(part).strip()])
    return 0
