"""四角色 token 与耗时归因。

两条刻意的设计约束：

1. **按上游的四个角色记账**（``extract`` / ``keyword`` / ``query`` / ``vlm``，与
   ``lightrag/llm_roles.py`` 的 ``ROLES`` 同名同序）。抽取阶段是整个 RAG 里最贵的一步，
   把它和查询阶段混在一张账上，就没法回答"钱花在哪"这个问题。
2. **不内置任何单价**。模型价格随时会变，写死的价格会产出一个"看起来很专业但已经错了"的成本数字，
   比没有成本数字更危险。缺配置时只报 token 与耗时，并明确说明为什么没有金额。
"""

from __future__ import annotations

import dataclasses
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

# 与上游 lightrag/llm_roles.py 的 ROLES 同名同序；tests 里有断言守着，防止上游改名后这里悄悄失配。
ROLES: tuple[str, ...] = ("extract", "keyword", "query", "vlm")

# 降级链的层级名，顺序与 chain 中从上到下的尝试顺序一致。
LAYERS: tuple[str, ...] = ("primary", "secondary", "local", "deterministic")


@dataclasses.dataclass(frozen=True)
class MeteringRecord:
    """一次模型调用的账目。

    ``tokens_in`` / ``tokens_out`` 允许为 ``None``：不是每个 provider 都回 usage，
    拿不到就写 ``None``，**不估算、不猜测**——估算出来的 token 数会让人误以为可以拿它算钱。
    """

    role: str
    layer: str
    ok: bool
    latency_ms: float
    tokens_in: int | None = None
    tokens_out: int | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class MeteringLedger:
    """线程安全的记账本。

    ``max_records`` 是硬上限：这是常驻演示服务，账目明细不能无限长，
    否则挂一周之后进程内存被自己的日志吃掉。超过上限只保留聚合值。
    """

    def __init__(
        self,
        *,
        unit_prices: Mapping[str, tuple[float, float]] | None = None,
        max_records: int = 2000,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self._lock = threading.Lock()
        self._records: list[MeteringRecord] = []
        self._dropped = 0
        self._max_records = max_records
        self._unit_prices = dict(unit_prices) if unit_prices else None
        self._wall_clock = wall_clock
        self._started_at = wall_clock()
        self._totals: dict[str, dict[str, float]] = {}

    # ---- 写入 ----

    def record(
        self,
        *,
        role: str,
        layer: str,
        ok: bool = True,
        latency_ms: float | None = None,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        reason: str | None = None,
        started_at: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> MeteringRecord:
        if latency_ms is None:
            latency_ms = (clock() - started_at) * 1000.0 if started_at is not None else 0.0
        record = MeteringRecord(
            role=role,
            layer=layer,
            ok=ok,
            latency_ms=round(float(latency_ms), 3),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            reason=reason,
        )
        with self._lock:
            if len(self._records) >= self._max_records:
                self._records.pop(0)
                self._dropped += 1
            self._records.append(record)
            bucket = self._totals.setdefault(
                f"{role}/{layer}", {"calls": 0.0, "failures": 0.0, "latency_ms": 0.0, "tokens_in": 0.0, "tokens_out": 0.0}
            )
            bucket["calls"] += 1
            bucket["failures"] += 0 if ok else 1
            bucket["latency_ms"] += record.latency_ms
            bucket["tokens_in"] += tokens_in or 0
            bucket["tokens_out"] += tokens_out or 0
        return record

    # ---- 读取 ----

    def records(self) -> list[MeteringRecord]:
        with self._lock:
            return list(self._records)

    def snapshot(self) -> dict[str, Any]:
        """按 role/layer 聚合的快照。数字口径：``latency_ms`` 是累计值，``latency_avg_ms`` 才是均值。"""
        with self._lock:
            buckets = {k: dict(v) for k, v in self._totals.items()}
            dropped = self._dropped
        for bucket in buckets.values():
            calls = bucket["calls"] or 1
            bucket["calls"] = int(bucket["calls"])
            bucket["failures"] = int(bucket["failures"])
            bucket["latency_ms"] = round(bucket["latency_ms"], 3)
            bucket["latency_avg_ms"] = round(bucket["latency_ms"] / calls, 3)
            bucket["tokens_in"] = int(bucket["tokens_in"])
            bucket["tokens_out"] = int(bucket["tokens_out"])
        return {
            "by_role_layer": buckets,
            "dropped_records": dropped,
            "started_at": self._started_at,
        }

    def cost(self) -> dict[str, Any]:
        """有单价才出金额；没有就明说没有，而不是给个 0 让人误读成"不花钱"。"""
        if not self._unit_prices:
            return {
                "priced": False,
                "reason": "未配置模型单价：价格会变，内置一份会变成错的成本数字；请通过 unit_prices 显式注入",
            }
        snapshot = self.snapshot()
        total = 0.0
        per_role: dict[str, float] = {}
        for key, bucket in snapshot["by_role_layer"].items():
            price = self._unit_prices.get(key) or self._unit_prices.get(bucket and key.split("/")[0])
            if not price:
                continue
            amount = bucket["tokens_in"] / 1e6 * price[0] + bucket["tokens_out"] / 1e6 * price[1]
            per_role[key] = round(amount, 6)
            total += amount
        return {"priced": True, "unit": "per_1M_tokens", "by_role_layer": per_role, "total": round(total, 6)}

    def prometheus_text(self) -> str:
        """零依赖的 Prometheus 文本暴露（counter / gauge 两类）。"""
        snapshot = self.snapshot()
        lines = [
            "# HELP lightrag_enterprise_calls_total 模型调用次数（按角色与生效层级）",
            "# TYPE lightrag_enterprise_calls_total counter",
        ]
        for key, bucket in sorted(snapshot["by_role_layer"].items()):
            role, _, layer = key.partition("/")
            labels = f'role="{role}",layer="{layer}"'
            lines.append(f"lightrag_enterprise_calls_total{{{labels}}} {bucket['calls']}")
            lines.append(f"lightrag_enterprise_failures_total{{{labels}}} {bucket['failures']}")
            lines.append(f"lightrag_enterprise_tokens_in_total{{{labels}}} {bucket['tokens_in']}")
            lines.append(f"lightrag_enterprise_tokens_out_total{{{labels}}} {bucket['tokens_out']}")
            lines.append(f"lightrag_enterprise_latency_ms_total{{{labels}}} {bucket['latency_ms']}")
        lines.append("# TYPE lightrag_enterprise_dropped_records gauge")
        lines.append(f"lightrag_enterprise_dropped_records {snapshot['dropped_records']}")
        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        with self._lock:
            self._records.clear()
            self._totals.clear()
            self._dropped = 0

    def track(self, *, role: str, layer: str) -> track:
        """``with ledger.track(role=..., layer=...) as timing:`` 的入口。"""
        return track(self, role=role, layer=layer)


class track:
    """``with track() as t: ...`` 语法糖，省掉每处手写 monotonic 的样板。

    用法::

        with ledger.track(role="query", layer="primary") as timing:
            text = await call_model(...)
        timing.settle(tokens_in=..., tokens_out=...)
    """

    def __init__(self, ledger: MeteringLedger, *, role: str, layer: str) -> None:
        self._ledger = ledger
        self._role = role
        self._layer = layer
        self._start = 0.0
        self._settled = False

    def __enter__(self) -> track:
        self._start = time.monotonic()
        return self

    def settle(
        self,
        *,
        ok: bool = True,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        reason: str | None = None,
    ) -> MeteringRecord:
        self._settled = True
        return self._ledger.record(
            role=self._role,
            layer=self._layer,
            ok=ok,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            reason=reason,
            started_at=self._start,
        )

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if not self._settled:
            self.settle(ok=exc_type is None, reason=None if exc_type is None else exc_type.__name__)
        return False  # 不吞异常：账要记，错也要往上抛
