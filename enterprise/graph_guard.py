"""知识图谱写入护栏：**没有来源的实体与关系不允许进图**。

这一层存在的理由，是 GraphRAG 相对向量 RAG 多出来的那个风险面：
向量库里多一条噪声 chunk，最坏结果是召回差一点；而**图谱里多一条无源的关系，
会被后续所有多跳查询当成事实使用**，而且它看起来和真事实一模一样。
上游对"属性必须是可存储标量"有契约（``lightrag.utils.validate_graph_attributes``），
但没有"这条关系必须有出处"的契约——这个缺口就是我们补的东西。

实现方式刻意选择**包装存储对象**而不是改上游函数：

* 上游在 ``__post_init__`` 里把 ``rag.chunk_entity_relation_graph`` 建好（``lightrag.py:1868``），
  之后所有读写都走这个属性，所以外面套一层代理即可全覆盖（含 rebuild 路径）；
* 上游代码零改动 → rebase 不痛，diff 能逐行讲，面试官一眼分得清"哪是基座、哪是我的"。

代价是：护栏只看得到"写入请求"，看不到"上游为什么要写"。这是有意的取舍——
护栏不知道业务，只认证据；业务规则留在上层。
"""

from __future__ import annotations

import dataclasses
import json
import os
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .contracts import SOURCE_SEP, GuardResult, Reason, count_sources, deny, fingerprint

REJECT_REASONS = (
    Reason.EMPTY_ID,
    Reason.NO_SOURCE,
    Reason.TYPE_NOT_ALLOWED,
    Reason.EMPTY_DESCRIPTION,
    Reason.TOO_MANY_SOURCES,
)


@dataclasses.dataclass(frozen=True)
class GraphGuardPolicy:
    """护栏策略。

    ``on_violation`` 有两个值，这不是"开关"，而是**先观测后收口**的工作方式：

    * ``record``：只记录不拦。用来先跑一遍真实语料，把上游实际抽出来的实体类型
      统计出来（``observed_entity_types``），据此**经验性地**定受控词表；
    * ``reject``：拦下来。词表定完之后的正式模式。

    直接把词表拍脑袋定死再开 ``reject``，结果会是"整批抽取被拒、索引空掉"，
    而日志上看起来一切正常——这正是最危险的那类静默失败。
    """

    require_source_id: bool = True
    require_description: bool = True
    allowed_entity_types: frozenset[str] | None = None
    max_sources: int = 64
    on_violation: str = "reject"
    validate_attributes: bool = False

    def __post_init__(self) -> None:
        if self.on_violation not in ("reject", "record"):
            raise ValueError("on_violation 只能是 'reject' 或 'record'")


class GraphGuardStorage:
    """对任意"图存储对象"的透明代理，只在写入路径上加校验。

    只依赖鸭子类型（``upsert_node`` / ``upsert_edge`` / 批量变体 + 其余属性透传），
    所以**不需要 import lightrag**——单测可以用一个 20 行的假存储跑完全部护栏逻辑。
    """

    def __init__(
        self,
        delegate: Any,
        *,
        policy: GraphGuardPolicy | None = None,
        rejects_path: str | os.PathLike[str] | None = None,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self._delegate = delegate
        self._policy = policy or GraphGuardPolicy()
        self._rejects_path = os.fspath(rejects_path) if rejects_path else None
        self._wall_clock = wall_clock
        self._lock = threading.Lock()
        self._stats: dict[str, Any] = {
            "node_attempts": 0,
            "node_rejected": 0,
            "edge_attempts": 0,
            "edge_rejected": 0,
            "rejected_by_reason": {},
            "observed_entity_types": {},
            "rejected_samples": [],
        }
        if self._rejects_path:
            directory = os.path.dirname(os.path.abspath(self._rejects_path))
            if directory:
                os.makedirs(directory, exist_ok=True)

    # ---------- 透明代理 ----------

    @property
    def delegate(self) -> Any:
        return self._delegate

    @property
    def policy(self) -> GraphGuardPolicy:
        return self._policy

    def __getattr__(self, item: str) -> Any:
        # 只在下划线私有属性缺失时才会走到这里；显式排除以避免递归。
        if item.startswith("_"):
            raise AttributeError(item)
        return getattr(self._delegate, item)

    # ---------- 校验 ----------

    def check_node(self, node_id: str, node_data: Mapping[str, Any]) -> GuardResult:
        policy = self._policy
        if not str(node_id or "").strip():
            return deny(Reason.EMPTY_ID, kind="node")
        sources = count_sources(node_data.get("source_id"), sep=SOURCE_SEP)
        if policy.require_source_id and sources == 0:
            return deny(Reason.NO_SOURCE, kind="node", node_id=str(node_id), source_raw=str(node_data.get("source_id"))[:120])
        if sources > policy.max_sources:
            return deny(Reason.TOO_MANY_SOURCES, kind="node", node_id=str(node_id), sources=sources, max_sources=policy.max_sources)
        if policy.require_description and not str(node_data.get("description") or "").strip():
            return deny(Reason.EMPTY_DESCRIPTION, kind="node", node_id=str(node_id))
        raw_type = str(node_data.get("entity_type") or "").strip()
        if policy.allowed_entity_types is not None and raw_type not in policy.allowed_entity_types:
            return deny(
                Reason.TYPE_NOT_ALLOWED,
                kind="node",
                node_id=str(node_id),
                entity_type=raw_type,
                vocab_size=len(policy.allowed_entity_types),
            )
        return GuardResult(allowed=True, detail={"sources": sources, "entity_type": raw_type})

    def check_edge(self, source_node_id: str, target_node_id: str, edge_data: Mapping[str, Any]) -> GuardResult:
        policy = self._policy
        if not str(source_node_id or "").strip() or not str(target_node_id or "").strip():
            return deny(Reason.EMPTY_ID, kind="edge")
        sources = count_sources(edge_data.get("source_id"), sep=SOURCE_SEP)
        if policy.require_source_id and sources == 0:
            return deny(
                Reason.NO_SOURCE,
                kind="edge",
                src=str(source_node_id),
                tgt=str(target_node_id),
                source_raw=str(edge_data.get("source_id"))[:120],
            )
        if sources > policy.max_sources:
            return deny(Reason.TOO_MANY_SOURCES, kind="edge", src=str(source_node_id), tgt=str(target_node_id), sources=sources)
        if policy.require_description and not str(edge_data.get("description") or "").strip():
            return deny(Reason.EMPTY_DESCRIPTION, kind="edge", src=str(source_node_id), tgt=str(target_node_id))
        return GuardResult(allowed=True, detail={"sources": sources})

    def _maybe_validate_attributes(self, attributes: Mapping[str, Any]) -> str | None:
        """复用上游的统一校验点，而不是自己再实现一遍标量规则。

        上游 ``lightrag.utils.validate_graph_attributes``（``utils.py:8178``）的 docstring 明确写了
        "prefer it over re-deriving the rule per backend"，照做即可。这里是**懒 import**：
        没装 lightrag 的环境（跑单测）不会因为这一行而失败。
        """
        if not self._policy.validate_attributes:
            return None
        try:
            from lightrag.utils import validate_graph_attributes  # type: ignore
        except Exception:  # pragma: no cover - 未安装上游时静默跳过
            return None
        try:
            validate_graph_attributes(dict(attributes), context="enterprise.graph_guard")
        except Exception as exc:  # noqa: BLE001 - 上游抛什么就转述什么
            return f"{type(exc).__name__}: {exc}"
        return None

    # ---------- 写入路径 ----------

    def _admit(self, result: GuardResult) -> bool:
        """放行判定：``allowed`` 直接放行；被拒时先记账，再由模式决定是否仍然放行。

        ``record`` 模式必须**真的写进去**——它的用途是"先跑一遍真实语料、把上游实际抽出来的
        实体类型统计出来"，如果只记录不放行，就永远观测不到拦截模式下的真实分布，
        词表也就无从谈起。
        """
        if result.allowed:
            return True
        self._on_reject(result)
        return self._policy.on_violation == "record"

    async def upsert_node(self, node_id: str, node_data: Mapping[str, Any]) -> None:
        self._bump("node_attempts")
        self._observe_type(node_data.get("entity_type"))
        if not self._admit(self.check_node(node_id, node_data)):
            return
        attr_error = self._maybe_validate_attributes(node_data)
        if attr_error is not None and not self._admit(
            deny(Reason.TYPE_NOT_ALLOWED, kind="node", node_id=str(node_id), attribute_error=attr_error)
        ):
            return
        await self._delegate.upsert_node(node_id, node_data=dict(node_data))

    async def upsert_edge(self, source_node_id: str, target_node_id: str, edge_data: Mapping[str, Any]) -> None:
        self._bump("edge_attempts")
        if not self._admit(self.check_edge(source_node_id, target_node_id, edge_data)):
            return
        attr_error = self._maybe_validate_attributes(edge_data)
        if attr_error is not None and not self._admit(
            deny(Reason.TYPE_NOT_ALLOWED, kind="edge", src=str(source_node_id), tgt=str(target_node_id), attribute_error=attr_error)
        ):
            return
        await self._delegate.upsert_edge(source_node_id, target_node_id, edge_data=dict(edge_data))

    async def upsert_nodes_batch(self, nodes: Sequence[tuple[str, Mapping[str, Any]]]) -> None:
        """批量必须在这里过滤，不能直接透传。

        上游 ``BaseGraphStorage.upsert_nodes_batch`` 的默认实现是逐条调用 ``upsert_node``，
        但各后端**会覆盖它做原生批量写**——如果这里原样透传，护栏就被整批绕过了。
        """
        kept: list[tuple[str, dict[str, Any]]] = []
        for node_id, node_data in nodes:
            self._bump("node_attempts")
            self._observe_type(node_data.get("entity_type"))
            if self._admit(self.check_node(node_id, node_data)):
                kept.append((node_id, dict(node_data)))
        if kept:
            await self._delegate.upsert_nodes_batch(kept)

    async def upsert_edges_batch(self, edges: Sequence[tuple[str, str, Mapping[str, Any]]]) -> None:
        kept: list[tuple[str, str, dict[str, Any]]] = []
        for source_node_id, target_node_id, edge_data in edges:
            self._bump("edge_attempts")
            if self._admit(self.check_edge(source_node_id, target_node_id, edge_data)):
                kept.append((source_node_id, target_node_id, dict(edge_data)))
        if kept:
            await self._delegate.upsert_edges_batch(kept)

    # ---------- 统计与审计 ----------

    def _bump(self, key: str) -> None:
        with self._lock:
            self._stats[key] += 1

    def _observe_type(self, value: Any) -> None:
        name = str(value or "").strip() or "<missing>"
        with self._lock:
            observed = self._stats["observed_entity_types"]
            observed[name] = observed.get(name, 0) + 1

    def _on_reject(self, result: GuardResult) -> None:
        strict = self._policy.on_violation == "reject"
        kind = str(result.detail.get("kind", "unknown"))
        with self._lock:
            if strict:
                self._stats[f"{kind}_rejected"] = self._stats.get(f"{kind}_rejected", 0) + 1
            by_reason = self._stats["rejected_by_reason"]
            by_reason[result.reason.value] = by_reason.get(result.reason.value, 0) + 1
            samples = self._stats["rejected_samples"]
            if len(samples) < 20:
                samples.append({"kind": kind, "reason": result.reason.value, "detail": dict(result.detail)})
            record = {
                "ts": self._wall_clock(),
                "kind": kind,
                "reason": result.reason.value,
                "enforced": strict,
                "detail_fingerprint": fingerprint(result.detail),
                "detail_keys": sorted(result.detail.keys()),
            }
        self._append_reject(record)

    def _append_reject(self, record: Mapping[str, Any]) -> None:
        if not self._rejects_path:
            return
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with self._lock:
            with open(self._rejects_path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def stats(self) -> dict[str, Any]:
        with self._lock:
            stats = json.loads(json.dumps(self._stats))  # 深拷贝，避免调用方改到内部状态
        stats["mode"] = self._policy.on_violation
        stats["enforced"] = self._policy.on_violation == "reject"
        stats["require_source_id"] = self._policy.require_source_id
        stats["vocab_size"] = None if self._policy.allowed_entity_types is None else len(self._policy.allowed_entity_types)
        observed = stats.get("observed_entity_types", {})
        stats["observed_entity_types_top"] = sorted(observed.items(), key=lambda kv: (-kv[1], kv[0]))[:30]
        stats["rejected_rate"] = round(
            (
                stats.get("node_rejected", 0) + stats.get("edge_rejected", 0)
            )
            / max(1, stats.get("node_attempts", 0) + stats.get("edge_attempts", 0)),
            4,
        )
        return stats

    def derive_vocabulary(self, *, min_count: int = 1) -> list[str]:
        """从观测到的实体类型里导出候选受控词表（含 ``<missing>`` 供人工裁决）。"""
        with self._lock:
            observed = dict(self._stats["observed_entity_types"])
        return sorted(name for name, count in observed.items() if count >= min_count)


def install_graph_guard(
    rag: Any,
    *,
    policy: GraphGuardPolicy | None = None,
    rejects_path: str | os.PathLike[str] | None = None,
    attribute: str = "chunk_entity_relation_graph",
) -> GraphGuardStorage:
    """把护栏套到已构造好的 ``LightRAG`` 实例上，并且**幂等**。

    幂等很重要：演示服务重启、热重载、或在测试里反复调用时若套了两层，
    计数会被重复统计，指标就变成了假的。
    """
    current = getattr(rag, attribute, None)
    if current is None:
        raise RuntimeError(
            f"实例上没有 {attribute}；上游可能改了存储初始化位置，请对照 docs/adr/0001-extension-points.md 复核"
        )
    if isinstance(current, GraphGuardStorage):
        return current
    guard = GraphGuardStorage(current, policy=policy, rejects_path=rejects_path)
    setattr(rag, attribute, guard)
    return guard


@dataclasses.dataclass
class GraphGuardReport:
    """给面试/评审用的一页纸结论：拦了多少、为什么拦、词表长什么样。"""

    attempts: int
    rejected: int
    rejected_rate: float
    by_reason: dict[str, int]
    observed_types: list[tuple[str, int]]

    @classmethod
    def from_stats(cls, stats: Mapping[str, Any]) -> GraphGuardReport:
        return cls(
            attempts=int(stats.get("node_attempts", 0)) + int(stats.get("edge_attempts", 0)),
            rejected=int(stats.get("node_rejected", 0)) + int(stats.get("edge_rejected", 0)),
            rejected_rate=float(stats.get("rejected_rate", 0.0)),
            by_reason=dict(stats.get("rejected_by_reason", {})),
            observed_types=[tuple(item) for item in stats.get("observed_entity_types_top", [])],  # type: ignore[misc]
        )

    def render(self) -> str:
        lines = [
            f"图谱写入尝试 {self.attempts} 次，被护栏拒绝 {self.rejected} 次（拒绝率 {self.rejected_rate:.2%}）",
        ]
        if self.by_reason:
            lines.append("拒绝原因分布：" + "、".join(f"{k}={v}" for k, v in sorted(self.by_reason.items())))
        if self.observed_types:
            top = "、".join(f"{name}({count})" for name, count in self.observed_types[:10])
            lines.append(f"观测到的实体类型 Top10：{top}")
        return "\n".join(lines)
