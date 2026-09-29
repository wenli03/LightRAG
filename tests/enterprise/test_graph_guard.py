"""图谱护栏测试。

核心断言只有一句：**没有来源的东西，一条都不许进图。**
包括三个容易漏掉的入口：批量写入、边（边比点更容易被忽略）、以及"看起来非空其实零来源"的字符串。
"""

from __future__ import annotations

import json

import pytest

from enterprise.contracts import Reason, count_sources
from enterprise.graph_guard import GraphGuardPolicy, GraphGuardStorage, GraphGuardReport, install_graph_guard

from .conftest import FakeGraphStorage, FakeRAG, edge_data, node_data


async def test_node_with_source_is_written(storage: FakeGraphStorage) -> None:
    guard = GraphGuardStorage(storage)
    await guard.upsert_node("供应商A", node_data())
    assert [item[0] for item in storage.nodes] == ["供应商A"]
    assert guard.stats()["node_rejected"] == 0


@pytest.mark.parametrize("bad_source", [None, "", "   ", "<SEP>", "<SEP><SEP>"])
async def test_node_without_effective_source_is_rejected(storage: FakeGraphStorage, bad_source: object) -> None:
    guard = GraphGuardStorage(storage)
    await guard.upsert_node("供应商A", node_data(source_id=bad_source))
    assert storage.nodes == []
    assert guard.stats()["node_rejected"] == 1
    assert guard.stats()["rejected_by_reason"][Reason.NO_SOURCE.value] == 1


async def test_edge_without_source_is_rejected(storage: FakeGraphStorage) -> None:
    guard = GraphGuardStorage(storage)
    await guard.upsert_edge("A", "B", edge_data(source_id=None))
    assert storage.edges == []
    assert guard.stats()["edge_rejected"] == 1


async def test_record_mode_observes_without_blocking(storage: FakeGraphStorage) -> None:
    """观测模式：写入照常发生，但拒绝原因被记账——这是定词表之前必须走的阶段。"""
    guard = GraphGuardStorage(storage, policy=GraphGuardPolicy(on_violation="record"))
    await guard.upsert_node("无源节点", node_data(source_id=None))
    assert len(storage.nodes) == 1
    assert guard.stats()["node_rejected"] == 0  # enforced=False，不计数为"已拦下"
    assert guard.stats()["rejected_by_reason"][Reason.NO_SOURCE.value] == 1
    assert guard.stats()["mode"] == "record"


async def test_batch_nodes_are_filtered_not_passed_through(storage: FakeGraphStorage) -> None:
    """负向用例：批量路径如果被原样透传，护栏就是摆设。"""
    guard = GraphGuardStorage(storage)
    await guard.upsert_nodes_batch([("good", node_data()), ("bad", node_data(source_id=None))])
    assert [item[0] for item in storage.batch_nodes] == ["good"]
    assert guard.stats()["node_attempts"] == 2
    assert guard.stats()["node_rejected"] == 1


async def test_batch_edges_are_filtered(storage: FakeGraphStorage) -> None:
    guard = GraphGuardStorage(storage)
    await guard.upsert_edges_batch([("A", "B", edge_data()), ("B", "C", edge_data(source_id=""))])
    assert len(storage.batch_edges) == 1
    assert guard.stats()["edge_rejected"] == 1


async def test_controlled_vocabulary_rejects_unknown_type(storage: FakeGraphStorage) -> None:
    guard = GraphGuardStorage(storage, policy=GraphGuardPolicy(allowed_entity_types=frozenset({"供应商"})))
    await guard.upsert_node("客户X", node_data(entity_type="客户", source_id="chunk-a"))
    await guard.upsert_node("供应商A", node_data(entity_type="供应商"))
    assert [item[0] for item in storage.nodes] == ["供应商A"]
    assert guard.stats()["rejected_by_reason"][Reason.TYPE_NOT_ALLOWED.value] == 1


async def test_empty_description_rejected_when_required(storage: FakeGraphStorage) -> None:
    guard = GraphGuardStorage(storage)
    await guard.upsert_node("A", node_data(description="   "))
    assert storage.nodes == []
    assert guard.stats()["rejected_by_reason"][Reason.EMPTY_DESCRIPTION.value] == 1


async def test_too_many_sources_rejected(storage: FakeGraphStorage) -> None:
    guard = GraphGuardStorage(storage, policy=GraphGuardPolicy(max_sources=2))
    await guard.upsert_node("A", node_data(source_id="<SEP>".join(f"chunk-{i}" for i in range(5))))
    assert storage.nodes == []
    assert guard.stats()["rejected_by_reason"][Reason.TOO_MANY_SOURCES.value] == 1


async def test_vocabulary_can_be_derived_from_observation(storage: FakeGraphStorage) -> None:
    guard = GraphGuardStorage(storage, policy=GraphGuardPolicy(on_violation="record"))
    for entity_type in ("供应商", "供应商", "物料", "工厂"):
        await guard.upsert_node(f"n-{entity_type}", node_data(entity_type=entity_type))
    vocab = guard.derive_vocabulary()
    assert set(vocab) == {"供应商", "物料", "工厂"}
    assert guard.stats()["observed_entity_types_top"][0] == ("供应商", 2)


async def test_rejects_log_keeps_fingerprint_not_content(storage: FakeGraphStorage, rejects_file: str) -> None:
    guard = GraphGuardStorage(storage, rejects_path=rejects_file)
    secret = "客户：某某集团 联系人 13800000000"
    await guard.upsert_edge("A", "B", edge_data(source_id=None, description=secret))
    with open(rejects_file, encoding="utf-8") as handle:
        line = json.loads(handle.readline())
    assert line["reason"] == Reason.NO_SOURCE.value
    assert "detail_fingerprint" in line
    assert secret not in json.dumps(line, ensure_ascii=False)


async def test_stats_report_renders_and_rejected_rate(storage: FakeGraphStorage) -> None:
    guard = GraphGuardStorage(storage)
    await guard.upsert_node("good", node_data())
    await guard.upsert_node("bad", node_data(source_id=None))
    report = GraphGuardReport.from_stats(guard.stats())
    assert report.attempts == 2
    assert report.rejected == 1
    assert report.rejected_rate == 0.5
    assert "拒绝率 50.00%" in report.render()


async def test_delegation_transparency(storage: FakeGraphStorage) -> None:
    guard = GraphGuardStorage(storage)
    assert guard.namespace == "fake"
    assert guard.something_else() == "delegated"


def test_install_is_idempotent(rag: FakeRAG) -> None:
    first = install_graph_guard(rag)
    second = install_graph_guard(rag)
    assert first is second, "重复安装会套两层代理，指标会被重复统计"


def test_install_reports_when_upstream_shape_changed() -> None:
    class Moved:
        pass

    with pytest.raises(RuntimeError, match="chunk_entity_relation_graph"):
        install_graph_guard(Moved())


def test_invalid_mode_rejected() -> None:
    with pytest.raises(ValueError):
        GraphGuardPolicy(on_violation="warn")


def test_count_sources_matches_guard_semantics() -> None:
    assert count_sources("<SEP>") == 0
    assert count_sources("chunk-1<SEP>chunk-2") == 2
