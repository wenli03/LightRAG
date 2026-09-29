"""治理层测试夹具。

刻意**不依赖上游 lightrag**：所有被测对象都只吃鸭子类型的入参，
所以这个测试套件可以在没装任何重依赖的机器上跑绿。
需要上游在场的断言（例如"角色名与上游一致"）单独标记为 ``integration``，
默认跳过——这样"护栏有没有拦住"这件事的验证不会被环境问题拖累。
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class FakeGraphStorage:
    """最小可用的假图存储：只记录被真正写进来的东西。

    单测要断言的正是"哪些没被写进来"，所以这里必须如实记录，不能吞掉。
    """

    def __init__(self) -> None:
        self.nodes: list[tuple[str, dict[str, Any]]] = []
        self.edges: list[tuple[str, str, dict[str, Any]]] = []
        self.batch_nodes: list[tuple[str, dict[str, Any]]] = []
        self.batch_edges: list[tuple[str, str, dict[str, Any]]] = []
        self.namespace = "fake"

    async def upsert_node(self, node_id: str, node_data: Mapping[str, Any]) -> None:
        self.nodes.append((node_id, dict(node_data)))

    async def upsert_edge(self, source_node_id: str, target_node_id: str, edge_data: Mapping[str, Any]) -> None:
        self.edges.append((source_node_id, target_node_id, dict(edge_data)))

    # 上游各后端会覆盖批量方法做原生批量写；夹具同样单独记录，用来验证护栏没有在批量路径上被绕过。
    async def upsert_nodes_batch(self, nodes: list[tuple[str, Mapping[str, Any]]]) -> None:
        for node_id, node_data in nodes:
            self.batch_nodes.append((node_id, dict(node_data)))

    async def upsert_edges_batch(self, edges: list[tuple[str, str, Mapping[str, Any]]]) -> None:
        for source_node_id, target_node_id, edge_data in edges:
            self.batch_edges.append((source_node_id, target_node_id, dict(edge_data)))

    def something_else(self) -> str:
        return "delegated"


class FakeRAG:
    def __init__(self, storage: FakeGraphStorage | None = None) -> None:
        self.chunk_entity_relation_graph = storage or FakeGraphStorage()


@pytest.fixture()
def storage() -> FakeGraphStorage:
    return FakeGraphStorage()


@pytest.fixture()
def rag(storage: FakeGraphStorage) -> FakeRAG:
    return FakeRAG(storage)


@pytest.fixture()
def rejects_file(tmp_path) -> str:
    return os.fspath(tmp_path / "graph_rejects.jsonl")


def node_data(**overrides: Any) -> dict[str, Any]:
    payload = {
        "entity_id": "供应商A",
        "entity_type": "供应商",
        "description": "供应紧固件的核心供应商",
        "source_id": "chunk-aaa<SEP>chunk-bbb",
        "file_path": "04-manufacturing-erp-data-qa.md",
    }
    payload.update(overrides)
    return payload


def edge_data(**overrides: Any) -> dict[str, Any]:
    payload = {
        "description": "供应商A 向 工厂B 供应紧固件",
        "keywords": "供应,紧固件",
        "weight": 1.0,
        "source_id": "chunk-ccc",
        "file_path": "04-manufacturing-erp-data-qa.md",
    }
    payload.update(overrides)
    return payload
