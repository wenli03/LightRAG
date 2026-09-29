"""检索后端：把"证据从哪来"和"证据怎么用"解耦。

两种实现共用同一个返回形状（也就是上游 ``convert_to_user_format`` 的 ``raw_data``），
所以护栏、摘录器、评测、网关都不用关心背后是 BM25 还是 LightRAG：

* ``OfflineRetriever``：零依赖 BM25。断网、无 Key、无额度时仍然可用，也是离线评测的底座；
* ``LightRAGRetriever``：真上游。**只取上下文、不取生成**——生成由降级链统一管，
  这样"用哪个模型"和"检索到什么"这两件事不会互相绑死。
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

from .offline import BM25Index, chunks_from_hits


def _as_raw_data(payload: Any) -> dict[str, Any]:
    """把上游各种可能的返回形状统一成 ``{"data": {...}}``。"""
    if payload is None:
        return {"data": {"chunks": [], "references": [], "entities": [], "relationships": []}}
    raw = getattr(payload, "raw_data", None)
    if isinstance(raw, Mapping):
        payload = raw
    if isinstance(payload, Mapping):
        if isinstance(payload.get("data"), Mapping):
            data = dict(payload["data"])
        else:
            data = {
                "chunks": payload.get("chunks") or [],
                "references": payload.get("references") or [],
                "entities": payload.get("entities") or [],
                "relationships": payload.get("relationships") or [],
            }
        for key, default in (("chunks", []), ("references", []), ("entities", []), ("relationships", [])):
            data.setdefault(key, default)
        return {"data": data, "metadata": dict(payload.get("metadata") or {})}
    return {"data": {"chunks": [], "references": [], "entities": [], "relationships": []}}


class OfflineRetriever:
    def __init__(self, index: BM25Index, *, corpus_dir: str | None = None) -> None:
        self._index = index
        self._corpus_dir = corpus_dir

    @classmethod
    def from_corpus(cls, corpus_dir: str | os.PathLike[str]) -> OfflineRetriever:
        from .eval.corpus import build_index, load_corpus

        chunks = load_corpus(corpus_dir)
        index, _mapping = build_index(chunks)
        return cls(index, corpus_dir=os.fspath(corpus_dir))

    @classmethod
    def from_index_file(cls, path: str | os.PathLike[str]) -> OfflineRetriever:
        return cls(BM25Index.load(path))

    @property
    def stats(self) -> dict[str, Any]:
        return {"backend": "offline-bm25", **self._index.stats()}

    async def retrieve(self, question: str, *, top_k: int = 5) -> dict[str, Any]:
        hits = self._index.search(question, top_k=top_k)
        return {
            "data": {
                "chunks": chunks_from_hits(hits),
                "references": [{"reference_id": hit.doc_id, "file_path": hit.file_path} for hit in hits],
                "entities": [],
                "relationships": [],
            },
            "metadata": {"backend": "offline-bm25", "hits": [hit.to_dict() for hit in hits]},
        }


class LightRAGRetriever:
    """把上游 ``LightRAG`` 实例包成检索后端。

    ``only_need_context`` 是关键：它让上游只跑检索与上下文拼装，不调用生成模型。
    如果我们直接调上游的问答接口，那么"模型的可用性"就会重新变成整个链路的单点，
    而我们知道那正是要避免的事。
    """

    def __init__(self, rag: Any, *, mode: str = "mix", guard: Any | None = None) -> None:
        self._rag = rag
        self._mode = mode
        self._guard = guard

    async def retrieve(self, question: str, *, top_k: int = 5) -> dict[str, Any]:
        from lightrag import QueryParam  # 懒 import：没装上游时不至于连模块都导不进来

        param = QueryParam(mode=self._mode, top_k=top_k, only_need_context=True, include_references=True)
        if hasattr(self._rag, "aquery_data"):
            payload = await self._rag.aquery_data(question, param=param)
        else:  # 兼容旧版本：退化为带 only_need_context 的问答调用
            payload = await self._rag.aquery_llm(question, param=param)
        return _as_raw_data(payload)

    @property
    def stats(self) -> dict[str, Any]:
        return {"backend": "lightrag", "mode": self._mode}


class FallbackRetriever:
    """先试主检索后端，失败就切到离线索引。

    检索也会失败（向量库损坏、embedding 服务不可用、上游抛异常），
    而检索失败应当退化成"召回变差"，不应当退化成"服务不可用"。
    """

    def __init__(self, primary: Any, secondary: Any) -> None:
        self._primary = primary
        self._secondary = secondary
        self.last_backend = "primary"

    async def retrieve(self, question: str, *, top_k: int = 5) -> dict[str, Any]:
        try:
            payload = await self._primary.retrieve(question, top_k=top_k)
            self.last_backend = getattr(self._primary, "stats", {}).get("backend", "primary")
            return payload
        except Exception:  # noqa: BLE001 - 检索降级是"正常路径"，不该把异常抛给用户
            payload = await self._secondary.retrieve(question, top_k=top_k)
            self.last_backend = "offline-bm25(degraded)"
            data = payload.setdefault("data", {})
            data.setdefault("chunks", [])
            payload.setdefault("metadata", {})["retrieval_degraded"] = True
            return payload

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "backend": "fallback",
            "last_backend": self.last_backend,
            "primary": getattr(self._primary, "stats", None),
            "secondary": getattr(self._secondary, "stats", None),
        }


def normalise_references(data: Mapping[str, Any]) -> list[dict[str, str]]:
    references = data.get("references") or []
    out: list[dict[str, str]] = []
    for index, reference in enumerate(references, start=1):
        if not isinstance(reference, Mapping):
            continue
        out.append(
            {
                "reference_id": str(reference.get("reference_id") or f"ref-{index}"),
                "file_path": str(reference.get("file_path") or "未标注来源文件"),
            }
        )
    return out


def sequence_or_empty(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else ()
