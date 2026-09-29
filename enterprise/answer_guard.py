"""回答护栏：证据门槛拒答 + 引用强制。

两条规则：

1. **证据不够就拒答**，判定用**关键词覆盖率**而不是检索器的绝对分。
   绝对分是伪判据：BM25 的分数会被常见字堆高，向量余弦相似度会被模型自身的
   各向异性压缩（同一个语料上换一个 embedding 模型，阈值就全变了）。
   覆盖率是"问题里的词到底有没有出现在检索结果里"，换模型、换语言，口径都不动。

2. **答得出来就必须带引用**，而且缺引用时是**补上引用块并显式标注**，
   不是静默把没有引用支持的句子删掉——静默删除会让回答看起来干净，却把
   "模型说了没根据的话"这件事藏掉了。要藏的话，护栏就没有意义。
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

from .contracts import GuardResult, Reason, deny
from .text import STOPWORDS, coverage, truncate


@dataclasses.dataclass(frozen=True)
class Evidence:
    """归一化后的证据包。

    上游 ``lightrag/utils.py:7611 convert_to_user_format`` 产出的是
    ``{"data": {"entities": [...], "relationships": [...], "chunks": [...], "references": [...]}}``，
    这里把它压成"护栏真正需要的三件事"：有没有原文片段、片段带不带出处、有多少结构化事实。
    """

    chunks: tuple[Mapping[str, Any], ...] = ()
    references: tuple[Mapping[str, Any], ...] = ()
    entity_count: int = 0
    relationship_count: int = 0

    @property
    def texts(self) -> list[str]:
        return [str(chunk.get("content") or "") for chunk in self.chunks]

    @property
    def has_evidence(self) -> bool:
        return bool(self.chunks) or self.entity_count > 0 or self.relationship_count > 0

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)


def normalize_evidence(raw: Any) -> Evidence:
    """宽容地接受三种输入：上游 raw_data、``{"data": ...}``、或直接的 chunks 列表。"""
    if raw is None:
        return Evidence()
    if isinstance(raw, Evidence):
        return raw
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return Evidence(chunks=tuple(dict(item) for item in raw if isinstance(item, Mapping)))

    data: Any = raw
    if isinstance(raw, Mapping) and "data" in raw and isinstance(raw["data"], Mapping):
        data = raw["data"]
    if not isinstance(data, Mapping):
        return Evidence()

    chunks = tuple(dict(item) for item in (data.get("chunks") or []) if isinstance(item, Mapping))
    references = tuple(dict(item) for item in (data.get("references") or []) if isinstance(item, Mapping))
    entities = data.get("entities") or []
    relationships = data.get("relationships") or []
    return Evidence(
        chunks=chunks,
        references=references,
        entity_count=len(entities) if isinstance(entities, Sequence) else 0,
        relationship_count=len(relationships) if isinstance(relationships, Sequence) else 0,
    )


@dataclasses.dataclass(frozen=True)
class AnswerGuardPolicy:
    min_coverage: float = 0.5
    min_chunks: int = 1
    require_citation: bool = True
    max_cited_sources: int = 5

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_coverage <= 1.0:
            raise ValueError("min_coverage 必须落在 [0, 1]")


def refusal_message(question: str, detail: Mapping[str, Any]) -> str:
    """拒答话术。

    规则：**必须复述用户原问题**，并说清"差在哪"，而不是一句"我无法回答"。
    客服与知识问答产品里，"答非所问"和"答不上来还不说为什么"是一票否决项。
    """
    missing = ", ".join(sorted(detail.get("missed_terms") or [])[:6]) or "（未提取到有效关键词）"
    reason = detail.get("reason_text") or "检索结果不足以支撑回答"
    return (
        f"关于「{truncate(question, 80)}」这个问题，我暂时不能给出结论：{reason}。\n"
        f"当前检索到的证据里没有覆盖到的关键信息：{missing}。\n"
        f"建议：换一个更具体的问法，或补充相关文档后再问。"
    )


class EvidenceGate:
    """证据门槛。``check`` 的入参是 ``{"question": str, "raw_data": ...}``。"""

    def __init__(self, policy: AnswerGuardPolicy | None = None, *, stopwords: frozenset[str] = STOPWORDS) -> None:
        self._policy = policy or AnswerGuardPolicy()
        self._stopwords = stopwords

    @property
    def policy(self) -> AnswerGuardPolicy:
        return self._policy

    def check(self, payload: Mapping[str, Any]) -> GuardResult:
        question = str(payload.get("question") or "")
        evidence = normalize_evidence(payload.get("raw_data"))
        policy = self._policy

        if evidence.chunk_count < policy.min_chunks:
            return deny(
                Reason.NO_EVIDENCE,
                reason_text=f"没有检索到任何原文片段（要求至少 {policy.min_chunks} 条）",
                chunk_count=evidence.chunk_count,
                entity_count=evidence.entity_count,
                relationship_count=evidence.relationship_count,
                missed_terms=sorted(self._question_terms(question)),
            )

        rate, hit, missed = coverage(question, evidence.texts, stopwords=self._stopwords)
        detail = {
            "coverage": round(rate, 4),
            "threshold": policy.min_coverage,
            "chunk_count": evidence.chunk_count,
            "entity_count": evidence.entity_count,
            "relationship_count": evidence.relationship_count,
            "hit_terms": sorted(hit),
            "missed_terms": sorted(missed),
        }
        if rate < policy.min_coverage:
            detail["reason_text"] = (
                f"检索到的 {evidence.chunk_count} 条片段对问题的关键词覆盖率只有 {rate:.0%}，低于阈值 {policy.min_coverage:.0%}"
            )
            return deny(Reason.LOW_COVERAGE, **detail)
        return GuardResult(allowed=True, detail=detail)

    def _question_terms(self, question: str) -> set[str]:
        from .text import term_set

        return term_set(question, stopwords=self._stopwords)


class CitationEnforcer:
    """引用强制：回答里没有引用标记就补上引用块，并返回一个"不完全合规"的结论。"""

    def __init__(self, policy: AnswerGuardPolicy | None = None) -> None:
        self._policy = policy or AnswerGuardPolicy()

    def enforce(self, answer: str, references: Sequence[Mapping[str, Any]]) -> tuple[str, GuardResult]:
        answer = answer or ""
        limit = self._policy.max_cited_sources
        sources: list[tuple[str, str]] = []
        for index, reference in enumerate(list(references)[:limit], start=1):
            ref_id = str(reference.get("reference_id") or f"ref-{index}")
            path = str(reference.get("file_path") or "（未标注来源文件）")
            sources.append((ref_id, path))

        if not self._policy.require_citation or not sources:
            return answer, GuardResult(allowed=True, detail={"citations": len(sources), "injected": False})

        if any(f"[{ref_id}]" in answer for ref_id, _ in sources):
            return answer, GuardResult(allowed=True, detail={"citations": len(sources), "injected": False})

        block = "\n\n引用来源：\n" + "\n".join(f"[{ref_id}] {path}" for ref_id, path in sources)
        note = "\n\n（已自动补入引用块：模型原始输出未携带引用标记，这属于需要人工复核的信号。）"
        return answer + block + note, GuardResult(
            allowed=False,
            reason=Reason.MISSING_CITATION,
            detail={"citations": len(sources), "injected": True},
        )
