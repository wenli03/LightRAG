"""确定性检索与摘录作答——降级链的链尾，也是评测可以在零模型环境下跑起来的原因。

这里刻意不追求"效果好"，追求的是三件事：

1. **可复现**：同样的语料 + 同样的问题 → 逐字节相同的输出（含排序、含小数位）；
2. **可解释**：每句话都能指回它来自哪条片段，因为**根本没有生成**，只有摘录；
3. **零依赖**：不装 numpy、不装向量库，纯标准库 BM25。

把它放进降级链尾，换来的是"面试现场断网/没额度也能把全流程走完"。
把它接进评测，换来的是"护栏阈值可以先在没有模型的情况下标定出来"——
这正是覆盖率这种判据比绝对分好用的地方。
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .text import split_sentences, term_set, tokenize, truncate

DEFAULT_LABEL = "[无模型模式·确定性摘录]"


@dataclasses.dataclass(frozen=True)
class SearchHit:
    doc_id: str
    score: float
    rank: int
    file_path: str = ""
    text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    file_path: str = ""


class BM25Index:
    """纯标准库 BM25（k1/b 与常见实现一致），排序确定性由 ``(-score, doc_id)`` 保证。"""

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs: dict[str, Document] = {}
        self._tf: dict[str, Counter[str]] = {}
        self._df: Counter[str] = Counter()
        self._lengths: dict[str, int] = {}
        self._avgdl = 0.0
        self._dirty = True

    def add(self, doc_id: str, text: str, *, file_path: str = "") -> None:
        self._docs[doc_id] = Document(doc_id=doc_id, text=text, file_path=file_path)
        tokens = tokenize(text)
        self._tf[doc_id] = Counter(tokens)
        self._lengths[doc_id] = max(1, len(tokens))
        self._dirty = True

    def extend(self, documents: Iterable[Document]) -> None:
        for document in documents:
            self.add(document.doc_id, document.text, file_path=document.file_path)

    def build(self) -> BM25Index:
        self._df = Counter()
        for counter in self._tf.values():
            self._df.update(counter.keys())
        total = sum(self._lengths.values())
        self._avgdl = (total / len(self._lengths)) if self._lengths else 0.0
        self._dirty = False
        return self

    @property
    def doc_count(self) -> int:
        return len(self._docs)

    def stats(self) -> dict[str, Any]:
        self._ensure_built()
        return {
            "doc_count": len(self._docs),
            "vocabulary": len(self._df),
            "avg_doc_length": round(self._avgdl, 3),
            "k1": self.k1,
            "b": self.b,
        }

    def document(self, doc_id: str) -> Document | None:
        return self._docs.get(doc_id)

    def documents(self) -> list[Document]:
        return [self._docs[key] for key in sorted(self._docs)]

    def _ensure_built(self) -> None:
        if self._dirty:
            self.build()

    def search(self, query: str, *, top_k: int = 5) -> list[SearchHit]:
        self._ensure_built()
        terms = tokenize(query)
        if not terms or not self._docs:
            return []
        total = len(self._docs)
        scores: dict[str, float] = {}
        for term in set(terms):
            df = self._df.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
            for doc_id, counter in self._tf.items():
                freq = counter.get(term, 0)
                if freq == 0:
                    continue
                length = self._lengths[doc_id]
                denominator = freq + self.k1 * (1 - self.b + self.b * length / (self._avgdl or 1.0))
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * (freq * (self.k1 + 1)) / denominator
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[: max(0, top_k)]
        return [
            SearchHit(
                doc_id=doc_id,
                score=round(score, 6),
                rank=index + 1,
                file_path=self._docs[doc_id].file_path,
                text=self._docs[doc_id].text,
            )
            for index, (doc_id, score) in enumerate(ranked)
        ]

    # ---- 持久化（保证"预置索引"这一招可复现，而不是每次启动重新算） ----

    def save(self, path: str | os.PathLike[str]) -> None:
        payload = {
            "k1": self.k1,
            "b": self.b,
            "documents": [
                {"doc_id": doc.doc_id, "text": doc.text, "file_path": doc.file_path}
                for doc in self.documents()
            ],
        }
        directory = os.path.dirname(os.path.abspath(os.fspath(path)))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> BM25Index:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        index = cls(k1=payload.get("k1", 1.5), b=payload.get("b", 0.75))
        for item in payload.get("documents", []):
            index.add(item["doc_id"], item["text"], file_path=item.get("file_path", ""))
        return index.build()


class DeterministicExtractive:
    """链尾作答器：把问题最相关的句子摘出来，并强制带上引用。

    这不是"假 AI"，它是**有意的降级**，而且在输出第一行就自我标注了这一点。
    一个不标注的降级，会让面试官以为模型答得这么差；一个标注了的降级，
    说明的是"产品在模型不可用时仍然可用"——这两件事的评价完全不同。
    """

    _SENTENCE_MIN_CHARS = 8

    def __init__(self, *, max_sentences: int = 3, label: str = DEFAULT_LABEL, max_chars_per_sentence: int = 220) -> None:
        self._max_sentences = max(1, max_sentences)
        self._label = label
        self._max_chars = max_chars_per_sentence

    def __call__(
        self,
        prompt: str,
        *,
        role: str = "query",  # noqa: ARG002 - 与降级链层契约对齐
        context: Sequence[Mapping[str, Any]] | None = None,
        question: str | None = None,
    ) -> str:
        asked = question or prompt
        chunks = list(context or [])
        if not chunks:
            return (
                f"{self._label}\n"
                f"关于「{truncate(asked, 80)}」，当前没有任何检索到的原文片段可作为依据，因此不作答。\n"
                f"（模型不可用且证据为空——两个原因同时出现时应直接告知用户，而不是编一段话。）"
            )

        wanted = term_set(asked)
        scored: list[tuple[float, int, str, str, str]] = []
        for index, chunk in enumerate(chunks):
            ref = str(chunk.get("reference_id") or f"ref-{index + 1}")
            path = str(chunk.get("file_path") or "（未标注来源文件）")
            for sentence in split_sentences(str(chunk.get("content") or "")):
                if len(sentence) < self._SENTENCE_MIN_CHARS:
                    continue
                overlap = len(term_set(sentence) & wanted)
                if overlap <= 0:
                    continue
                scored.append((float(overlap), -index, sentence, ref, path))

        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        picked = scored[: self._max_sentences]
        if not picked:
            return (
                f"{self._label}\n"
                f"检索到了 {len(chunks)} 条片段，但其中没有任何句子与「{truncate(asked, 60)}」有词元重叠，因此不作答。\n"
                f"（宁可空手而归，也不要摘一句无关的话充数。）"
            )

        lines = [self._label, "以下内容由检索到的原文片段直接摘录，未经模型改写："]
        used_sources: list[tuple[str, str]] = []
        for order, (_, _, sentence, ref, path) in enumerate(picked, start=1):
            lines.append(f"（{order}）{truncate(sentence.strip(), self._max_chars)} [{ref}]")
            if (ref, path) not in used_sources:
                used_sources.append((ref, path))
        lines.append("")
        lines.append("引用来源：")
        lines.extend(f"[{ref}] {path}" for ref, path in used_sources)
        return "\n".join(lines)


def chunks_from_hits(hits: Sequence[SearchHit], *, max_chars: int = 1200) -> list[dict[str, str]]:
    """把检索结果转成证据包形状，让下游护栏与摘录器用同一种输入。"""
    chunks: list[dict[str, str]] = []
    for hit in hits:
        chunks.append(
            {
                "reference_id": hit.doc_id,
                "file_path": hit.file_path or hit.doc_id,
                "content": truncate(hit.text, max_chars),
            }
        )
    return chunks


def split_into_chunks(text: str, *, target_chars: int = 600, overlap_chars: int = 80) -> list[str]:
    """**按段落边界**切块，段落超长才退回定长切。

    定长切会把一条业务规则拦腰截断，检索到半条规则比检索不到更危险。
    这里先按空行切段，段太长再按句号二次切，最后才允许硬切。
    """
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text or "") if part.strip()]
    pieces: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= target_chars:
            pieces.append(paragraph)
            continue
        buffer = ""
        for sentence in split_sentences(paragraph):
            if len(buffer) + len(sentence) > target_chars and buffer:
                pieces.append(buffer.strip())
                buffer = buffer[-overlap_chars:] if overlap_chars else ""
            buffer += sentence
        if buffer.strip():
            pieces.append(buffer.strip())
    out: list[str] = []
    for piece in pieces:
        if len(piece) <= target_chars * 2:
            out.append(piece)
            continue
        step = max(1, target_chars - overlap_chars)
        out.extend(piece[start : start + target_chars] for start in range(0, len(piece), step))
    return out
