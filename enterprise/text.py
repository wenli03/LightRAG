"""中文友好的零依赖分词与切句。

为什么不用 jieba/tiktoken：

* **确定性优先**。这里的词元会被写进评测口径（覆盖率怎么算、命中怎么判），
  口径必须每次都一样，且换台机器也一样。带词典的分词器换个版本就换一个口径。
* **零依赖**。护栏与评测要能在"什么都没装的机器"上跑，否则它就不能进 CI 门禁。

中文用 **bigram**（"存储后端" → 存储/储后/后端）而不是单字：单字命中率虚高，
"的""是"能把覆盖率刷满，覆盖率就废了；bigram 是中文上不用词典时最稳的折中。
"""

from __future__ import annotations

import re
from collections.abc import Iterable

_CJK = r"\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff"
_TOKEN_RE = re.compile(rf"[a-z0-9]+|[_{_CJK}]+")
_CJK_RUN_RE = re.compile(rf"[{_CJK}]+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;\n])")

# 只包含询问词与高频虚词：这些词在证据里本来就少见，留在分母里会系统性压低覆盖率，
# 把"答得出来"误判成"证据不足"（也就是误拒）。它们的**唯一作用**是让口径稳定。
STOPWORDS: frozenset[str] = frozenset(
    {
        "的",
        "了",
        "是",
        "在",
        "和",
        "与",
        "有",
        "我",
        "你",
        "它",
        "这",
        "那",
        "吗",
        "呢",
        "吧",
        "把",
        "被",
        "对",
        "从",
        "到",
        "为",
        "以",
        "及",
        "等",
        "就",
        "都",
        "也",
        "还",
        "只",
        "会",
        "能",
        "可以",
        "需要",
        "一个",
        "这个",
        "那个",
        "请问",
        "请",
        "一下",
        "什么",
        "怎么",
        "怎样",
        "如何",
        "哪些",
        "哪个",
        "为什么",
        "多少",
        "是否",
        "以及",
        "the",
        "a",
        "an",
        "is",
        "are",
        "of",
        "to",
        "and",
        "or",
        "what",
        "how",
        "which",
        "why",
        "in",
        "on",
        "for",
        "do",
        "does",
        "can",
    }
)


def tokenize(text: str) -> list[str]:
    """保留重复项（词频要参与 BM25 打分）。"""
    if not text:
        return []
    lowered = text.lower()
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(lowered):
        piece = match.group(0)
        if piece.isdigit() or piece.isascii():
            tokens.append(piece)
            continue

        def bigrams(run: str) -> Iterable[str]:
            if len(run) == 1:
                yield run
                return
            for index in range(len(run) - 1):
                yield run[index : index + 2]

        tokens.extend(bigrams(piece))
    return tokens


def term_set(text: str, *, stopwords: frozenset[str] = STOPWORDS) -> set[str]:
    """去重、去停用词后的词元集合——覆盖率与命中判定都用它，保证两侧口径一致。"""
    return {token for token in tokenize(text) if token not in stopwords}


def coverage(question: str, evidence_texts: Iterable[str], *, stopwords: frozenset[str] = STOPWORDS) -> tuple[float, set[str], set[str]]:
    """关键词覆盖率 = |问题词元 ∩ 证据词元| / |问题词元|。

    返回 ``(覆盖率, 命中的词, 未命中的词)``——把"漏了哪些词"一起返回，
    是为了让误拒可归因：是问题本身超纲，还是检索没召回，看漏词就能分辨。

    空问题视为覆盖率 0 而不是 1：无信息的问题不该被放行成一个自信的回答。
    """
    wanted = term_set(question, stopwords=stopwords)
    if not wanted:
        return 0.0, set(), set()
    found: set[str] = set()
    for text in evidence_texts:
        found |= term_set(text or "", stopwords=stopwords)
    hit = wanted & found
    missed = wanted - found
    return (len(hit) / len(wanted), hit, missed)


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    parts = [part.strip() for part in _SENTENCE_SPLIT_RE.split(text)]
    return [part for part in parts if part]


def truncate(text: str, limit: int, *, marker: str = "…") -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - len(marker))] + marker
