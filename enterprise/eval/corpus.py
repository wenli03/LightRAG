"""语料装载与切块。

两个刻意的决定：

1. **只用自带/自写的语料**。上游仓库自带的 ``lightrag/evaluation/sample_documents/`` 是 MIT，
   自写的领域文档版权在项目作者手里，两边都不引入许可风险。把公司内部资料放进公开 Demo
   是最容易犯、也最难挽回的错误。
2. **按段落边界切块**（``split_into_chunks``），块 id 形如 ``文件名#0003``。
   块 id 里带文件名，是为了评测能落回"文档级命中"——召回一条片段就算命中了那份文档，
   这与"人是怎么判断答案对不对的"一致。
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Iterable, Sequence

from ..offline import BM25Index, split_into_chunks

SUPPORTED_SUFFIXES = (".md", ".txt")


@dataclasses.dataclass(frozen=True)
class Chunk:
    chunk_id: str
    file_name: str
    text: str

    @property
    def ordinal(self) -> int:
        _, _, tail = self.chunk_id.rpartition("#")
        return int(tail) if tail.isdigit() else 0


def load_corpus(directory: str | os.PathLike[str], *, max_chars: int = 600) -> list[Chunk]:
    directory = os.fspath(directory)
    chunks: list[Chunk] = []
    for file_name in sorted(os.listdir(directory)):
        if not file_name.lower().endswith(SUPPORTED_SUFFIXES):
            continue
        path = os.path.join(directory, file_name)
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8", errors="replace") as handle:
            text = handle.read()
        for index, piece in enumerate(split_into_chunks(text, target_chars=max_chars), start=1):
            chunks.append(Chunk(chunk_id=f"{file_name}#{index:04d}", file_name=file_name, text=piece))
    return chunks


def build_index(chunks: Iterable[Chunk], *, k1: float = 1.5, b: float = 0.75) -> tuple[BM25Index, dict[str, str]]:
    """返回 ``(索引, chunk_id -> 文件名)`` 映射。映射单独返回，避免从 id 里反解字符串。"""
    index = BM25Index(k1=k1, b=b)
    mapping: dict[str, str] = {}
    for chunk in chunks:
        index.add(chunk.chunk_id, chunk.text, file_path=chunk.file_name)
        mapping[chunk.chunk_id] = chunk.file_name
    index.build()
    return index, mapping


def corpus_stats(chunks: Sequence[Chunk]) -> dict[str, object]:
    files = sorted({chunk.file_name for chunk in chunks})
    return {
        "file_count": len(files),
        "chunk_count": len(chunks),
        "files": files,
        "avg_chunk_chars": round(sum(len(chunk.text) for chunk in chunks) / max(1, len(chunks)), 1),
    }
