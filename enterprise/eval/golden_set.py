"""golden set 的数据结构与自检。

自检（``validate``）不是形式主义：一个"无答案"的样本如果被误标成"有答案"，
误拒率会凭空升高、拒答准确性会凭空降低，而**报告上看不出任何异常**。
所以样本集必须自己先能证明自己是一致的。
"""

from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Iterable, Sequence
from typing import Any


@dataclasses.dataclass(frozen=True)
class GoldenItem:
    item_id: str
    question: str
    answerable: bool
    expected_doc_ids: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "item_id": self.item_id,
            "question": self.question,
            "answerable": self.answerable,
        }
        if self.expected_doc_ids:
            payload["expected_doc_ids"] = list(self.expected_doc_ids)
        if self.keywords:
            payload["keywords"] = list(self.keywords)
        if self.note:
            payload["note"] = self.note
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GoldenItem:
        return cls(
            item_id=str(payload["item_id"]),
            question=str(payload["question"]),
            answerable=bool(payload.get("answerable", True)),
            expected_doc_ids=tuple(str(item) for item in payload.get("expected_doc_ids", ())),
            keywords=tuple(str(item) for item in payload.get("keywords", ())),
            note=str(payload.get("note", "")),
        )


def load_jsonl(path: str | os.PathLike[str]) -> list[GoldenItem]:
    items: list[GoldenItem] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                items.append(GoldenItem.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"{path}:{line_number} 不是合法的 golden 样本：{exc}") from exc
    return items


def save_jsonl(items: Iterable[GoldenItem], path: str | os.PathLike[str]) -> None:
    directory = os.path.dirname(os.path.abspath(os.fspath(path)))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


def validate(items: Sequence[GoldenItem]) -> list[str]:
    """返回问题清单，空列表 = 样本集自洽。"""
    problems: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item.item_id in seen:
            problems.append(f"item_id 重复: {item.item_id}")
        seen.add(item.item_id)
        if not item.question.strip():
            problems.append(f"{item.item_id}: 问题为空")
        if item.answerable and not item.expected_doc_ids:
            problems.append(f"{item.item_id}: 标注为有答案，但没有给 expected_doc_ids，Recall 会被算成漏召回")
        if not item.answerable and item.expected_doc_ids:
            problems.append(f"{item.item_id}: 标注为无答案，却给了 expected_doc_ids，两个口径互相矛盾")
    if not items:
        problems.append("样本集为空")
    return problems


def expected_map(items: Sequence[GoldenItem]) -> dict[str, tuple[str, ...]]:
    return {item.item_id: item.expected_doc_ids for item in items}
