"""检索与护栏的评测指标。

每个指标都**带明确口径**，因为"报一个准确率"这件事本身没有信息量：

* **Recall@K**：分母是**有答案的问题**（``answerable=True``）；分子是 top-K 里
  命中至少一条期望来源的题数。K 与"命中"的定义都在报告里写出来。
* **MRR**：只对命中的题算 1/首个命中排名，**未命中的题计 0 而不是跳过**——
  跳过会让 MRR 在召回差的时候反而变好看。
* **拒答准确性**：``(有答案且没拒 + 无答案且拒了) / 总数``。把两类错误合并成一个数，
  是为了避免"拒答率调高就能刷好看"。
* **误拒率**：有答案的题里被拒的比例——拒答阈值调高的代价，**必须单独披露**，
  藏起来就等于只报了好消息。
* **引用率**：作答题里带引用标记的比例。
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


@dataclasses.dataclass(frozen=True)
class Prediction:
    """一条预测结果。``retrieved_files`` 按排名从高到低排列。"""

    item_id: str
    retrieved_files: tuple[str, ...] = ()
    answerable: bool = True
    refused: bool = False
    coverage: float = 0.0
    has_citation: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "retrieved_files": list(self.retrieved_files),
            "answerable": self.answerable,
            "refused": self.refused,
            "coverage": self.coverage,
            "has_citation": self.has_citation,
        }


def recall_at_k(predictions: Sequence[Prediction], expected: Mapping[str, Sequence[str]], k: int) -> float:
    answerable = [p for p in predictions if p.answerable]
    if not answerable:
        return 0.0
    hits = 0
    for prediction in answerable:
        wanted = set(expected.get(prediction.item_id, ()))
        if wanted & set(prediction.retrieved_files[:k]):
            hits += 1
    return round(hits / len(answerable), 4)


def mrr(predictions: Sequence[Prediction], expected: Mapping[str, Sequence[str]]) -> float:
    answerable = [p for p in predictions if p.answerable]
    if not answerable:
        return 0.0
    total = 0.0
    for prediction in answerable:
        wanted = set(expected.get(prediction.item_id, ()))
        for rank, file_name in enumerate(prediction.retrieved_files, start=1):
            if file_name in wanted:
                total += 1.0 / rank
                break
    return round(total / len(answerable), 4)


def refusal_accuracy(predictions: Sequence[Prediction]) -> float:
    if not predictions:
        return 0.0
    correct = sum(1 for p in predictions if (not p.refused) == bool(p.answerable))
    return round(correct / len(predictions), 4)


def false_refusal_rate(predictions: Sequence[Prediction]) -> float:
    answerable = [p for p in predictions if p.answerable]
    if not answerable:
        return 0.0
    return round(sum(1 for p in answerable if p.refused) / len(answerable), 4)


def over_answer_rate(predictions: Sequence[Prediction]) -> float:
    """无答案的题里被作答的比例——与误拒率互为代价，必须两个一起报。"""
    unanswerable = [p for p in predictions if not p.answerable]
    if not unanswerable:
        return 0.0
    return round(sum(1 for p in unanswerable if not p.refused) / len(unanswerable), 4)


def citation_rate(predictions: Sequence[Prediction]) -> float:
    answered = [p for p in predictions if not p.refused]
    if not answered:
        return 0.0
    return round(sum(1 for p in answered if p.has_citation) / len(answered), 4)


def mean_coverage(predictions: Sequence[Prediction]) -> float:
    if not predictions:
        return 0.0
    return round(sum(p.coverage for p in predictions) / len(predictions), 4)


def evaluate(
    predictions: Sequence[Prediction],
    expected: Mapping[str, Sequence[str]],
    *,
    ks: Iterable[int] = (1, 3, 5),
) -> dict[str, Any]:
    ks = tuple(sorted(set(ks)))
    report: dict[str, Any] = {
        "sample_size": len(predictions),
        "answerable_size": sum(1 for p in predictions if p.answerable),
        "unanswerable_size": sum(1 for p in predictions if not p.answerable),
    }
    for k in ks:
        report[f"recall@{k}"] = recall_at_k(predictions, expected, k)
    report["mrr"] = mrr(predictions, expected)
    report["refusal_accuracy"] = refusal_accuracy(predictions)
    report["false_refusal_rate"] = false_refusal_rate(predictions)
    report["over_answer_rate"] = over_answer_rate(predictions)
    report["citation_rate"] = citation_rate(predictions)
    report["mean_coverage"] = mean_coverage(predictions)
    report["per_item"] = [p.to_dict() for p in predictions]
    return report


def diff_reports(first: Mapping[str, Any], second: Mapping[str, Any]) -> list[str]:
    """两次运行的逐字段差异。空列表 = 完全一致。

    只比较标量字段与 ``per_item`` 的标量部分；``per_item`` 的列表按顺序比对，
    因为**顺序本身就是结果的一部分**（检索排名变了，指标也就变了）。
    """
    differences: list[str] = []
    keys = sorted(set(first) | set(second))
    for key in keys:
        left, right = first.get(key), second.get(key)
        if left == right:
            continue
        if key == "per_item" and isinstance(left, list) and isinstance(right, list):
            for index, (a, b) in enumerate(zip(left, right, strict=False)):
                if a != b:
                    differences.append(f"per_item[{index}] 不一致: {a} != {b}")
            if len(left) != len(right):
                differences.append(f"per_item 长度不一致: {len(left)} != {len(right)}")
            continue
        differences.append(f"{key} 不一致: {left!r} != {right!r}")
    return differences


def sweep_thresholds(
    build_predictions,
    thresholds: Sequence[float],
    expected: Mapping[str, Sequence[str]],
    *,
    ks: Iterable[int] = (1, 3, 5),
) -> list[dict[str, Any]]:
    """阈值标定扫描。

    ``build_predictions(threshold)`` 返回该阈值下的预测列表。选点不是"取最大值"，
    而是先要求 ``over_answer_rate == 0``（宁可拒答也不编），再在满足条件的阈值里选误拒率最低的。
    把选择规则写进代码，是为了让"阈值 0.5"这个数字**可以被复现和质疑**，而不是一个魔法数。
    """
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        report = evaluate(build_predictions(threshold), expected, ks=ks)
        rows.append(
            {
                "threshold": round(float(threshold), 4),
                "over_answer_rate": report["over_answer_rate"],
                "false_refusal_rate": report["false_refusal_rate"],
                "refusal_accuracy": report["refusal_accuracy"],
                "recall@3": report.get("recall@3"),
                "mrr": report["mrr"],
            }
        )
    return rows


def recommend_threshold(rows: Sequence[Mapping[str, Any]]) -> float | None:
    safe = [row for row in rows if float(row.get("over_answer_rate") or 0.0) == 0.0]
    if not safe:
        return None
    best = min(safe, key=lambda row: (float(row.get("false_refusal_rate") or 0.0), -float(row.get("threshold") or 0.0)))
    return float(best["threshold"])
