"""评测口径测试。

指标的定义要能被手算复核，所以这里的期望值都是**人算出来写死的**，
不是"跑一遍看看输出什么就抄下来"——后者等于把 bug 固化成基线。
"""

from __future__ import annotations

import pytest

from enterprise.eval.golden_set import GoldenItem, expected_map, validate
from enterprise.eval.metrics import (
    Prediction,
    diff_reports,
    evaluate,
    false_refusal_rate,
    mrr,
    over_answer_rate,
    recall_at_k,
    recommend_threshold,
    refusal_accuracy,
)
from enterprise.eval.run_eval import run

from .conftest import REPO_ROOT


def test_recall_at_k_hand_computed() -> None:
    predictions = [
        Prediction("a", retrieved_files=("x.md", "y.md")),
        Prediction("b", retrieved_files=("x.md",)),
        Prediction("c", retrieved_files=("z.md",)),
    ]
    expected = {"a": ("y.md",), "b": ("x.md",), "c": ("w.md",)}
    assert recall_at_k(predictions, expected, 1) == 0.3333  # a 的第一条是 x 不是 y
    assert recall_at_k(predictions, expected, 2) == 0.6667  # a 命中
    assert recall_at_k(predictions, expected, 3) == 0.6667  # c 的 w.md 从未被召回


def test_recall_denominator_excludes_unanswerable() -> None:
    """分母只算有答案的题；把无答案的题也算进去，拒答做得越狠 Recall 反而越好看。"""
    predictions = [
        Prediction("a", retrieved_files=("x.md",)),
        Prediction("u", retrieved_files=("nope.md",), answerable=False),
    ]
    assert recall_at_k(predictions, {"a": ("x.md",)}, 1) == 1.0


def test_mrr_counts_misses_as_zero_instead_of_skipping() -> None:
    predictions = [
        Prediction("hit-first", retrieved_files=("want.md",)),
        Prediction("hit-third", retrieved_files=("a.md", "b.md", "want.md")),
        Prediction("miss", retrieved_files=("a.md",)),
    ]
    expected = {"hit-first": ("want.md",), "hit-third": ("want.md",), "miss": ("want.md",)}
    assert mrr(predictions, expected) == round((1.0 + 1 / 3 + 0.0) / 3, 4)


def test_refusal_metrics() -> None:
    predictions = [
        Prediction("ok", answerable=True, refused=False),
        Prediction("false-refusal", answerable=True, refused=True),
        Prediction("ok-unanswerable", answerable=False, refused=True),
        Prediction("over-answer", answerable=False, refused=False),
    ]
    assert refusal_accuracy(predictions) == 0.5
    assert false_refusal_rate(predictions) == 0.5
    assert over_answer_rate(predictions) == 0.5


def test_evaluate_reports_cost_of_the_threshold() -> None:
    predictions = [Prediction("a", retrieved_files=("x.md",), refused=True), Prediction("u", answerable=False, refused=True)]
    report = evaluate(predictions, {"a": ("x.md",)}, ks=(1, 3))
    assert report["false_refusal_rate"] == 1.0 and report["over_answer_rate"] == 0.0
    assert "recall@1" in report and "recall@5" not in report
    assert report["per_item"][0]["item_id"] == "a"


def test_recommend_threshold_prefers_zero_over_answer_then_lowest_false_refusal() -> None:
    rows = [
        {"threshold": 0.3, "over_answer_rate": 0.25, "false_refusal_rate": 0.0},
        {"threshold": 0.5, "over_answer_rate": 0.0, "false_refusal_rate": 0.2},
        {"threshold": 0.6, "over_answer_rate": 0.0, "false_refusal_rate": 0.1},
    ]
    assert recommend_threshold(rows) == 0.6
    assert recommend_threshold([{**rows[0]}]) is None


def test_diff_reports_catches_scalar_and_per_item_changes() -> None:
    first = {"mrr": 1.0, "per_item": [{"item_id": "a", "refused": False}]}
    same = {"mrr": 1.0, "per_item": [{"item_id": "a", "refused": False}]}
    changed = {"mrr": 0.9, "per_item": [{"item_id": "a", "refused": True}]}
    assert diff_reports(first, same) == []
    assert len(diff_reports(first, changed)) == 2


def test_golden_validation_catches_inconsistent_labels() -> None:
    problems = validate(
        [
            GoldenItem("a", "问题", answerable=True),
            GoldenItem("a", "重复 id", answerable=False, expected_doc_ids=("x.md",)),
        ]
    )
    joined = " | ".join(problems)
    assert "item_id 重复" in joined
    assert "没有给 expected_doc_ids" in joined
    assert "两个口径互相矛盾" in joined


def test_golden_expected_map() -> None:
    items = [GoldenItem("a", "q", True, ("x.md",))]
    assert expected_map(items) == {"a": ("x.md",)}


def test_end_to_end_eval_is_deterministic_on_shipped_corpus() -> None:
    """对仓库自带语料跑真实评测：一是数字拿得到，二是两次逐字段一致。"""
    report = run(
        corpus_dir=str(REPO_ROOT / "data" / "corpus"),
        golden_path=str(REPO_ROOT / "data" / "golden" / "golden.jsonl"),
        repeat=2,
    )
    assert report["golden"]["validation_problems"] == []
    assert report["golden"]["sample_size"] == 22
    assert report["deterministic"] is True
    assert report["differences"] == []
    assert report["chosen_threshold"] > 0
    assert report["metrics"]["over_answer_rate"] == 0.0
    assert report["metrics"]["citation_rate"] == 1.0
    assert report["corpus"]["file_count"] == 7
    assert report["selection_rule"]
