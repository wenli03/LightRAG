"""可复现评测：语料装载 → 检索 → 护栏判定 → 指标 → 两次运行一致性。"""

from __future__ import annotations

from .corpus import Chunk, build_index, corpus_stats, load_corpus
from .golden_set import GoldenItem, expected_map, load_jsonl, save_jsonl, validate
from .metrics import (
    Prediction,
    citation_rate,
    diff_reports,
    evaluate,
    false_refusal_rate,
    mean_coverage,
    mrr,
    over_answer_rate,
    recall_at_k,
    recommend_threshold,
    refusal_accuracy,
    sweep_thresholds,
)

__all__ = [
    "Chunk",
    "GoldenItem",
    "Prediction",
    "build_index",
    "citation_rate",
    "corpus_stats",
    "diff_reports",
    "evaluate",
    "expected_map",
    "false_refusal_rate",
    "load_corpus",
    "load_jsonl",
    "mean_coverage",
    "mrr",
    "over_answer_rate",
    "recall_at_k",
    "recommend_threshold",
    "refusal_accuracy",
    "save_jsonl",
    "sweep_thresholds",
    "validate",
]
