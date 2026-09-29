"""评测 CLI：把"护栏调得怎么样"变成一条能复跑的命令。

    python -m enterprise.eval.run_eval \
        --corpus data/corpus \
        --golden data/golden/golden.jsonl \
        --sweep 0.3,0.4,0.5,0.6,0.7 \
        --repeat 2 \
        --out data/eval_report.json

跑完会打出：阈值扫描表 → 推荐阈值及其选择规则 → 该阈值下的完整指标 → 两次运行的逐字段差异。

**为什么必须默认跑两次**：只跑一次得到的数字，无法区分"确定的系统"和"碰巧对上的系统"。
两次逐字段一致，是"这个数字可以被别人复跑"的最小证明。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from ..answer_guard import AnswerGuardPolicy, CitationEnforcer, EvidenceGate
from ..offline import DeterministicExtractive, chunks_from_hits
from .corpus import build_index, corpus_stats, load_corpus
from .golden_set import GoldenItem, expected_map, load_jsonl, validate
from .metrics import Prediction, diff_reports, evaluate, recommend_threshold, sweep_thresholds


def build_predictions(
    chunks_by_file: Mapping[str, str],  # noqa: ARG001 - 保留签名以便未来接入 LightRAG 后端
    index: Any,
    items: Sequence[GoldenItem],
    *,
    threshold: float,
    top_k: int,
) -> list[Prediction]:
    gate = EvidenceGate(AnswerGuardPolicy(min_coverage=threshold))
    enforcer = CitationEnforcer()
    extractor = DeterministicExtractive()
    predictions: list[Prediction] = []

    for item in items:
        hits = index.search(item.question, top_k=top_k)
        raw = {
            "data": {
                "chunks": chunks_from_hits(hits),
                "references": [{"reference_id": hit.doc_id, "file_path": hit.file_path} for hit in hits],
                "entities": [],
                "relationships": [],
            }
        }
        verdict = gate.check({"question": item.question, "raw_data": raw})
        if verdict.rejected:
            predictions.append(
                Prediction(
                    item_id=item.item_id,
                    retrieved_files=tuple(hit.file_path for hit in hits),
                    answerable=item.answerable,
                    refused=True,
                    coverage=0.0,
                    has_citation=False,
                )
            )
            continue
        answer = extractor(item.question, context=raw["data"]["chunks"], question=item.question)
        _text, citation_verdict = enforcer.enforce(answer, raw["data"]["references"])
        predictions.append(
            Prediction(
                item_id=item.item_id,
                retrieved_files=tuple(hit.file_path for hit in hits),
                answerable=item.answerable,
                refused=False,
                coverage=float(verdict.detail.get("coverage") or 0.0),
                has_citation=bool(citation_verdict.detail.get("citations")),
            )
        )
    return predictions


def run(
    *,
    corpus_dir: str,
    golden_path: str,
    top_k: int = 5,
    thresholds: Sequence[float] = (0.3, 0.4, 0.5, 0.6, 0.7),
    repeat: int = 2,
    ks: Sequence[int] = (1, 3, 5),
) -> dict[str, Any]:
    chunks = load_corpus(corpus_dir)
    index, mapping = build_index(chunks)
    items = load_jsonl(golden_path)
    problems = validate(items)

    expected = expected_map(items)
    rows = sweep_thresholds(
        lambda threshold: build_predictions(mapping, index, items, threshold=threshold, top_k=top_k),
        thresholds,
        expected,
        ks=ks,
    )
    chosen = recommend_threshold(rows)
    if chosen is None:
        chosen = max(thresholds)

    runs: list[dict[str, Any]] = []
    for _ in range(max(1, repeat)):
        predictions = build_predictions(mapping, index, items, threshold=chosen, top_k=top_k)
        runs.append(evaluate(predictions, expected, ks=ks))

    differences = diff_reports(runs[0], runs[1]) if len(runs) > 1 else []

    return {
        "corpus": corpus_stats(chunks),
        "index": index.stats(),
        "golden": {
            "path": os.path.basename(golden_path),
            "sample_size": len(items),
            "validation_problems": problems,
        },
        "top_k": top_k,
        "threshold_sweep": rows,
        "selection_rule": "先要求 over_answer_rate == 0（宁可拒答也不编），再在其中选 false_refusal_rate 最低的阈值",
        "chosen_threshold": chosen,
        "metrics": {key: value for key, value in runs[0].items() if key != "per_item"},
        "runs_compared": len(runs),
        "deterministic": not differences,
        "differences": differences,
    }


def _render(report: Mapping[str, Any]) -> str:
    lines: list[str] = []
    corpus = report["corpus"]
    index = report["index"]
    lines.append(f"语料：{corpus['file_count']} 份文档 / {corpus['chunk_count']} 个切片，平均 {corpus['avg_chunk_chars']} 字")
    lines.append(f"索引：BM25（k1={index['k1']}, b={index['b']}），词表 {index['vocabulary']} 项，平均文档长度 {index['avg_doc_length']}")
    golden = report["golden"]
    lines.append(f"golden set：{golden['sample_size']} 条，自检问题 {len(golden['validation_problems'])} 个")
    for problem in golden["validation_problems"]:
        lines.append(f"  ⚠ {problem}")
    lines.append("")
    lines.append("阈值扫描（选择规则：先保证 over_answer_rate=0，再最小化误拒率）")
    lines.append(f"{'阈值':>6}  {'误答率':>8}  {'误拒率':>8}  {'拒答准确性':>10}  {'Recall@3':>9}  {'MRR':>6}")
    for row in report["threshold_sweep"]:
        lines.append(
            f"{row['threshold']:>6.2f}  {row['over_answer_rate']:>8.4f}  {row['false_refusal_rate']:>8.4f}  "
            f"{row['refusal_accuracy']:>10.4f}  {row['recall@3']:>9.4f}  {row['mrr']:>6.4f}"
        )
    lines.append("")
    lines.append(f"选定阈值：{report['chosen_threshold']}")
    lines.append("该阈值下的指标：")
    for key, value in report["metrics"].items():
        lines.append(f"  {key} = {value}")
    lines.append("")
    lines.append(f"两次运行逐字段一致：{report['deterministic']}（比较 {report['runs_compared']} 次）")
    for difference in report["differences"][:10]:
        lines.append(f"  ✗ {difference}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="LightRAG 治理层评测（零模型依赖，可离线复跑）")
    parser.add_argument("--corpus", required=True, help="语料目录（.md/.txt）")
    parser.add_argument("--golden", required=True, help="golden set JSONL")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--sweep", default="0.3,0.4,0.5,0.6,0.7", help="覆盖率阈值扫描点，逗号分隔")
    parser.add_argument("--repeat", type=int, default=2, help="重复运行次数，用于一致性检查")
    parser.add_argument("--out", default="", help="报告 JSON 落盘路径")
    args = parser.parse_args(argv)

    thresholds = tuple(float(part) for part in args.sweep.split(",") if part.strip())
    report = run(
        corpus_dir=args.corpus,
        golden_path=args.golden,
        top_k=args.top_k,
        thresholds=thresholds,
        repeat=args.repeat,
    )
    print(_render(report))
    if args.out:
        directory = os.path.dirname(os.path.abspath(args.out))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"\n报告已写入 {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
