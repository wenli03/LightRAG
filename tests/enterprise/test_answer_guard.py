"""回答护栏测试：该拒的要拒、该答的要答、缺引用要补而不是删。"""

from __future__ import annotations

import pytest

from enterprise.answer_guard import (
    AnswerGuardPolicy,
    CitationEnforcer,
    Evidence,
    EvidenceGate,
    normalize_evidence,
    refusal_message,
)
from enterprise.contracts import Reason
from enterprise.text import coverage, term_set


def raw_data(contents: list[str], *, refs: bool = True) -> dict[str, object]:
    chunks = [
        {"reference_id": f"ref-{index}", "file_path": f"doc-{index}.md", "content": text}
        for index, text in enumerate(contents, start=1)
    ]
    return {
        "data": {
            "chunks": chunks,
            "references": [{"reference_id": c["reference_id"], "file_path": c["file_path"]} for c in chunks] if refs else [],
            "entities": [],
            "relationships": [],
        }
    }


# --- 覆盖率口径 ---


def test_coverage_counts_union_of_evidence_not_single_chunk() -> None:
    question = "图谱写入为什么需要来源"
    one = coverage(question, ["图谱写入需要来源校验"])[0]
    two = coverage(question, ["图谱写入需要来源校验", "没有来源的关系会被多跳查询引用"])[0]
    assert two >= one, "召回更多片段不应该把覆盖率拉低"


def test_coverage_is_zero_when_question_has_no_terms() -> None:
    assert coverage("？", ["随便什么内容"])[0] == 0.0


def test_coverage_lists_missed_terms_for_attribution() -> None:
    rate, hit, missed = coverage("供应商报价与交付周期怎么算", ["供应商报价由采购维护"])
    assert 0 < rate < 1
    assert "报价" in hit
    assert missed, "漏掉的词必须被列出来，否则误拒无法归因"


def test_question_words_are_stopwords() -> None:
    assert "哪些" not in term_set("支持哪些存储后端")


# --- 证据门槛 ---


def test_gate_refuses_when_no_chunk_retrieved() -> None:
    verdict = EvidenceGate().check({"question": "图谱写入为什么需要来源", "raw_data": raw_data([])})
    assert verdict.rejected
    assert verdict.reason is Reason.NO_EVIDENCE


def test_gate_refuses_on_low_coverage() -> None:
    gate = EvidenceGate(AnswerGuardPolicy(min_coverage=0.8))
    verdict = gate.check({"question": "Neo4j 集群分片需要几个 coordinator 节点", "raw_data": raw_data(["本文只讲工程主题"])})
    assert verdict.rejected
    assert verdict.reason is Reason.LOW_COVERAGE
    assert verdict.detail["threshold"] == 0.8
    assert verdict.detail["missed_terms"]


def test_gate_allows_on_sufficient_coverage() -> None:
    gate = EvidenceGate(AnswerGuardPolicy(min_coverage=0.4))
    verdict = gate.check(
        {"question": "图谱写入为什么需要来源", "raw_data": raw_data(["图谱写入必须带来源标识，否则多跳查询会把无源关系当成事实"])}
    )
    assert verdict.allowed
    assert verdict.detail["coverage"] >= 0.4
    assert verdict.detail["chunk_count"] == 1


def test_normalize_accepts_three_shapes() -> None:
    assert normalize_evidence(None).chunk_count == 0
    assert normalize_evidence(raw_data(["a"])).chunk_count == 1
    assert normalize_evidence([{"content": "a"}]).chunk_count == 1
    assert normalize_evidence({"data": {"entities": [1, 2]}}).entity_count == 2
    evidence = Evidence(chunks=(), references=(), entity_count=1)
    assert evidence.has_evidence


def test_refusal_message_quotes_the_question() -> None:
    message = refusal_message("库存里 M8 螺栓还有多少", {"missed_terms": ["螺栓", "库存"]})
    assert "库存里 M8 螺栓还有多少" in message
    assert "螺栓" in message


def test_invalid_threshold_rejected() -> None:
    with pytest.raises(ValueError):
        AnswerGuardPolicy(min_coverage=1.5)


# --- 引用强制 ---


def test_citation_enforcer_keeps_answer_that_already_has_marker() -> None:
    answer = "结论如下 [ref-1]"
    text, verdict = CitationEnforcer().enforce(answer, [{"reference_id": "ref-1", "file_path": "a.md"}])
    assert text == answer
    assert verdict.allowed
    assert verdict.detail["injected"] is False


def test_citation_enforcer_appends_block_and_flags_missing_marker() -> None:
    text, verdict = CitationEnforcer().enforce("模型给了一段没有任何出处的话", [{"reference_id": "ref-9", "file_path": "b.md"}])
    assert "[ref-9] b.md" in text
    assert verdict.rejected, "缺引用是一个需要被看见的问题，不能返回 allowed"
    assert verdict.reason is Reason.MISSING_CITATION
    assert "需要人工复核" in text


def test_citation_enforcer_respects_max_sources() -> None:
    references = [{"reference_id": f"r{i}", "file_path": f"{i}.md"} for i in range(10)]
    text, verdict = CitationEnforcer(AnswerGuardPolicy(max_cited_sources=2)).enforce("无引用", references)
    assert verdict.detail["citations"] == 2
    assert "[r3]" not in text


def test_citation_enforcer_marks_missing_file_path() -> None:
    text, _ = CitationEnforcer().enforce("无引用", [{"reference_id": "r1"}])
    assert "未标注来源文件" in text
