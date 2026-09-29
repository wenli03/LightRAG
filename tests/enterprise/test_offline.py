"""离线检索与确定性作答测试：这一层的全部价值在于"每次跑都一样"。"""

from __future__ import annotations

from enterprise.offline import BM25Index, DeterministicExtractive, chunks_from_hits, split_into_chunks
from enterprise.text import tokenize


def test_tokenize_uses_bigrams_for_cjk() -> None:
    tokens = tokenize("存储后端")
    assert {"存储", "储后", "后端"} <= set(tokens)
    assert "存储后端" not in tokens


def test_tokenize_keeps_ascii_words_and_numbers() -> None:
    tokens = tokenize("LightRAG 用 BM25 k1=1.5")
    assert "lightrag" in tokens
    assert "bm25" in tokens
    assert "1" in tokens and "5" in tokens


def _index() -> BM25Index:
    index = BM25Index()
    index.add("b.md#0001", "图谱里的噪声是全局的，一条没有来源的关系会被多跳查询引用。", file_path="b.md")
    index.add("c.md#0001", "拒答阈值用关键词覆盖率标定，不用 BM25 的绝对分。", file_path="c.md")
    index.add("d.md#0001", "报工数据高频写入、不可变、强时序。", file_path="d.md")
    return index.build()


def test_search_is_deterministic_and_source_aware() -> None:
    index = _index()
    first = index.search("图谱里的噪声为什么是全局的", top_k=3)
    second = index.search("图谱里的噪声为什么是全局的", top_k=3)
    assert first == second
    assert first[0].file_path == "b.md"
    assert first[0].rank == 1


def test_search_tie_break_is_by_doc_id() -> None:
    index = BM25Index()
    index.add("z.md#0001", "同样的词元")
    index.add("a.md#0001", "同样的词元")
    index.build()
    hits = index.search("同样的词元", top_k=2)
    assert [hit.doc_id for hit in hits] == ["a.md#0001", "z.md#0001"], "打平时必须按 id 定序，否则结果不可复现"


def test_empty_query_returns_nothing() -> None:
    assert _index().search("的 了 是", top_k=3) == []


def test_index_save_load_roundtrip_is_identical(tmp_path) -> None:
    index = _index()
    path = tmp_path / "index.json"
    index.save(path)
    restored = BM25Index.load(path)
    assert restored.stats() == index.stats()
    assert [hit.to_dict() for hit in restored.search("报工数据", top_k=3)] == [
        hit.to_dict() for hit in index.search("报工数据", top_k=3)
    ]


def test_split_into_chunks_prefers_paragraph_boundaries() -> None:
    text = "第一段内容比较短。\n\n第二段内容也比较短。\n\n第三段内容同样短。"
    chunks = split_into_chunks(text, target_chars=600)
    assert chunks == ["第一段内容比较短。", "第二段内容也比较短。", "第三段内容同样短。"]


def test_split_into_chunks_splits_oversized_paragraph() -> None:
    sentence = "这是一句需要被切开的比较长的句子。"
    chunks = split_into_chunks(sentence * 40, target_chars=200, overlap_chars=20)
    assert len(chunks) > 1
    assert all(len(chunk) <= 400 for chunk in chunks)


def test_deterministic_extractive_picks_relevant_sentence_and_cites() -> None:
    extractor = DeterministicExtractive()
    context = [
        {"reference_id": "c1", "file_path": "02.md", "content": "图谱里的噪声是全局的。无关的一句。"},
        {"reference_id": "c2", "file_path": "06.md", "content": "审计日志只保留参数指纹，不保留参数原文。"},
    ]
    answer = extractor("审计日志为什么不记录参数原文", context=context)
    assert "无模型模式" in answer
    assert "参数指纹" in answer
    assert "[c2]" in answer
    assert "[c1]" not in answer


def test_deterministic_extractive_returns_label_when_context_empty() -> None:
    answer = DeterministicExtractive()("随便问点什么", context=[])
    assert "不作答" in answer


def test_chunks_from_hits_shape() -> None:
    hit = _index().search("报工数据", top_k=1)[0]
    chunks = chunks_from_hits([hit])
    assert set(chunks[0]) == {"reference_id", "file_path", "content"}
