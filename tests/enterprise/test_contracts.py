"""契约层断言：这几条是全套护栏的地基，错了上层全是错的。"""

from __future__ import annotations

import pytest

from enterprise.contracts import SOURCE_SEP, Reason, count_sources, deny, fingerprint


def test_source_sep_matches_upstream() -> None:
    """上游把多个来源切片标识用 ``<SEP>`` 拼起来；值一旦漂移，来源计数会静默变成"每串算 1 个"。

    没装上游时跳过（不是失败）：这条断言的价值在于"上游在场时必须对得上"，
    而不在于"没装上游也要能跑"。
    """
    constants = pytest.importorskip("lightrag.constants")

    assert SOURCE_SEP == constants.GRAPH_FIELD_SEP


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, 0),
        ("", 0),
        ("   ", 0),
        ("<SEP>", 0),  # 非空字符串但零来源——正是"只看非空"会漏掉的那一种
        ("<SEP><SEP>", 0),
        ("chunk-a", 1),
        ("chunk-a<SEP>chunk-b", 2),
        ("chunk-a<SEP> <SEP>chunk-b", 2),
        (["chunk-a", "chunk-b"], 2),
        (["chunk-a", ""], 1),
        ([], 0),
    ],
)
def test_count_sources(value: object, expected: int) -> None:
    assert count_sources(value) == expected


def test_fingerprint_is_stable_and_key_order_independent() -> None:
    assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})
    assert fingerprint({"a": 1}) != fingerprint({"a": 2})
    assert len(fingerprint({"a": 1})) == 16


def test_fingerprint_does_not_leak_content() -> None:
    """审计只留指纹：原始字符串本身不能出现在输出里。"""
    secret = "客户名称：某某集团；联系人：13800000000"
    assert secret not in fingerprint({"args": secret})


def test_deny_carries_structured_detail() -> None:
    result = deny(Reason.LOW_COVERAGE, coverage=0.31, threshold=0.5)
    assert result.rejected
    assert result.reason is Reason.LOW_COVERAGE
    assert result.detail["coverage"] == 0.31
    assert result.to_dict()["reason"] == "low_coverage"


def test_reason_values_are_stable_strings() -> None:
    assert str(Reason.NO_SOURCE) == "no_source"
    assert Reason("no_source") is Reason.NO_SOURCE
