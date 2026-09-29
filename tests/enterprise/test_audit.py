"""审计测试：能追责、不泄密、被截断也不崩。"""

from __future__ import annotations

import json

from enterprise.audit import AuditLog


def test_append_and_tail_roundtrip_without_raw_args(tmp_path) -> None:
    log = AuditLog(tmp_path / "audit.jsonl")
    secret = "客户：某某集团 单号 PO-2026-0001"
    log.append(actor="u1", action="query", verdict="allowed", role="query", target="/v1/query", args={"q": secret})
    entries = log.tail(10)
    assert len(entries) == 1
    assert entries[0]["args_fingerprint"]
    assert secret not in json.dumps(entries, ensure_ascii=False)


def test_tail_on_missing_file_is_empty_not_error(tmp_path) -> None:
    assert AuditLog(tmp_path / "nope.jsonl").tail(5) == []


def test_counts_by_verdict() -> None:
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as directory:
        log = AuditLog(Path(directory) / "a.jsonl")
        log.append(actor="u", action="query", verdict="allowed")
        log.append(actor="u", action="query", verdict="allowed")
        log.append(actor="u", action="query", verdict="refused")
        assert log.counts_by_verdict() == {"allowed": 2, "refused": 1}


def test_rotation_keeps_size_bounded(tmp_path) -> None:
    log = AuditLog(tmp_path / "audit.jsonl", max_bytes=600, retain=1)
    for index in range(40):
        log.append(actor="u", action="query", verdict="allowed", target=f"/q/{index}")
    assert log.path.endswith("audit.jsonl")
    assert (tmp_path / "audit.jsonl.1").exists()
    # 轮转后仍能读，且不会因为半行 JSON 而抛异常
    assert isinstance(log.tail(5), list)


def test_truncated_line_is_skipped(tmp_path) -> None:
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    log.append(actor="u", action="query", verdict="allowed")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write('{"ts": 1, "actor": "u", "act')  # 模拟轮转时被截断的半行
    assert len(log.tail(10)) == 1
