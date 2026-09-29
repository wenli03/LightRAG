"""JSONL 审计日志。

字段只有 ``who / when / action / verdict / role / target / args_fingerprint``，
**不留原文参数**。理由：审计要能追责，但不能变成第二个泄密面——工具参数与检索结果里
经常带着客户数据、内部单号与联系方式。需要还原细节时，用指纹去业务侧日志里对齐。

按大小轮转，避免演示服务挂久了把磁盘写满。
"""

from __future__ import annotations

import dataclasses
import json
import os
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

from .contracts import fingerprint


@dataclasses.dataclass(frozen=True)
class AuditEntry:
    ts: float
    actor: str
    action: str
    verdict: str
    role: str | None = None
    target: str | None = None
    args_fingerprint: str | None = None
    detail: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        return {k: v for k, v in payload.items() if v is not None}


class AuditLog:
    """线程安全、按大小轮转的 JSONL 审计日志。"""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        max_bytes: int = 8 * 1024 * 1024,
        retain: int = 2,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self._path = os.fspath(path)
        self._lock = threading.Lock()
        self._max_bytes = max_bytes
        self._retain = max(1, retain)
        self._wall_clock = wall_clock
        directory = os.path.dirname(os.path.abspath(self._path))
        if directory:
            os.makedirs(directory, exist_ok=True)

    @property
    def path(self) -> str:
        return self._path

    def append(
        self,
        *,
        actor: str,
        action: str,
        verdict: str,
        role: str | None = None,
        target: str | None = None,
        args: Any = None,
        detail: Mapping[str, Any] | None = None,
    ) -> AuditEntry:
        entry = AuditEntry(
            ts=self._wall_clock(),
            actor=actor,
            action=action,
            verdict=verdict,
            role=role,
            target=target,
            args_fingerprint=fingerprint(args) if args is not None else None,
            detail=dict(detail) if detail else None,
        )
        line = json.dumps(entry.to_dict(), ensure_ascii=False, sort_keys=True)
        with self._lock:
            self._rotate_if_needed()
            with open(self._path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        return entry

    def tail(self, limit: int = 50) -> list[dict[str, Any]]:
        """读最后 ``limit`` 条。文件不存在返回空列表，不抛异常（健康检查会调它）。"""
        if not os.path.exists(self._path):
            return []
        with open(self._path, encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
        out: list[dict[str, Any]] = []
        for line in lines[-max(0, limit) :]:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # 轮转时可能截断出半行，跳过而不是让整个接口 500
        return out

    def counts_by_verdict(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in self.tail(limit=100000):
            verdict = str(entry.get("verdict", "unknown"))
            counts[verdict] = counts.get(verdict, 0) + 1
        return counts

    def _rotate_if_needed(self) -> None:
        if not os.path.exists(self._path):
            return
        try:
            size = os.path.getsize(self._path)
        except OSError:  # pragma: no cover - 文件被外部删掉
            return
        if size < self._max_bytes:
            return
        for index in range(self._retain, 0, -1):
            src = f"{self._path}.{index}"
            dst = f"{self._path}.{index + 1}"
            if index == self._retain:
                if os.path.exists(src):
                    os.remove(src)
                continue
            if os.path.exists(src):
                os.replace(src, dst)
        os.replace(self._path, f"{self._path}.1")
