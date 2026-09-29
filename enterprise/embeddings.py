"""向量化：能用真实嵌入模型就用，不能用就退到**本地确定性哈希向量**。

为什么要留一个"哈希向量"这种看起来不专业的东西：因为在免费额度下，
嵌入接口经常比对话接口更早被限流（实测智谱免费额度下嵌入接口直接返回 429，
而对话接口正常）。如果整套系统只在"嵌入接口可用"时才跑得起来，
那它就又变成了一个依赖第三方可用性的系统——而这正是本项目要消除的东西。

必须说清楚它的**边界**：哈希向量只保证"同样的文本永远得到同样的向量"和"字面重叠的词元落在相同维度上"，
它**不具备语义泛化能力**。所以它适合验证链路与做离线评测，不适合评估检索质量。
把这个限定写进文档和 README，比把数字做得好看重要。
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any

from .providers import ProviderConfig
from .text import tokenize

DEFAULT_DIM = 1024


def hashing_embedding(texts: Sequence[str], *, dim: int = DEFAULT_DIM) -> list[list[float]]:
    """词元级哈希 + 符号折叠，再做 L2 归一化。确定性：同输入必然同输出。"""
    vectors: list[list[float]] = []
    for text in texts:
        vector = [0.0] * dim
        for token in tokenize(text or ""):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        norm = sum(value * value for value in vector) ** 0.5
        vectors.append([value / norm for value in vector] if norm else vector)
    return vectors


@dataclasses.dataclass(frozen=True)
class EmbeddingChoice:
    kind: str  # "remote" | "hashing"
    dim: int
    model: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _remote_embed(texts: Sequence[str], config: ProviderConfig, *, dim: int) -> list[list[float]]:
    payload = json.dumps({"model": config.model, "input": list(texts)}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{config.base_url}/embeddings",
        data=payload,
        headers={"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
        body = json.loads(response.read().decode("utf-8", errors="replace"))
    rows = sorted(body.get("data") or [], key=lambda item: item.get("index", 0))
    vectors = [list(row.get("embedding") or []) for row in rows]
    if not vectors or any(len(vector) != dim for vector in vectors):
        raise RuntimeError(f"嵌入接口返回维度与声明不符：声明 {dim}，实际 {[len(v) for v in vectors][:3]}")
    return vectors


def probe_remote(config: ProviderConfig, *, dim: int) -> tuple[bool, str]:
    """真发一次最小请求探活。返回 ``(可用, 原因)``——原因要能直接写进启动日志。"""
    try:
        _remote_embed(["探测"], config, dim=dim)
        return True, "ok"
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:160] if hasattr(exc, "read") else ""
        return False, f"HTTP {exc.code}: {detail}"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


class EmbeddingService:
    """先试远程，失败就落到本地哈希，并把"当前实际在用哪一个"暴露出来。

    **不静默降级**：切换嵌入后端会让已有索引失效（向量空间不是同一个），
    所以这里的做法是：降级只在**构建索引之前**发生，且把选择结果写进索引旁的一份元数据文件。
    索引一旦建好，后端就固定下来；换后端必须重建索引。
    """

    def __init__(self, config: ProviderConfig | None = None, *, dim: int = DEFAULT_DIM, force_hashing: bool = False) -> None:
        self._dim = dim
        self._config = config
        self._calls = 0
        if force_hashing or config is None:
            reason = "未配置嵌入模型" if config is None else "显式要求使用本地哈希向量"
            self._choice = EmbeddingChoice("hashing", dim, "local-hashing", reason)
        else:
            available, detail = probe_remote(config, dim=dim)
            self._choice = (
                EmbeddingChoice("remote", dim, config.model, "ok")
                if available
                else EmbeddingChoice("hashing", dim, "local-hashing", f"远程嵌入不可用（{detail}），退回本地哈希向量")
            )

    @property
    def choice(self) -> EmbeddingChoice:
        return self._choice

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def calls(self) -> int:
        return self._calls

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        self._calls += 1
        if self._choice.kind == "remote" and self._config is not None:
            try:
                return _remote_embed(texts, self._config, dim=self._dim)
            except Exception:  # noqa: BLE001 - 运行期掉线也不能让整条链路断掉
                self._choice = EmbeddingChoice(
                    "hashing", self._dim, "local-hashing", "远程嵌入在运行期失败，本次起改用本地哈希向量（已有索引可能已不一致）"
                )
        return hashing_embedding(texts, dim=self._dim)

    # ---- 索引元数据：把"用的是哪个嵌入"写下来，避免"换了模型却用旧索引"这类静默错误 ----

    def write_metadata(self, working_dir: str | os.PathLike[str]) -> str:
        path = os.path.join(os.fspath(working_dir), "embedding_choice.json")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self._choice.to_dict(), handle, ensure_ascii=False, indent=2, sort_keys=True)
        return path

    @staticmethod
    def read_metadata(working_dir: str | os.PathLike[str]) -> Mapping[str, Any] | None:
        path = os.path.join(os.fspath(working_dir), "embedding_choice.json")
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)


def build_embedding_func(service: EmbeddingService) -> Any:
    """包成上游 ``EmbeddingFunc``。上游内部会按 ``embedding_dim`` 校验维度，所以必须声明准确。"""
    from lightrag.utils import EmbeddingFunc  # 懒 import：本模块其余部分不依赖上游

    async def _embed(texts: Sequence[str]) -> Any:
        import numpy as np

        return np.array(service(texts), dtype="float32")

    return EmbeddingFunc(
        embedding_dim=service.dim,
        max_token_size=8192,
        func=_embed,
        model_name=service.choice.model,
    )
