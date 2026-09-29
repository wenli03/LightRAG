"""把上游 LightRAG 装配起来，并把治理层挂上去。

装配这件事本身就是二开的证据：**上游一行没改**，治理层通过三个公开挂载点接进去——

1. ``llm_model_func``：换成带记账、带降级的实现（上游只要求它接受
   ``prompt / system_prompt / history_messages`` 这几个参数）；
2. ``embedding_func``：换成"优先真实模型、失败退本地哈希"的实现；
3. ``rag.chunk_entity_relation_graph``：套上图谱写入护栏（上游在 ``__post_init__`` 里
   把存储建好，之后所有读写都走这个属性，所以外面套一层代理就能全覆盖，含 rebuild 路径）。

唯一真正动到上游文件的地方是启动自检（``enforce_exposure``），
它解决的是上游默认白名单把 Ollama 兼容路由放行这个安全问题，属于可以提回上游的修复。
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping, Sequence
from typing import Any

from .embeddings import EmbeddingService, build_embedding_func
from .graph_guard import GraphGuardPolicy, GraphGuardStorage, install_graph_guard
from .metering import ROLES, MeteringLedger
from .providers import OpenAICompatibleChat, ProviderConfig


def _env_int(name: str, default: int, environ: Mapping[str, str] | None = None) -> int:
    raw = (environ if environ is not None else os.environ).get(name)
    try:
        return int(str(raw).strip()) if raw not in (None, "") else default
    except (TypeError, ValueError):
        return default


def resolve_llm_config(environ: Mapping[str, str] | None = None) -> ProviderConfig | None:
    """按优先级找一个可用的对话模型配置。

    ``ENTERPRISE_PRIMARY_*`` 是治理层自己的配置；``LIGHTRAG_LLM_*`` 是给"只想跑上游"的人留的兼容口。
    """
    for prefix in ("ENTERPRISE_PRIMARY", "LIGHTRAG_LLM", "ENTERPRISE_SECONDARY"):
        config = ProviderConfig.from_env(prefix, environ)
        if config is not None:
            return config
    return None


def resolve_embedding_config(environ: Mapping[str, str] | None = None) -> ProviderConfig | None:
    for prefix in ("ENTERPRISE_EMBEDDING", "LIGHTRAG_EMBEDDING"):
        config = ProviderConfig.from_env(prefix, environ)
        if config is not None:
            return config
    return None


def build_llm_func(
    config: ProviderConfig,
    *,
    ledger: MeteringLedger | None = None,
    role: str = "extract",
) -> Any:
    """包一层上游要的 LLM 调用签名，并在里面记账。

    记账放在这里而不是放在降级链里，是因为**抽取阶段根本不走降级链**——
    它是离线批处理，失败一次就整批重来。如果这里不记，抽取这笔最贵的开销就会从账上消失。
    """
    chat = OpenAICompatibleChat(config)

    async def _complete(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: Sequence[Mapping[str, str]] | None = None,
        **kwargs: Any,
    ) -> str:
        del kwargs  # 上游会传 enable_cot / stream / response_format 等；这里用统一实现，不逐个支持
        merged = prompt
        if system_prompt:
            merged = f"{system_prompt}\n\n{prompt}"
        context = list(history_messages or [])
        text, usage = await chat(merged, role=role, context=context)  # type: ignore[arg-type]
        if ledger is not None:
            ledger.record(
                role=role,
                layer=config.name,
                ok=True,
                latency_ms=0.0,
                tokens_in=usage.get("tokens_in"),
                tokens_out=usage.get("tokens_out"),
            )
        return text

    return _complete


async def build_lightrag(  # noqa: PLR0913 - 装配函数，参数就是它的说明书
    settings: Any,
    *,
    ledger: MeteringLedger | None = None,
    guard_policy: GraphGuardPolicy | None = None,
    environ: Mapping[str, str] | None = None,
) -> tuple[Any, GraphGuardStorage | None]:
    """构造并初始化上游实例，返回 ``(rag, graph_guard)``。

    护栏默认是 **record 模式**：先观测真实语料抽出来的实体类型，再据此收口词表。
    直接把词表拍脑袋定死再开拦截，结果会是整批抽取被静默丢弃——这条路径我们不想再走一遍。
    """
    from lightrag import LightRAG  # 懒 import：没装上游时本模块仍可被 import

    config = resolve_llm_config(environ)
    if config is None:
        raise RuntimeError(
            "没有可用的对话模型配置：请至少设置 ENTERPRISE_PRIMARY_BASE_URL / _API_KEY / _MODEL"
        )

    embedding_service = EmbeddingService(resolve_embedding_config(environ))
    working_dir = settings.working_dir
    os.makedirs(working_dir, exist_ok=True)

    # 每个角色单独包一层，而不是共用同一个包装函数：
    # 共用的话记账只能把角色写死成其中一个（实测第一次跑时查询调用被记在 extract 名下），
    # 那样「按四角色归因」就成了一句空话——账上只有一行，而且那一行还是错的。
    #
    # max_async 必须可配：按角色拆池之后并发度是"角色数 × 每池并发"，实测在免费额度上
    # 直接把接口打到限流，抽取阶段撞上上游默认的 480s worker 超时（任务失败、不是慢）。
    # 便宜/免费模型的正确姿势是先把它压到 1，sequential 反而更快跑完。
    max_async = _env_int("ENTERPRISE_LLM_MAX_ASYNC", 4)
    role_timeout = _env_int("ENTERPRISE_LLM_TIMEOUT", 900)
    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=build_llm_func(config, ledger=ledger, role="extract"),
        role_llm_configs={
            name: {
                "func": build_llm_func(config, ledger=ledger, role=name),
                "max_async": max_async,
                "timeout": role_timeout,
            }
            for name in ROLES
        },
        embedding_func=build_embedding_func(embedding_service),
        graph_storage="NetworkXStorage",
        kv_storage="JsonKVStorage",
        vector_storage="NanoVectorDBStorage",
        doc_status_storage="JsonDocStatusStorage",
    )

    policy = guard_policy or GraphGuardPolicy(on_violation="record")
    guard = install_graph_guard(rag, policy=policy, rejects_path=settings.rejects_path)

    await rag.initialize_storages()
    if hasattr(rag, "initialize_pipeline_status"):
        await rag.initialize_pipeline_status()

    metadata = {
        "llm": {"name": config.name, "model": config.model, "base_url": config.base_url},
        "embedding": embedding_service.choice.to_dict(),
        "graph_guard": {
            "mode": policy.on_violation,
            "require_source_id": policy.require_source_id,
            "vocab_size": None if policy.allowed_entity_types is None else len(policy.allowed_entity_types),
        },
    }
    embedding_service.write_metadata(working_dir)
    _write_json(os.path.join(working_dir, "runtime_choice.json"), metadata)
    return rag, guard


async def ingest(rag: Any, texts: Sequence[str], *, file_paths: Sequence[str] | None = None) -> None:
    """入库。上游会对每个切片跑实体关系抽取，这一步是**最贵**的，也是护栏真正生效的地方。

    ``file_paths`` 必须与 ``texts`` **等长并整体传入**：实测传入一个裸字符串时，
    下游把它当可迭代对象逐字符处理，引用里最终显示的是 ``['d']`` 这种被截断的第一个字符——
    一个不会报错、只会让"引用来源"这一栏变成垃圾值的问题。
    """
    documents = list(texts)
    if file_paths is None:
        await rag.ainsert(documents)
        return
    # 兼容单个字符串：Python 里 str 也是序列，直接 list() 会把它拆成一个个字符，
    # 而"拆成字符"这件事不会报错，只会让引用来源变成垃圾值——所以这里显式识别。
    paths = [file_paths] if isinstance(file_paths, str) else list(file_paths)
    if len(paths) != len(documents):
        raise ValueError(f"file_paths 与 texts 必须等长：{len(paths)} != {len(documents)}")
    await rag.ainsert(documents, file_paths=paths)


def _write_json(path: str, payload: Mapping[str, Any]) -> None:
    import json

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)


def run_sync(coro: Any) -> Any:  # pragma: no cover - 供脚本使用
    return asyncio.run(coro)
