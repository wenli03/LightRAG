"""治理网关：对外只暴露一个**带鉴权、带配额、带证据门槛**的只读问答接口。

为什么要有这一层，而不是直接把上游服务端暴露出去：上游服务端是一个"功能完整"的服务，
它假定调用方是可信的、上下文是充足的、额度是无限的。公网演示把这三个假定全部推翻。
网关把假设重新立起来——它不是多余的中间层，它是**把演示环境变成可公开环境的那个转换器**。

    python -m enterprise.app          # 启动前会先跑暴露面自检，不通过直接拒绝启动
"""

# ⚠️ 本模块**刻意不用** `from __future__ import annotations`。
#
# 原因是一个被真实 HTTP 测试抓到的缺陷：本模块为了"没装 fastapi 也能 import"而把 fastapi
# 延迟到 `create_app()` 里导入，于是路由函数的注解 `request: Request` 里的 `Request` 是**函数局部名字**。
# 一旦开启 PEP 563，注解变成字符串，FastAPI 只会在**模块全局**里解析它 —— 解析不到，
# 就把 request 当成一个必需的查询参数，结果是 `POST /v1/query` **一律返回 422**，
# 而进程内直接调 `gateway.answer()` 的测试全绿，完全发现不了。
# 保持注解的即时求值，它会在 `create_app()` 的局部作用域里正确解析。
import asyncio
import dataclasses
import os
import sys
import time
from collections.abc import Mapping, Sequence
from typing import Any

from .answer_guard import AnswerGuardPolicy, CitationEnforcer, EvidenceGate, normalize_evidence, refusal_message
from .audit import AuditLog
from .auth import (
    BearerGuard,
    DailyBudget,
    Finding,
    TokenBucketRateLimiter,
    audit_exposure,
    enforce_exposure,
    parse_whitelist,
)
from .contracts import Reason
from .fallback import LayerSpec, LayeredFallback, NoLayerAvailableError
from .metering import MeteringLedger
from .offline import DeterministicExtractive
from .providers import OpenAICompatibleChat, ProviderConfig
from .retrievers import FallbackRetriever, OfflineRetriever, normalise_references


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name) or default)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name) or default)
    except ValueError:
        return default


# 拒绝原因 → HTTP 状态码。**不能统一返回 200 再在响应体里写 refused**：
# 网关前面的 WAF、监控、日志聚合全都按状态码统计，"有人正在扫我的接口"这件事
# 如果只写在响应体里，在运维视图里就是完全不可见的。
# 响应体保持原样，客户端拿到的信息一点没少——这是一次纯粹的「让失败可见」的改动。
REFUSAL_STATUS: dict[str, int] = {
    Reason.UNAUTHORIZED.value: 401,
    Reason.RATE_LIMITED.value: 429,
    Reason.BUDGET_EXCEEDED.value: 429,
}


def refusal_status(reason: Any) -> int:
    """暴露面类拒绝返回 401 / 429；业务层拒答（证据不足等）保持 200——它是正常业务结果。"""
    return REFUSAL_STATUS.get(str(getattr(reason, "value", reason)), 200)


@dataclasses.dataclass(frozen=True)
class GatewaySettings:
    api_key: str | None = None
    host: str = "0.0.0.0"
    port: int = 9622
    whitelist: tuple[str, ...] = ("/health",)
    cors_origins: str | None = None
    rate_per_minute: float = 20.0
    burst: int = 5
    daily_calls: int = 200
    top_k: int = 5
    min_coverage: float = 0.4
    corpus_dir: str = "data/corpus"
    working_dir: str = "data/lightrag_storage"
    audit_path: str = "data/audit/audit.jsonl"
    rejects_path: str = "data/audit/graph_rejects.jsonl"
    retriever: str = "offline"  # offline | lightrag
    mode: str = "mix"

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "GatewaySettings":
        # 这里必须写成字符串：本模块不用 PEP 563（见文件头的说明），注解是即时求值的，
        # 而类名在类体内尚未绑定。整个模块里**只有这一处**需要这样写。

        env = environ if environ is not None else os.environ

        def get(name: str, default: str = "") -> str:
            return (env.get(name) or default).strip()

        def get_float(name: str, default: float) -> float:
            try:
                return float(get(name) or default)
            except ValueError:
                return default

        def get_int(name: str, default: int) -> int:
            try:
                return int(get(name) or default)
            except ValueError:
                return default

        return cls(
            api_key=get("ENTERPRISE_API_KEY") or get("LIGHTRAG_API_KEY") or None,
            host=get("ENTERPRISE_HOST", "0.0.0.0") or "0.0.0.0",
            port=get_int("ENTERPRISE_PORT", 9622),
            whitelist=tuple(parse_whitelist(get("ENTERPRISE_WHITELIST_PATHS", "/health"))),
            cors_origins=get("ENTERPRISE_CORS_ORIGINS") or None,
            rate_per_minute=get_float("ENTERPRISE_RATE_PER_MINUTE", 20.0),
            burst=get_int("ENTERPRISE_BURST", 5),
            daily_calls=get_int("ENTERPRISE_DAILY_CALLS", 200),
            top_k=get_int("ENTERPRISE_TOP_K", 5),
            min_coverage=get_float("ENTERPRISE_MIN_COVERAGE", 0.4),
            corpus_dir=get("ENTERPRISE_CORPUS_DIR", "data/corpus") or "data/corpus",
            working_dir=get("ENTERPRISE_WORKING_DIR", "data/lightrag_storage") or "data/lightrag_storage",
            audit_path=get("ENTERPRISE_AUDIT_PATH", "data/audit/audit.jsonl") or "data/audit/audit.jsonl",
            rejects_path=get("ENTERPRISE_REJECTS_PATH", "data/audit/graph_rejects.jsonl") or "data/audit/graph_rejects.jsonl",
            retriever=(get("ENTERPRISE_RETRIEVER", "offline") or "offline").lower(),
            mode=get("ENTERPRISE_QUERY_MODE", "mix") or "mix",
        )

    def exposure_kwargs(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "api_key": self.api_key,
            "whitelist_paths": list(self.whitelist),
            "cors_origins": self.cors_origins,
        }


class GovernanceGateway:
    """把鉴权、配额、检索、护栏、降级、记账、审计串成一次问答。"""

    def __init__(
        self,
        *,
        settings: GatewaySettings,
        retriever: Any,
        ledger: MeteringLedger,
        audit: AuditLog,
        fallback: LayeredFallback,
        gate: EvidenceGate | None = None,
        enforcer: CitationEnforcer | None = None,
        graph_guard: Any | None = None,
        clock: Any = time.monotonic,
    ) -> None:
        self.settings = settings
        self._retriever = retriever
        self._ledger = ledger
        self._audit = audit
        self._fallback = fallback
        self._gate = gate or EvidenceGate(AnswerGuardPolicy(min_coverage=settings.min_coverage))
        self._enforcer = enforcer or CitationEnforcer()
        self._graph_guard = graph_guard
        self._clock = clock
        self._bearer = BearerGuard(settings.api_key, whitelist=settings.whitelist, allow_anonymous_when_unset=False)
        self._limiter = TokenBucketRateLimiter(rate_per_minute=settings.rate_per_minute, burst=settings.burst)
        self._budget = DailyBudget(daily_calls=settings.daily_calls)

    # ---- 只读视图 ----

    def health(self) -> dict[str, Any]:
        findings = audit_exposure(**self.settings.exposure_kwargs())
        return {
            "status": "ok",
            "auth_configured": self._bearer.configured,
            "budget": self._budget.snapshot(),
            "retriever": getattr(self._retriever, "stats", {}),
            "fallback_layers": self._fallback.describe(),
            "exposure_findings": [finding.to_dict() for finding in findings],
        }

    def metering(self) -> dict[str, Any]:
        return {"snapshot": self._ledger.snapshot(), "cost": self._ledger.cost()}

    def graph_stats(self) -> dict[str, Any]:
        if self._graph_guard is None:
            return {"enabled": False}
        return {"enabled": True, **self._graph_guard.stats()}

    def audit_tail(self, limit: int = 30) -> dict[str, Any]:
        return {"entries": self._audit.tail(limit), "by_verdict": self._audit.counts_by_verdict()}

    # ---- 主流程 ----

    async def answer(
        self,
        *,
        question: str,
        client_id: str = "anonymous",
        path: str = "/v1/query",
        headers: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        started = self._clock()
        auth = self._bearer.check({"path": path, "headers": headers or {}})
        if auth.rejected:
            self._audit.append(actor=client_id, action="query", verdict=auth.reason.value, target=path)
            return self._envelope(question, refused=True, reason=auth.reason, detail=auth.detail, started=started)

        if not self._limiter.allow(client_id):
            self._audit.append(actor=client_id, action="query", verdict=Reason.RATE_LIMITED.value, target=path)
            return self._envelope(
                question,
                refused=True,
                reason=Reason.RATE_LIMITED,
                detail={"hint": "请求过于频繁，请稍后再试"},
                started=started,
            )

        if not self._budget.consume():
            self._audit.append(actor=client_id, action="query", verdict=Reason.BUDGET_EXCEEDED.value, target=path)
            return self._envelope(
                question,
                refused=True,
                reason=Reason.BUDGET_EXCEEDED,
                detail={"hint": "演示服务今日额度已用完，明天自动恢复", "budget": self._budget.snapshot()},
                started=started,
            )

        raw_data = await self._retriever.retrieve(question, top_k=self.settings.top_k)
        data = raw_data.get("data") or {}
        evidence = normalize_evidence(raw_data)
        chunks = list(data.get("chunks") or [])
        references = normalise_references(data)

        verdict = self._gate.check({"question": question, "raw_data": raw_data})
        if verdict.rejected:
            self._audit.append(
                actor=client_id,
                action="query",
                verdict=verdict.reason.value,
                target=path,
                args={"q": question},
                detail=verdict.detail,
            )
            return self._envelope(
                question,
                refused=True,
                reason=verdict.reason,
                detail=verdict.detail,
                started=started,
                retrieval={"chunks": len(chunks), "entities": evidence.entity_count},
                answer_override=refusal_message(question, verdict.detail),
            )

        try:
            outcome = await self._fallback.generate(question, context=chunks, question=question)
        except NoLayerAvailableError as exc:
            self._audit.append(actor=client_id, action="generate", verdict=Reason.NO_LAYER_AVAILABLE.value, target=path)
            return self._envelope(
                question,
                refused=True,
                reason=Reason.NO_LAYER_AVAILABLE,
                detail={"failures": exc.failures},
                started=started,
                answer_override="模型与本地兜底均不可用，本次不作答。",
            )

        answer, citation_verdict = self._enforcer.enforce(outcome.text, references)
        self._audit.append(
            actor=client_id,
            action="query",
            verdict="allowed",
            target=path,
            role=outcome.layer,
            args={"q": question},
            detail={"coverage": verdict.detail.get("coverage"), "layer": outcome.layer},
        )
        return self._envelope(
            question,
            refused=False,
            reason=citation_verdict.reason if citation_verdict.rejected else Reason.OK,
            detail={
                "coverage": verdict.detail.get("coverage"),
                "threshold": verdict.detail.get("threshold"),
                "citation_injected": citation_verdict.detail.get("injected"),
            },
            started=started,
            retrieval={"chunks": len(chunks), "entities": evidence.entity_count, "references": references},
            answer_override=answer,
            layer=outcome.layer,
            degraded=outcome.degraded,
        )

    # ---- 输出封装 ----

    def _envelope(
        self,
        question: str,
        *,
        refused: bool,
        reason: Reason,
        detail: Mapping[str, Any],
        started: float,
        retrieval: Mapping[str, Any] | None = None,
        answer_override: str | None = None,
        layer: str | None = None,
        degraded: bool = False,
    ) -> dict[str, Any]:
        return {
            "question": question,
            "answer": answer_override or "",
            "refused": refused,
            "reason": reason.value,
            "detail": dict(detail),
            "layer": layer,
            "degraded": degraded,
            "retrieval": dict(retrieval or {}),
            "latency_ms": round((self._clock() - started) * 1000.0, 3),
        }


def build_gateway(settings: GatewaySettings, *, rag: Any | None = None, graph_guard: Any | None = None) -> GovernanceGateway:
    """按配置装配网关。**不再需要任何"魔法"**：每一步都能在测试里替换。"""
    offline = OfflineRetriever.from_corpus(settings.corpus_dir)
    retriever: Any = offline
    if settings.retriever == "lightrag" and rag is not None:
        from .retrievers import LightRAGRetriever

        retriever = FallbackRetriever(LightRAGRetriever(rag, mode=settings.mode), offline)

    ledger = MeteringLedger()
    audit = AuditLog(settings.audit_path)

    layers: list[LayerSpec] = []
    for prefix in ("ENTERPRISE_PRIMARY", "ENTERPRISE_SECONDARY", "ENTERPRISE_LOCAL"):
        config = ProviderConfig.from_env(prefix)
        if config is None:
            name = prefix.rsplit("_", 1)[-1].lower()
            layers.append(LayerSpec(name, None, note=f"未配置 {prefix}_BASE_URL/_API_KEY/_MODEL"))
            continue
        chat = OpenAICompatibleChat(config)
        layers.append(LayerSpec(chat.name, chat, role=config.role, note=f"{config.model} @ {config.base_url}"))
    layers.append(LayerSpec("deterministic", DeterministicExtractive(), role="query", note="零依赖、无网络、恒可用"))

    fallback = LayeredFallback(layers, metering=ledger)
    return GovernanceGateway(
        settings=settings,
        retriever=retriever,
        ledger=ledger,
        audit=audit,
        fallback=fallback,
        graph_guard=graph_guard,
    )


def create_app(settings: GatewaySettings | None = None, *, gateway: GovernanceGateway | None = None) -> Any:
    """FastAPI 应用。延迟 import，保证没装 fastapi 的环境也能 import 本模块。"""
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, PlainTextResponse

    settings = settings or GatewaySettings.from_env()
    gateway = gateway or build_gateway(settings)

    app = FastAPI(title="LightRAG 治理网关", version="0.1.0", docs_url="/docs")
    origins = [item.strip() for item in (settings.cors_origins or "").split(",") if item.strip()]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type", "X-API-Key"],
        )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return gateway.health()

    def _as_response(payload: dict[str, Any]) -> Any:
        status = refusal_status(payload.get("reason")) if payload.get("refused") else 200
        return payload if status == 200 else JSONResponse(status_code=status, content=payload)

    def _authed(request: Request, handler: Any) -> Any:
        """三个只读接口共用的鉴权收口。

        刻意抽出来而不是各写一遍：**新增接口时漏掉鉴权不会有任何报错**，
        只会安静地多暴露一个端点——这类缺陷靠 code review 抓不住，只能靠"只有一个地方可写"。
        """
        verdict = gateway._bearer.check({"path": request.url.path, "headers": dict(request.headers)})  # noqa: SLF001
        if verdict.rejected:
            return _as_response({"refused": True, "reason": verdict.reason.value})
        return _as_response(handler())

    @app.post("/v1/query")
    async def query(request: Request) -> Any:
        payload = await request.json()
        question = str(payload.get("q") or payload.get("question") or "").strip()
        client = request.client.host if request.client else "unknown"
        if not question:
            return _as_response(
                {"refused": True, "reason": "empty_question", "answer": "请提供问题内容。", "detail": {}}
            )
        result = await gateway.answer(
            question=question,
            client_id=client,
            path=request.url.path,
            headers=dict(request.headers),
        )
        return _as_response(result)

    @app.get("/v1/metering")
    async def metering(request: Request) -> Any:
        return _authed(request, gateway.metering)

    @app.get("/v1/guard/graph")
    async def graph_stats(request: Request) -> Any:
        return _authed(request, gateway.graph_stats)

    @app.get("/v1/audit/tail")
    async def audit_tail(request: Request, limit: int = 30) -> Any:
        return _authed(request, lambda: gateway.audit_tail(limit))

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics() -> str:
        return gateway._ledger.prometheus_text()  # noqa: SLF001

    app.state.gateway = gateway
    app.state.settings = settings
    return app


def serve_uvicorn(app: Any, settings: GatewaySettings) -> None:
    """同步启动。

    演示面板的回调是异步函数，但它们运行在 uvicorn 自己的事件循环里，
    所以这里不需要（也不应该）再套一层 ``asyncio.run``——套了会让 Gradio 的协程落到不同的循环上。
    """
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port, log_level="warning")


async def serve(settings: GatewaySettings) -> None:
    """启动顺序是有讲究的：**先自检、再装配、最后才监听**。

    上游 storages 是异步初始化的，所以这里必须在同一个事件循环里完成装配与服务，
    否则文件型存储在跨事件循环时会出现难以定位的偶发失败。
    """
    import uvicorn

    for finding in enforce_exposure(**settings.exposure_kwargs()):
        print(f"[警告] {finding.code}: {finding.message}")

    rag = None
    graph_guard = None
    if settings.retriever == "lightrag":
        from .lightrag_integration import build_lightrag

        rag, graph_guard = await build_lightrag(settings)
        print(f"[启动] 上游 LightRAG 已就绪，工作目录 {settings.working_dir}，护栏模式 {graph_guard.policy.on_violation}")

    gateway = build_gateway(settings, rag=rag, graph_guard=graph_guard)
    app = create_app(settings, gateway=gateway)
    print(f"[启动] 网关监听 {settings.host}:{settings.port}，检索后端 {settings.retriever}")
    server = uvicorn.Server(uvicorn.Config(app, host=settings.host, port=settings.port, log_level="warning"))
    await server.serve()


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    asyncio.run(serve(GatewaySettings.from_env()))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
