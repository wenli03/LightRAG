"""LightRAG 企业落地治理层（overlay）。

**它是什么**：一套加在 HKUDS/LightRAG 之上的治理层，补的是"从能跑到能进生产"之间的那段路——
公网暴露面加固、图谱写入护栏、证据门槛拒答与引用强制、四级降级链、四角色成本归因、可复现评测。

**它不是什么**：不是另一个 RAG 框架，也不接管检索与生成。检索、抽取、图谱构建、
WebUI 全部仍由上游完成，本层只在**写入前**与**回答前**插两道闸门，外加一层记账。

**设计上的三条硬约束**：

1. **不 import 上游**（除非显式开启 ``validate_attributes``）。整个包只依赖标准库，
   所以它能脱离重依赖跑单测、能进 CI 门禁——护栏"有没有真的拦住"必须是可证伪的。
2. **不改上游函数**，通过包装 ``rag.chunk_entity_relation_graph`` 与包裹 ``llm_model_func`` 接入，
   上游 diff 只保留一处启动自检。rebase 不痛，diff 能逐行讲。
3. **链尾必须是确定性的**。整个系统在断网、无 Key、无余额的情况下仍然要能跑完全流程，
   否则"面试现场演示"这件事就永远依赖第三方。

入口通常只有两行::

    from enterprise import install_graph_guard, GraphGuardPolicy
    guard = install_graph_guard(rag, policy=GraphGuardPolicy(on_violation="record"))
"""

from __future__ import annotations

from .answer_guard import (
    AnswerGuardPolicy,
    CitationEnforcer,
    Evidence,
    EvidenceGate,
    normalize_evidence,
    refusal_message,
)
from .audit import AuditEntry, AuditLog
from .auth import (
    BearerGuard,
    DailyBudget,
    ExposureError,
    Finding,
    Severity,
    TokenBucketRateLimiter,
    audit_exposure,
    enforce_exposure,
    is_loopback,
    parse_whitelist,
    path_is_whitelisted,
)
from .contracts import ALLOW, ContractError, Guard, GuardResult, Reason, count_sources, deny, fingerprint
from .fallback import (
    EXPECTED_FAILURE_MARKERS,
    FallbackOutcome,
    LayerSpec,
    LayeredFallback,
    NoLayerAvailableError,
    classify_failure,
)
from .graph_guard import GraphGuardPolicy, GraphGuardReport, GraphGuardStorage, install_graph_guard
from .metering import LAYERS, ROLES, MeteringLedger, MeteringRecord
from .offline import BM25Index, DeterministicExtractive, SearchHit, chunks_from_hits, split_into_chunks
from .text import STOPWORDS, coverage, split_sentences, term_set, tokenize

__version__ = "0.1.0"

__all__ = [
    "ALLOW",
    "AnswerGuardPolicy",
    "AuditEntry",
    "AuditLog",
    "BM25Index",
    "BearerGuard",
    "CitationEnforcer",
    "ContractError",
    "DailyBudget",
    "DeterministicExtractive",
    "EXPECTED_FAILURE_MARKERS",
    "Evidence",
    "EvidenceGate",
    "ExposureError",
    "FallbackOutcome",
    "Finding",
    "GraphGuardPolicy",
    "GraphGuardReport",
    "GraphGuardStorage",
    "Guard",
    "GuardResult",
    "LAYERS",
    "LayerSpec",
    "LayeredFallback",
    "MeteringLedger",
    "MeteringRecord",
    "NoLayerAvailableError",
    "ROLES",
    "Reason",
    "STOPWORDS",
    "SearchHit",
    "Severity",
    "TokenBucketRateLimiter",
    "__version__",
    "audit_exposure",
    "chunks_from_hits",
    "classify_failure",
    "count_sources",
    "coverage",
    "deny",
    "enforce_exposure",
    "fingerprint",
    "install_graph_guard",
    "is_loopback",
    "normalize_evidence",
    "parse_whitelist",
    "path_is_whitelisted",
    "refusal_message",
    "split_into_chunks",
    "split_sentences",
    "term_set",
    "tokenize",
]
