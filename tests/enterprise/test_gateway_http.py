"""网关的 HTTP 级断言：把「上线检查清单」变成会失败的测试。

为什么必须单独有这一层：`test_auth.py` 测的是**零件**（BearerGuard / 令牌桶 / 日预算）。
零件全对、装到 HTTP 层却没接上，是完全可能发生的事——最典型的是**新增一个只读接口时忘了挂鉴权**，
这种缺陷零件测试永远抓不到，而且不会报错，只会安静地多暴露一个端点。
所以这里只测"装好之后的样子"：状态码、响应体里的原因码，以及"免鉴权的只有 /health"。

另一条被这批测试固定下来的设计：**鉴权失败是 401、配额拒绝是 429，业务层拒答仍是 200**。
前两者是暴露面问题，必须让 WAF / 监控 / 日志聚合按状态码就能统计到；
后者是正常业务结果（接口工作正常，只是证据不足），混成 4xx 会把两件事搅在一起。
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from enterprise.app import GatewaySettings, create_app

ROOT = Path(__file__).resolve().parents[2]
KEY = "k" * 40
AUTH = {"Authorization": f"Bearer {KEY}"}


def _load_golden() -> list[dict[str, Any]]:
    path = ROOT / "data" / "golden" / "golden.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


GOLDEN = _load_golden()
IN_DOMAIN = next(item["question"] for item in GOLDEN if item.get("answerable"))
OUT_OF_DOMAIN = next(item["question"] for item in GOLDEN if not item.get("answerable"))


def _make_client(tmp_path: Path, **overrides: Any) -> Any:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from starlette.testclient import TestClient

    base: dict[str, Any] = {
        "api_key": KEY,
        "host": "127.0.0.1",
        "corpus_dir": str(ROOT / "data" / "corpus"),
        "audit_path": str(tmp_path / "audit.jsonl"),
        "rejects_path": str(tmp_path / "rejects.jsonl"),
        "retriever": "offline",
        "rate_per_minute": 600.0,
        "burst": 100,
        "daily_calls": 0,
    }
    base.update(overrides)  # 让调用方可以同时覆盖"默认值"和"会被覆盖的字段"（replace 不允许重复传参）
    return TestClient(create_app(dataclasses.replace(GatewaySettings(), **base)))


@pytest.fixture()
def client(tmp_path: Path) -> Any:
    return _make_client(tmp_path)


# ---------- 免鉴权的两个端点 ----------


def test_health_is_public_and_reports_no_exposure_findings(client: Any) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["exposure_findings"] == [], "启动自检留下的问题会在这里暴露出来"


def test_metrics_is_public_prometheus_text(client: Any) -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "lightrag_enterprise_calls_total" in response.text


# ---------- 鉴权：状态码必须是 401 ----------


@pytest.mark.parametrize(
    "headers",
    [
        pytest.param({}, id="no-credential"),
        pytest.param({"Authorization": "Bearer wrong"}, id="wrong-bearer"),
        pytest.param({"X-API-Key": "nope"}, id="wrong-api-key"),
    ],
)
def test_query_without_valid_credential_is_401(client: Any, headers: dict[str, str]) -> None:
    response = client.post("/v1/query", json={"q": IN_DOMAIN}, headers=headers)
    assert response.status_code == 401, "鉴权失败必须是 401：只写在响应体里的话，WAF 与监控都看不见"
    assert response.json()["reason"] == "unauthorized"


def test_upstream_default_whitelist_path_is_blocked(client: Any) -> None:
    """上游默认放行 `/api/*`（Ollama 兼容路由）；网关这一侧必须拦住它。"""
    response = client.post("/api/tags", json={"q": IN_DOMAIN})
    assert response.status_code in (401, 404, 405)


@pytest.mark.parametrize("path", ["/v1/metering", "/v1/guard/graph", "/v1/audit/tail"])
def test_every_readonly_endpoint_requires_a_credential(client: Any, path: str) -> None:
    """「新增接口忘了挂鉴权」的守门员：任何一个只读接口漏鉴权，这里立刻红。"""
    assert client.get(path).status_code == 401
    assert client.get(path, headers=AUTH).status_code == 200


# ---------- 配额：状态码必须是 429 ----------


def test_rate_limit_returns_429(tmp_path: Path) -> None:
    client = _make_client(tmp_path, rate_per_minute=60.0, burst=1)
    assert client.post("/v1/query", json={"q": IN_DOMAIN}, headers=AUTH).status_code == 200
    limited = client.post("/v1/query", json={"q": IN_DOMAIN}, headers=AUTH)
    assert limited.status_code == 429
    assert limited.json()["reason"] == "rate_limited"


def test_daily_budget_returns_429_after_limit(tmp_path: Path) -> None:
    client = _make_client(tmp_path, daily_calls=1)
    assert client.post("/v1/query", json={"q": IN_DOMAIN}, headers=AUTH).status_code == 200
    exhausted = client.post("/v1/query", json={"q": IN_DOMAIN}, headers=AUTH)
    assert exhausted.status_code == 429
    assert exhausted.json()["reason"] == "budget_exceeded"


# ---------- 业务结果：仍然是 200 ----------


def test_in_domain_question_is_answered(client: Any) -> None:
    body = client.post("/v1/query", json={"q": IN_DOMAIN}, headers=AUTH).json()
    assert body["refused"] is False
    assert body["answer"]
    assert body["detail"]["coverage"] >= body["detail"]["threshold"]


def test_out_of_domain_question_is_refused_with_http_200(client: Any) -> None:
    response = client.post("/v1/query", json={"q": OUT_OF_DOMAIN}, headers=AUTH)
    assert response.status_code == 200, "业务层拒答不是暴露面拒绝，不能混成 4xx"
    body = response.json()
    assert body["refused"] is True
    # 不写死具体码：一句域外提问可能"检索不到任何片段"（no_evidence），
    # 也可能"检索到片段但覆盖率不够"（low_coverage）。要守的不变量是"它拒答了，
    # 且原因是证据类原因"，而不是"恰好走了哪一条分支"——写死具体码的断言
    # 在语料一改就会红，而那种红并不代表代码坏了。
    assert body["reason"] in {"no_evidence", "low_coverage"}


def test_empty_question_is_business_refusal_not_4xx(client: Any) -> None:
    response = client.post("/v1/query", json={"q": "   "}, headers=AUTH)
    assert response.status_code == 200
    assert response.json()["reason"] == "empty_question"


# ---------- 「演示不会挂」的可证伪版本 ----------


def _clear_model_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for prefix in ("ENTERPRISE_PRIMARY_", "ENTERPRISE_SECONDARY_", "ENTERPRISE_LOCAL_"):
        for suffix in ("API_KEY", "BASE_URL", "MODEL"):
            monkeypatch.delenv(prefix + suffix, raising=False)


def test_gateway_answers_with_no_model_configured_at_all(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """一个模型都没配、也没有任何 Key 时，全流程仍然走得完，并且**自己说是哪一层在答**。

    这是「演示不会在面试现场挂掉」的可证伪版本——不是承诺，是断言。
    """
    _clear_model_env(monkeypatch)
    client = _make_client(tmp_path)
    body = client.post("/v1/query", json={"q": IN_DOMAIN}, headers=AUTH).json()
    assert body["refused"] is False
    assert body["layer"] == "deterministic"
    assert body["degraded"] is True
    assert "无模型模式" in body["answer"], "降级必须对使用者可见：不标注的降级会被读成「模型答得真差」"


def test_audit_tail_records_verdicts_without_storing_question_text(tmp_path: Path) -> None:
    """审计只留指纹不留原文：这条不变量没有报错，只能靠断言守。"""
    client = _make_client(tmp_path)
    client.post("/v1/query", json={"q": IN_DOMAIN}, headers=AUTH)
    rows = client.get("/v1/audit/tail", headers=AUTH).json()
    assert rows, "审计应该有记录"
    joined = json.dumps(rows, ensure_ascii=False)
    assert IN_DOMAIN not in joined, "审计里不应出现问题原文"
