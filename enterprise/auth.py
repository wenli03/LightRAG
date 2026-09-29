"""公网暴露面加固：启动自检 + 强制鉴权 + 限流 + 日预算熔断。

为什么有启动自检这一层：上游 ``lightrag/api/config.py`` 的 ``WHITELIST_PATHS`` 默认值是
``"/health,/api/*"``，而 ``/api/*`` 是 **Ollama 兼容路由**——也就是说，按默认配置把服务绑到
0.0.0.0 时，任何人不需要任何凭据就能调用模型。上游 README 自己写了这条警告，
但**警告不等于防线**。所以这里把它做成一个会失败的启动检查：
"绑到非回环地址 + 没配 API Key"、"绑到非回环地址 + 白名单里还留着 /api/*"，都必须拒绝启动。

这一条同时也是我们准备提给上游的最小改动（见 docs/adr/0001-extension-points.md §上游改动）。
"""

from __future__ import annotations

import dataclasses
import hmac
import ipaddress
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from enum import Enum
from typing import Any

from .contracts import GuardResult, Reason, deny

# 与上游同值，仅用于比对"有没有被改成安全的值"；不 import 上游以保持零依赖。
UPSTREAM_DEFAULT_WHITELIST = "/health,/api/*"
UPSTREAM_DEFAULT_TOKEN_SECRET = "lightrag-jwt-default-secret-key!"
OLLAMA_COMPAT_PATTERN = "/api/*"


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclasses.dataclass(frozen=True)
class Finding:
    severity: Severity
    code: str
    message: str
    hint: str = ""

    def to_dict(self) -> dict[str, str]:
        return dataclasses.asdict(self)


def is_loopback(host: str | None) -> bool:
    """只有真正只在本机可达的绑定地址才算"没暴露"。

    ``0.0.0.0`` / ``::`` 是"监听所有网卡"，是最常见的**误判成"本地"**的写法，
    所以它们必须返回 False。
    """
    if not host:
        return False
    host = host.strip().strip("[]")
    if host in ("localhost",):
        return True
    if host in ("0.0.0.0", "::", "*"):
        return False
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False  # 域名等一律按"可能是外部可达"处理，宁可误报


def parse_whitelist(raw: str | Sequence[str] | None) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [item.strip() for item in raw.split(",") if item.strip()]
    return [str(item).strip() for item in raw if str(item).strip()]


def _pattern_hits(pattern: str, path: str) -> bool:
    if pattern.endswith("*"):
        return path.startswith(pattern[:-1])
    return path == pattern


def path_is_whitelisted(path: str, whitelist: Sequence[str]) -> bool:
    return any(_pattern_hits(pattern, path) for pattern in whitelist)


def audit_exposure(
    *,
    host: str | None,
    api_key: str | None,
    whitelist_paths: str | Sequence[str] | None,
    auth_accounts: str | None = None,
    token_secret: str | None = None,
    cors_origins: str | None = None,
    min_api_key_length: int = 24,
) -> list[Finding]:
    """返回所有暴露面问题。**纯函数**——不读环境、不抛异常，方便单测穷举组合。"""
    whitelist = parse_whitelist(whitelist_paths)
    findings: list[Finding] = []
    public = not is_loopback(host)

    if public and not (api_key or "").strip():
        findings.append(
            Finding(
                Severity.ERROR,
                "public_without_api_key",
                f"服务绑定到 {host!r}（非回环地址）但没有配置 LIGHTRAG_API_KEY，任何人都能读写知识库",
                "设置 LIGHTRAG_API_KEY（>=24 位随机串），或改为绑定 127.0.0.1 由反向代理对外",
            )
        )

    if public and any(_pattern_hits(OLLAMA_COMPAT_PATTERN, p) or p == OLLAMA_COMPAT_PATTERN for p in whitelist):
        findings.append(
            Finding(
                Severity.ERROR,
                "ollama_paths_whitelisted_on_public_host",
                f"白名单里包含 {OLLAMA_COMPAT_PATTERN}（Ollama 兼容路由），而这些路由会绕过鉴权，"
                f"绑定到 {host!r} 等于把模型额度公开出去",
                "把 WHITELIST_PATHS 收敛为 /health（或 /health,/metrics），需要 /api/* 时再加独立网关鉴权",
            )
        )

    if (auth_accounts or "").strip() and (not token_secret or token_secret == UPSTREAM_DEFAULT_TOKEN_SECRET):
        findings.append(
            Finding(
                Severity.ERROR,
                "default_token_secret",
                "配置了 AUTH_ACCOUNTS 但 TOKEN_SECRET 仍是上游默认值，签发出来的 token 可被任何人伪造",
                "设置一个足够随机的 TOKEN_SECRET",
            )
        )

    key = (api_key or "").strip()
    if key and len(key) < min_api_key_length:
        findings.append(
            Finding(
                Severity.WARNING,
                "short_api_key",
                f"LIGHTRAG_API_KEY 长度只有 {len(key)}，低于建议的 {min_api_key_length}",
                "用 `python -c \"import secrets;print(secrets.token_urlsafe(32))\"` 生成",
            )
        )

    if public and cors_origins and "*" in cors_origins:
        findings.append(
            Finding(
                Severity.WARNING,
                "wildcard_cors",
                "CORS_ORIGINS 为 *，任意站点都能在浏览器里带上用户凭据调用本服务",
                "收敛到实际前端域名；公开 Demo 至少限制为只读接口",
            )
        )

    return findings


class ExposureError(RuntimeError):
    """启动自检发现 ERROR 级问题。刻意不提供"忽略"开关：能被忽略的检查等于没有检查。"""


def enforce_exposure(**kwargs: Any) -> list[Finding]:
    findings = audit_exposure(**kwargs)
    errors = [f for f in findings if f.severity is Severity.ERROR]
    if errors:
        joined = "\n".join(f"  - [{f.code}] {f.message}\n    → {f.hint}" for f in errors)
        raise ExposureError(f"启动自检未通过（{len(errors)} 个错误）：\n{joined}")
    return findings


class BearerGuard:
    """强制鉴权。未配置 Key 时默认"仅在回环地址上允许匿名"。

    ``allow_anonymous_when_unset=False`` 是本项目在公网 Demo 上使用的设置：
    宁可服务起不来，也不要起成一个裸奔的服务。
    """

    def __init__(
        self,
        api_key: str | None,
        *,
        whitelist: Sequence[str] = ("/health",),
        allow_anonymous_when_unset: bool = False,
        bearer_header: str = "authorization",
        api_key_header: str = "x-api-key",
    ) -> None:
        self._api_key = (api_key or "").strip()
        self._whitelist = list(whitelist)
        self._allow_anonymous = allow_anonymous_when_unset
        self._bearer_header = bearer_header.lower()
        self._api_key_header = api_key_header.lower()

    @property
    def configured(self) -> bool:
        return bool(self._api_key)

    def check(self, payload: Mapping[str, Any]) -> GuardResult:
        path = str(payload.get("path", "/"))
        headers = {str(k).lower(): str(v) for k, v in dict(payload.get("headers") or {}).items()}

        if path_is_whitelisted(path, self._whitelist):
            return GuardResult(allowed=True, detail={"reason": "whitelisted", "path": path})
        if not self._api_key:
            if self._allow_anonymous:
                return GuardResult(allowed=True, detail={"reason": "anonymous_allowed", "path": path})
            return deny(Reason.UNAUTHORIZED, detail="服务未配置 LIGHTRAG_API_KEY，公网模式下拒绝匿名访问", path=path)

        presented = ""
        header_value = headers.get(self._bearer_header, "")
        if header_value.lower().startswith("bearer "):
            presented = header_value[7:].strip()
        elif headers.get(self._api_key_header):
            presented = headers[self._api_key_header].strip()

        # compare_digest 而不是 ==：避免用响应时间把 Key 一位一位试出来。
        if presented and hmac.compare_digest(presented, self._api_key):
            return GuardResult(allowed=True, detail={"path": path})
        return deny(Reason.UNAUTHORIZED, detail="缺少或错误的凭据", path=path)


class TokenBucketRateLimiter:
    """每客户端一个令牌桶。够用且零依赖；不追求分布式精度（单实例演示服务）。"""

    def __init__(
        self,
        *,
        rate_per_minute: float = 20.0,
        burst: int = 5,
        clock: Callable[[], float] = time.monotonic,
        max_clients: int = 4096,
    ) -> None:
        self._rate = max(rate_per_minute, 0.0) / 60.0
        self._burst = max(burst, 1)
        self._clock = clock
        self._max_clients = max_clients
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, *, cost: float = 1.0) -> bool:
        if self._rate <= 0:
            return True  # 0 = 关闭限流（本地开发）
        now = self._clock()
        with self._lock:
            tokens, last = self._buckets.get(key, (float(self._burst), now))
            tokens = min(float(self._burst), tokens + (now - last) * self._rate)
            if tokens < cost:
                self._buckets[key] = (tokens, now)
                return False
            self._buckets[key] = (tokens - cost, now)
            if len(self._buckets) > self._max_clients:
                self._prune(now)
        return True

    def _prune(self, now: float) -> None:
        stale = [k for k, (_, last) in self._buckets.items() if now - last > 3600]
        for key in stale:
            self._buckets.pop(key, None)


class DailyBudget:
    """日调用上限。公网 Demo 的必备件：没有它，一个爬虫就能把额度刷完。"""

    def __init__(self, *, daily_calls: int = 0, clock: Callable[[], float] = time.time) -> None:
        self._limit = max(0, daily_calls)
        self._clock = clock
        self._used = 0
        self._day = self._today()
        self._lock = threading.Lock()

    def _today(self) -> int:
        return int(self._clock() // 86400)

    def _rollover(self) -> None:
        today = self._today()
        if today != self._day:
            self._day = today
            self._used = 0

    @property
    def enabled(self) -> bool:
        return self._limit > 0

    def allow(self) -> bool:
        if not self.enabled:
            return True
        with self._lock:
            self._rollover()
            return self._used < self._limit

    def consume(self, amount: int = 1) -> bool:
        """先消费后调用。返回是否消费成功（False = 已超预算，调用方必须拒绝）。"""
        if not self.enabled:
            return True
        with self._lock:
            self._rollover()
            if self._used + amount > self._limit:
                return False
            self._used += amount
            return True

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            self._rollover()
            return {"enabled": self.enabled, "limit": self._limit, "used": self._used, "remaining": max(0, self._limit - self._used)}
