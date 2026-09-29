# 公网演示部署

目标只有一个：**给面试官一个点开就能用的链接**，而且这个链接不能依赖本机环境、不能依赖余额、
不能依赖第三方的可用性。

## 平台选择（2026-09-29 实测连通性）

| 平台 | 实测结果 | 结论 |
| --- | --- | --- |
| **ModelScope 魔搭「创空间」** | `https://www.modelscope.cn` 返回 200，**国内直连可达**；官方页面写明「CPU 资源长期免费」「创空间免费部署托管应用」，需注册并绑定阿里云账号 | ✅ **主平台** |
| Hugging Face Spaces | `https://huggingface.co` **超时不可达**；`https://hf-mirror.com` 返回 200 | ⚠️ 备份（需要自备网络） |
| 任意 Docker 主机 / 内网机器 | 把 `Dockerfile.enterprise` 构建出来即可 | ✅ 兜底 |

**为什么主选创空间**：面试官在中国，点开链接就能看到；而且它给的是"CPU 长期免费"，
这点直接决定了这套演示的**成本是 0**，不构成任何持续负担。

> ⚠️ 未实测项：本仓库**没有实际执行过创空间的创建流程**，因此不写具体的配置文件格式。
> 平台侧的创建步骤请以平台当前文档为准；本仓库保证的是**容器内的行为**是可复现的
> （`Dockerfile.enterprise` + 下面这组环境变量），平台差异只影响"怎么把容器跑起来"。

## 必须配置的环境变量

| 变量 | 是否必需 | 说明 |
| --- | --- | --- |
| `ENTERPRISE_API_KEY` | **必需** | 公网绑定但没有它会**直接拒绝启动**（这是设计，不是故障）。用 `python -c "import secrets;print(secrets.token_urlsafe(32))"` 生成，填在平台的 Secret 里 |
| `ENTERPRISE_HOST` | 必需 | `0.0.0.0` |
| `ENTERPRISE_PORT` | 必需 | 与平台暴露的端口一致（示例用 `7860`） |
| `ENTERPRISE_WHITELIST_PATHS` | 必需 | 只放 `/health`。**不要把 `/api/*` 加回来**——那是上游的 Ollama 兼容路由，会绕过鉴权 |
| `ENTERPRISE_DAILY_CALLS` | 建议 `200` | 日调用上限。没有它，一个爬虫就能把额度刷完，而服务不会因此报错，只会安静地花钱 |
| `ENTERPRISE_RATE_PER_MINUTE` / `ENTERPRISE_BURST` | 建议 `20` / `5` | 每客户端令牌桶 |
| `ENTERPRISE_RETRIEVER` | `offline` 或 `lightrag` | 默认 `offline`（零外部依赖，恒可用）；配好模型后可切 `lightrag` |
| `ENTERPRISE_PRIMARY_BASE_URL/_API_KEY/_MODEL` | 可选 | 配了就有真实模型作答；不配就自动走确定性摘录并**在输出第一行标注** |
| `ENTERPRISE_EMBEDDING_BASE_URL/_API_KEY/_MODEL` | 可选 | 配了用真实嵌入；不配或不可用则退回本地哈希向量（会写进 `embedding_choice.json`） |

## 构建与启动

```bash
docker build -f Dockerfile.enterprise -t lightrag-enterprise-demo .
docker run --rm -p 7860:7860 \
    -e ENTERPRISE_API_KEY="$ENTERPRISE_API_KEY" \
    -e ENTERPRISE_HOST=0.0.0.0 \
    -e ENTERPRISE_PORT=7860 \
    lightrag-enterprise-demo
```

打开 `http://localhost:7860` 是演示面板；`/health`、`/metrics` 与 `/v1/query`（需 Bearer）
是同端口下的 REST 接口。

## 上线前检查（逐条可验证）

- [ ] `ENTERPRISE_API_KEY` 已设置为足够长的随机串（≥ 32 位）
- [ ] `ENTERPRISE_WHITELIST_PATHS` 里**没有** `/api/*`
- [ ] 不带 `Authorization` 直接访问 `/v1/query`，**返回 401**（不是答案）
- [ ] 连发超过 `burst` 次请求，**被限流**（返回 `rate_limited`）
- [ ] `ENTERPRISE_DAILY_CALLS` 生效（把值临时设成 1，第二次请求被 `budget_exceeded` 拒绝）
- [ ] `/health` 能返回 `exposure_findings: []`（没有暴露面问题）
- [ ] **断网/不配模型**时仍然能完成一次问答，且输出第一行带 `[无模型模式…]` 标注
- [ ] 截图与录屏里**不出现任何密钥**（面板不回显密钥，但录屏前仍应扫一眼）

## 关于"能不能演示"这件事

这套演示的可用性不依赖模型：链尾是确定性的。所以即使额度用完、模型下线、网络中断，
面试官点开链接仍然能看到：检索、证据门槛判定、拒答、引用块、覆盖率与降级层级。
**"演示在关键时刻挂掉"是这类项目最致命的失败方式**，本设计就是为了消掉它。
