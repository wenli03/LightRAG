# LightRAG 企业落地治理层（enterprise overlay）

本仓库是 [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG)（MIT，4 万星级别的开源图 RAG 框架）
的 **二次开发分支**。上游提供的是"从文档到知识图谱再到问答"的完整能力；
本分支补的是它**刻意没有做**的那一段：**从"能跑起来"到"能对外提供"之间**。

> 上游基线：提交 `453dce83`（2026-09-29 克隆）。所有改动都可以用
> `git diff upstream/main` 逐行看。

## 一句话说清"改了什么"

| 类别 | 内容 |
| --- | --- |
| **改到上游文件** | ① `pyproject.toml` 的包发现列表加上 `enterprise*`（不加就"本地能跑、装完 import 失败"）；② 启动自检（拦的是上游默认白名单把 Ollama 兼容路由 `/api/*` 放行这个安全问题） |
| **未改上游文件** | 检索、抽取、图谱构建、上下文拼装、WebUI、查询模式——全部原样使用 |
| **新增（自包含）** | `enterprise/` 包：暴露面加固、图谱写入护栏、证据门槛与引用强制、四级降级链、四角色成本归因、零依赖评测 |

接入方式只有三个挂载点，每一个都是上游的公开扩展面（详见
[`docs/adr/0002-overlay-not-deep-fork.md`](docs/adr/0002-overlay-not-deep-fork.md)）：

1. `rag.chunk_entity_relation_graph`（图谱存储对象）→ 套一层写入护栏代理；
2. `llm_model_func`（可调用对象）→ 换成带记账与降级的实现；
3. `QueryParam.only_need_context`（原生参数）→ **只取上下文、不调生成模型**，
   让"检索用上游"和"生成用我们的降级链"彻底解耦。

## 六件治理能力

| 能力 | 解决的问题 | 代码 |
| --- | --- | --- |
| **暴露面自检** | 上游默认 `WHITELIST_PATHS="/health,/api/*"` + 默认绑定 `0.0.0.0` = 无凭据可调用模型。警告不是防线，所以做成**会拒绝启动的检查** | `enterprise/auth.py` |
| **HTTP 语义正确** | 鉴权失败 **401**、配额拒绝 **429**、业务拒答仍 **200**。只写在响应体里的话，WAF / 监控 / 日志聚合全都看不见"有人在扫我的接口" | `enterprise/app.py` |
| **图谱写入护栏** | 向量库里的噪声是局部的，图谱里的无源关系是**全局**的——它会被后续所有多跳查询当成事实。因此"无来源不入图" | `enterprise/graph_guard.py` |
| **证据门槛拒答** | 用**关键词覆盖率**而不是检索绝对分做判据（绝对分会被常见字堆高、会随嵌入模型漂移） | `enterprise/answer_guard.py` |
| **引用强制** | 模型输出缺引用时**补上引用块并显式标注**，而不是静默删掉无依据的句子 | `enterprise/answer_guard.py` |
| **四级降级链** | 主模型 → 备模型 → 本地 → **确定性摘录**；**余额不足与内容拦截属可预期失败，直接降层不重试**（重试只烧额度和时间） | `enterprise/fallback.py` |
| **四角色成本归因** | 按上游的 `extract / keyword / query / vlm` 四角色分别记 token 与耗时；**刻意不内置单价**（价格会变，错的成本数字比没有更危险） | `enterprise/metering.py` |

## 快速开始

```bash
# 1. 零依赖：先跑离线评测（不需要模型、不需要网络）
python -m enterprise.eval.run_eval \
    --corpus data/corpus \
    --golden data/golden/golden.jsonl \
    --sweep 0.3,0.4,0.5,0.6,0.7,0.8 \
    --repeat 2 \
    --out data/eval_report.json

# 2. 只跑治理层测试（不依赖上游，可进 CI 门禁）
pytest tests/enterprise --confcutdir=tests/enterprise -m "not integration"

# 3. 起网关（默认离线检索后端，保证起得来）
export ENTERPRISE_API_KEY="$(python -c 'import secrets;print(secrets.token_urlsafe(32))')"
python -m enterprise.app

# 4. 起网关 + 真实上游（需要模型配置）
export ENTERPRISE_RETRIEVER=lightrag
export ENTERPRISE_PRIMARY_BASE_URL=https://api.deepseek.com/v1
export ENTERPRISE_PRIMARY_API_KEY=sk-...
export ENTERPRISE_PRIMARY_MODEL=deepseek-chat
python -m enterprise.app
```

启动时会先跑暴露面自检：**绑公网 + 没配 Key**、**绑公网 + 白名单里还留着 `/api/*`**、
**配了账号但 Token 密钥还是上游默认值**——这三条任一命中都会直接拒绝启动。

## 实测数字（可复跑）

由 `python -m enterprise.eval.run_eval` 产出，落在 `data/eval_report.json`：

| 项 | 值 | 口径 |
| --- | --- | --- |
| 治理层测试 | **143 / 143** | 其中 **126 条零上游依赖**（不装上游也能跑，因此可进 CI 门禁）；**16 条是 HTTP 级安全断言**：未带凭据一律 401、配额拒绝 429、免鉴权的只有 `/health` |
| 语料 | 7 份文档 / 102 个切片 | 本仓库自写并随仓库发布（MIT） |
| golden set | 22 条（18 有答案 + 4 无答案） | 含 2 条"提到相关名词但知识库里没有答案"的近似样本 |
| 阈值选择 | 覆盖率 0.4 | 规则：先要求无答案样本的**误答率为 0**，再在其中选**误拒率最低**的阈值 |
| Recall@1 / @3 / @5 | 1.0 / 1.0 / 1.0 | 文档级命中：top-K 里召回任一片段即算命中该文档 |
| MRR | 1.0 | 未命中的题计 0（不是跳过——跳过会让召回差时 MRR 反而好看） |
| 拒答准确性 / 误拒率 / 误答率 | 1.0 / 0.0 / 0.0 | 拒答准确性 = (有答案且没拒 + 无答案且拒了) / 总数 |
| 引用覆盖率 | 1.0 | 作答题里带引用编号的比例 |
| 两次运行一致性 | **逐字段一致** | 同参数跑两次，比较全部标量字段与逐条结果顺序 |

**这组数字的边界必须说清楚**：语料与 golden set 出自同一作者，
所以它衡量的是**链路正确性**（阈值能不能按规则选出来、两次跑是不是一样、
该拒的有没有拒），**不是泛化能力**。用别人的语料与别人的标注去测，数字一定会掉——
本仓库不打算用"自己出的卷子"冒充效果证明。

## 目录

```
enterprise/              二开治理层（自包含，除显式开关外不 import 上游）
├── contracts.py         统一契约：GuardResult / Reason / 来源计数 / 审计指纹
├── auth.py              暴露面自检 + Bearer 鉴权 + 令牌桶限流 + 日预算熔断
├── graph_guard.py       图谱写入护栏（存储代理 + 受控词表 + 拒绝落盘）
├── answer_guard.py      证据门槛（关键词覆盖率）+ 引用强制
├── fallback.py          四级降级链 + 失败分类（可预期失败不重试）
├── metering.py          四角色 token/耗时归因（不内置单价）
├── audit.py             JSONL 审计（只留指纹不留原文、按大小轮转）
├── text.py              中文 bigram 分词与切句（口径稳定、零依赖）
├── offline.py           零依赖 BM25 + 确定性摘录作答
├── embeddings.py        真实嵌入优先，失败退回本地确定性哈希向量
├── providers.py         OpenAI 兼容客户端（标准库实现，含不可信内容围栏）
├── retrievers.py        离线 / 上游 / 检索降级三种检索后端
├── lightrag_integration.py  上游装配（三个挂载点都在这里接）
├── app.py               治理网关（FastAPI）
└── eval/                golden set + 指标 + 阈值标定 + 两次运行一致性
docs/adr/                三条决策记录（扩展点清单 / 为什么 overlay / 为什么链尾必须确定性）
data/corpus/             自写语料（MIT）
data/golden/golden.jsonl golden set
tests/enterprise/        治理层测试（不依赖上游，可进 CI）
```

## 真实上游端到端结果（2026-09-29 实跑）

`scripts/lightrag_smoke.py` 真跑了「装配 → 入库 → 检索 → 护栏 → 记账」，
原始日志：`data/smoke.log`。要点：

| 观察项 | 实测值 | 说明 |
| --- | --- | --- |
| 四角色记账 | `extract` **2 次 / 6653 tokens**；`keyword` **2 次 / 1361 tokens** | 两个角色各记各的账。**第一次实现是共用一个包装函数，结果查询调用被记在 `extract` 名下**——按角色注入 `role_llm_configs` 之后才对 |
| 抽取结果 | **28 个实体 + 12 条关系** | 入库 1 篇文档（切成 1 个切片） |
| 图谱写入 | 尝试 **28 点 / 12 边**，护栏拒绝 **0** | 上游每个写入点都带 `source_id`，record 模式下照实写进图；日志 "Writing graph with 28 nodes, 12 edges" 与之对得上 |
| 观测到的实体类型 | `concept(13)` / `artifact(9)` / `UNKNOWN(2)` / `data(2)` / `content(1)` / `person(1)` | 即 `derive_vocabulary()` 给出的**经验性受控词表**候选——先观测后收口，不是拍脑袋 |
| 域内提问 | 覆盖率 **0.6** ≥ 阈值 0.4 → 放行，来源 `02-graph-rag-vs-vector-rag.md` | 检索到 1 个切片 |
| 域外提问（"上海今天适合穿什么衣服"） | **拒答**（`no_evidence`，0 切片） | 与离线评测的结论方向一致 |
| 嵌入 | 本地哈希向量（`embedding_dim=1024`） | 免费额度下嵌入接口返回 429，`EmbeddingService` 自动退回并记入 `embedding_choice.json` |
| 成本 | `priced: False` | 未注入单价，只报 token——这是刻意的 |

### 一个真实踩到的坑：并发池把免费额度打爆

按四个角色各建一个 worker 池之后，并发度变成「角色数 × 每池并发」。
实测这个并发度直接触发限流，抽取阶段撞上上游默认的 **480s worker 超时**——
任务**失败**，而且失败在"看起来还在跑"的中间态（日志里先是一堆
`found 4/5 fields on RELATION` 的格式告警，最后才是超时）。

把 `ENTERPRISE_LLM_MAX_ASYNC` 压到 **1** 之后稳定跑通。这不是"免费额度不靠谱"，
而是**给弱模型配并发**这件事本身需要按供给能力标定：并发不是越大越好，
它有一个"刚好把对方打到限流"的临界点，过了之后吞吐反而下降。

### 本机实测：演示面板（**一个模型都没配**）

`python -m enterprise.demo_ui` 起在 `127.0.0.1:7899`，用真实 HTTP 请求验：

| 请求 | 结果 |
| --- | --- |
| `GET /health` | 200；`exposure_findings: []`；`fallback_layers` 里 primary / secondary / local 三项 `configured: false`，deterministic `configured: true` |
| `GET /`（Gradio 演示面板，与网关同端口同源） | 200，页面标题含 LightRAG |
| `POST /v1/query` 不带凭据 | **401**，`reason: unauthorized` |
| `GET /v1/metering` 不带凭据 | **401**（三个只读接口共用的鉴权收口生效） |
| `POST /v1/query` 带凭据 | **200**：`layer: deterministic`、`degraded: true`、`coverage 0.5 ≥ 阈值 0.4`、5 条引用，答案首行是 `[无模型模式·确定性摘录]` |

最后一条是关键：这份响应是在**没有配置任何模型、也没有任何 Key** 的环境下拿到的。
也就是说"演示不会在面试现场挂掉"在这里是**可证伪的**，而不是一句承诺。
原始响应留档在 `data/demo_query.json` 与 `data/demo_health.json`。

### 只有真跑 / 真测才暴露的缺陷清单

这一节是留给"为什么必须做端到端和 HTTP 级测试"的答案。下面每一条的共同特征是：
**本地看起来全对、单元测试全绿，但真实使用路径是坏的。**

| 缺陷 | 为什么会漏过单测 | 现在由什么守住 |
| --- | --- | --- |
| 四角色记账被写死成一个角色（查询调用被记在 `extract` 名下） | 记账函数本身工作正常，只是拿错了角色名 | 逐角色注入 + 真实端到端日志里能看到 `extract` 与 `keyword` 各记各的 |
| `file_paths` 传裸字符串被下游按字符拆，引用来源显示成 `['d']` | 不报错，只是"引用来源"这一栏变成垃圾值 | `ingest()` 显式识别字符串并校验与 `texts` 等长 |
| 按角色拆并发池把免费额度打到限流，抽取撞上上游默认 480s worker 超时 | 单测不打网络；失败发生在"看起来还在跑"的中间态 | `ENTERPRISE_LLM_MAX_ASYNC` / `ENTERPRISE_LLM_TIMEOUT` 可配，压到 1 后稳定 |
| **`POST /v1/query` 一律返回 422** | 根因是 `from __future__ import annotations` 把 `request: Request` 变成字符串，而 `Request` 是在 `create_app()` 里**局部导入**的——FastAPI 只在**模块全局**解析注解，解析不到就把它当成必需的查询参数。进程内直接调 `gateway.answer()` 的测试全绿，HTTP 路径全坏 | `tests/enterprise/test_gateway_http.py` 走真实 HTTP 栈 |
| 镜像构建失败（`.dockerignore` 整体排除了 `/data` 与 `/tests`） | 只有真的 `docker build` 才会暴露 | 忽略规则改为 `/data/*` + 反向放行 `corpus`/`golden`；运行时镜像不再打包测试目录 |

## 已知问题与未验证项（如实标注）

- **关系抽取质量受模型能力限制**：上游对关系要求 5 个字段，`glm-4-flash` 经常只给 4 个
  （日志里大量 `found 4/5 fields on RELATION ...`），上游会要求模型补全，于是同一条关系被反复重试、
  抽取耗时被显著拉长。这是**模型侧的输出格式问题**，不是护栏误杀（护栏在 record 模式下对 28 个实体 0 拒绝）。
  换成能力更强的模型或收紧抽取提示词应可缓解，本仓库不声称已经解决。
- **抽取阶段的 worker 超时**：上游默认 480s。并发调高时实测撞上过（见上节），
  压到 `ENTERPRISE_LLM_MAX_ASYNC=1` 后稳定；更高并发是否稳定取决于 provider 的限流曲线，
  本仓库没有做系统性压测。
- **嵌入用的是本地哈希向量**：免费额度下嵌入接口返回 429。哈希向量只保证"同输入同输出"
  与字面词元落在相同维度，**不具备语义泛化能力**，因此**不能**用它来评估检索质量。
- **上游自带的 RAGAS 评测未跑**：它需要 `evaluation` extra（`ragas` / `datasets`）与真实模型额度。
- **嵌入后端切换后的索引一致性**：切换嵌入后端会让已有索引失效（不在同一向量空间）。
  当前做法是"降级只发生在建索引之前 + 把选择写进 `embedding_choice.json`"，
  但**没有实现自动重建**——换后端必须手工重建，这一点写在 `embeddings.py` 的注释里。
- **拦截模式在真实语料上的拒绝率未测**：默认 `record` 模式，实测 0 拒绝；
  切到 `reject` 后有多少比例被拦，取决于语料里到底有没有无源写入，目前没有样本。
- **公网 Demo 尚未创建**：本机没有目标平台的账号，`spaces/` 下只有可复制的部署件与检查清单。
  指向公网链接的字段在部署完成前**保持为空**，不写占位链接。
- **镜像未在本机完成端到端构建**：`registry-1.docker.io` 在本机网络不可达，改用镜像源后
  base image 能拉到，但 pip 层在容器网络里长时间停滞（实测 15 分钟无输出），因此**中止**。
  构建尝试本身是有价值的——它暴露并修掉了两处真实缺陷（`.dockerignore` 整体排除 `/data` 与 `/tests`
  导致 COPY 失败；pip 索引源与基础镜像未参数化则完全无法构建）。
  但请把「Dockerfile 能构建」当作**尚未验证**：目标平台的构建环境与包缓存与本机不同。
- **演示面板已在宿主环境实测**（见上一节的表格），但它跑在 `python -m enterprise.demo_ui` 直启路径上，
  **容器内**的运行未经本机验证。
- **上游 PR 未提交**：启动自检这一处修复具备提回上游的价值（它拦的是上游默认配置下的真实暴露面），
  但尚未提交，状态以仓库 Issue/PR 为准。

## 许可与致谢

上游 LightRAG 为 MIT 许可，本分支沿用同一许可。`data/corpus/` 下所有文档为**本项目作者原创撰写**，
随仓库以 MIT 发布，不包含任何第三方或内部资料。
