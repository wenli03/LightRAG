# ADR 0001 · 扩展点核实清单（先读源码，再写代码）

- 状态：已接受
- 日期：2026-09-29
- 上游基线：`HKUDS/LightRAG` 提交 `453dce83`（2026-09-29 克隆，`main` 分支）

## 背景

在别人的代码库上做二次开发，最容易犯的错误是**凭记忆写上游 API**。
一旦函数名或参数名记错，写出来的东西不是"报错"，而是"看起来跑通了但实际没接上"——
比如把护栏套在了一个没人调用的方法上，测试全绿、日志正常、数据照样进去。

因此本方案的第一步不是写代码，而是**逐条核实扩展点**，产出本清单。
下面每一条都给出文件路径、行号与真实符号名；核实不到的项，写在最后。

## 1. 服务端与中间件

| 项 | 事实（路径:行号 —— 符号） |
| --- | --- |
| 应用工厂 | `lightrag/api/lightrag_server.py:1436` —— `create_app(args)` |
| 中间件挂载方式 | 全部用 `app.add_middleware(...)`，**没有** `@app.middleware("http")` 装饰器写法 |
| 根路径归一化 | `lightrag/api/lightrag_server.py:427` `_RootPathNormalizationMiddleware`，挂载在 `:1797` |
| 准入控制 | `lightrag/api/admission_middleware.py:52` `AdmissionMiddleware`，挂载在 `:1811-1817` |
| 请求体上限 | `lightrag/api/body_limit_middleware.py:41` `BodyLimitMiddleware`，挂载在 `:1826-1827` |
| CORS | `lightrag/api/lightrag_server.py:1777` `get_cors_origins()`，挂载在 `:1847`；默认值 `lightrag/api/config.py:848`（`CORS_ORIGINS` 默认 `*`） |

**关键结论**：鉴权**不是中间件**，而是 FastAPI 依赖 `get_combined_auth_dependency(...)`
（`lightrag/api/utils_api.py:392`）。所以"加一道中间件把所有请求拦住"不是最小改动路径；
最小路径是**在外层再加一个网关**，这也决定了本项目的整体形态（见 ADR 0002）。

## 2. 鉴权、白名单与上游的那条默认值

| 项 | 事实 |
| --- | --- |
| API Key 读取 | `lightrag/api/config.py:443`（`LIGHTRAG_API_KEY`）；生效值 `lightrag/api/lightrag_server.py:1635` |
| AUTH_ACCOUNTS | 读取 `lightrag/api/config.py:876`；校验 `lightrag/api/config.py:166-173` `validate_auth_configuration(args)` |
| Token 密钥默认值 | `lightrag/api/config.py:65` `DEFAULT_TOKEN_SECRET = "lightrag-jwt-default-secret-key!"` |
| **白名单默认值** | `lightrag/api/config.py:850` —— `WHITELIST_PATHS` 默认 **`"/health,/api/*"`** |
| 白名单解析 | `lightrag/api/utils_api.py:258-271` |
| 白名单判定 | `lightrag/api/utils_api.py:321` `path_is_whitelisted(scope, *, mount_prefix="")` |
| 实际放行点 | `lightrag/api/utils_api.py:441`（在 `combined_dependency` 内） |
| 凭据判定 | `lightrag/api/utils_api.py:353` `credentials_accepted(...)`；定义 `:377-389` |

**这就是我们要补的第一个缺口**：`/api/*` 是 Ollama 兼容路由，默认在白名单里，
而默认监听地址是 `0.0.0.0`（`lightrag/api/config.py:364`）。
两者的组合意味着"默认配置 + 上公网 = 无凭据可调用模型"。
上游 README 已经写了这条警告，但**警告不是防线**。
对应实现：`enterprise/auth.py` 的 `audit_exposure` / `enforce_exposure`。

## 3. 查询链路

| 项 | 事实 |
| --- | --- |
| REST 端点 | `lightrag/api/routers/query_routes.py:376` —— `POST /query`（另有 `/query/stream`:745、`/query/data`:1114） |
| 端点内调用 | `lightrag/api/routers/query_routes.py:602` —— `await rag.aquery_llm(request.query, param=param)` |
| 库入口 | `lightrag/lightrag.py:5114` `aquery_llm(...)`；仅取数据的 `aquery_data` 在 `:4901` |
| 图谱查询 | `lightrag/operate.py:4693` `kg_query(...)` |
| 朴素查询 | `lightrag/operate.py:6827` `naive_query(...)` |
| `QueryParam` | `lightrag/base.py:90`；关键字段 `mode`(:93)、`only_need_context`(:102)、`top_k`(:114)、`chunk_top_k`(:117)、`include_references`(:171) |

**关键结论**：`QueryParam.only_need_context` 存在。
这让我们可以**只取上下文、不调生成模型**，从而把"检索用上游"和"生成用我们自己的降级链"
彻底解耦。对应实现：`enterprise/retrievers.py::LightRAGRetriever`。

## 4. 检索结果的结构（护栏要吃的东西）

| 项 | 事实 |
| --- | --- |
| 用户态结构生成 | `lightrag/utils.py:7611` `convert_to_user_format(...)`，返回体见 `:7722-7732` |
| 四个数据键 | `data.entities` / `data.relationships` / `data.chunks` / `data.references` |
| chunk 字段 | `lightrag/utils.py:7700-7707` —— `reference_id` / `content` / `file_path` / `chunk_id` |
| 实体字段 | `lightrag/utils.py:7634-7642`（含 `source_id` / `file_path`） |
| 关系字段 | `lightrag/utils.py:7671-7681`（含 `source_id` / `file_path` / `keywords` / `weight`） |
| 引用列表 | `lightrag/utils.py:7735` `generate_reference_list_from_chunks(chunks)`，产出 `{"reference_id","file_path"}` |
| 上下文拼装 | `lightrag/operate.py:5873` `_build_context_str(...)`；最终格式化 `:6047-6052` |

**关键结论**：证据包里**同时**有原文切片与来源文件，所以"引用强制"不需要额外查询就能做。

## 5. 图谱写入路径（护栏要套的地方）

| 项 | 事实 |
| --- | --- |
| 实体合并写入 | `lightrag/operate.py:2429` `_merge_nodes_then_upsert(...)` |
| 关系合并写入 | `lightrag/operate.py:2782` `_merge_edges_then_upsert(...)` |
| 抽取流水线收口 | `lightrag/operate.py:3514` `merge_nodes_and_edges(...)`（调用点 `:3751` / `:3841`） |
| **存储层唯一收口** | `lightrag/base.py:958-959` `upsert_node`（抽象方法）；`:1054-1055` `upsert_edge` |
| 批量写入 | `lightrag/base.py:1007` `upsert_nodes_batch`；`:1039` `upsert_edges_batch`（默认实现逐条调用，**各后端可覆盖为原生批量**） |
| 存储初始化位置 | `lightrag/lightrag.py:1868` —— `self.chunk_entity_relation_graph = self.graph_storage_cls(...)`，位于 `__post_init__`（`:1638`） |
| 来源字段真名 | `source_id`；分隔符 `lightrag/constants.py:49` `GRAPH_FIELD_SEP = "<SEP>"` |
| 属性契约校验点 | `lightrag/utils.py:8178` `validate_graph_attributes(attributes, *, context)` |

**两个关键结论**：

1. **`operate.py` 层面没有唯一写入收口**：除 `merge_nodes_and_edges` 外，rebuild 路径
   （`operate.py:1817` / `:2286` / `:2337`）会**绕过合并函数直接写**。
   所以护栏如果挂在 `merge_nodes_and_edges` 上，rebuild 路径就会漏过去。
   → 决定：套在**存储对象**上（`rag.chunk_entity_relation_graph`），它是跨路径的唯一收口。
2. **批量方法必须自己过滤**：上游默认实现逐条调到 `upsert_node`，但后端可以覆盖成原生批量写。
   把批量方法原样透传 = 护栏被整批绕过。测试里有一条负向用例专门守这一点
   （`tests/enterprise/test_graph_guard.py::test_batch_nodes_are_filtered_not_passed_through`）。

## 6. LLM 调用与四角色

| 项 | 事实 |
| --- | --- |
| 角色定义 | `lightrag/llm_roles.py:52-57` —— `ROLES`：`extract` / `keyword` / `query` / `vlm` |
| 候选列表 | `:58` `ROLE_NAMES`；`:59` `ROLES_BY_NAME`；运行时装配 `lightrag/lightrag.py:1611` |
| 查询调用签名 | `lightrag/operate.py:4907` —— `await use_model_func(user_query, system_prompt=..., history_messages=..., enable_cot=True, stream=...)` |
| 关键词调用 | `lightrag/operate.py:5178` —— `await use_model_func(kw_prompt, response_format={"type": "json_object"})` |
| 实例参数 | `LightRAG` 是 dataclass（**没有手写 `__init__`**）：`llm_model_func`(:995)、`embedding_func`(:933)、`role_llm_configs`(:998)、`graph_storage`(:702) |
| provider 实现 | `lightrag/llm/openai.py`（OpenAI 兼容）、`lightrag/llm/ollama.py` 等 |

**关键结论**：`llm_model_func` 只是一个可替换的可调用对象，参数就是
`prompt / system_prompt / history_messages`。因此记账与降级可以完全在**我们自己的包装里**做，
不需要改上游一行。对应实现：`enterprise/lightrag_integration.py::build_llm_func`。

## 7. 评测

| 项 | 事实 |
| --- | --- |
| RAGAS 脚本 | `lightrag/evaluation/eval_rag_quality.py`，`from ragas import evaluate` 在 `:85` |
| 离线词法核查 | `lightrag/evaluation/offline_retrieval_check.py`（不启动 LightRAG/API/embedding） |
| 样例语料 | `lightrag/evaluation/sample_documents/`（5 篇 md + README） |
| 依赖声明 | `pyproject.toml:186-190` —— `evaluation` extra 需要 `ragas>=0.3.7` `datasets>=4.3.0` |

**关键结论**：上游的评测入口是**可选的 extra**，且 RAGAS 需要真实模型与网络。
本项目因此**自建了一套零依赖评测**（`enterprise/eval/`），
并在文档里明确区分"零模型离线评测"与"需模型端到端评测"两种口径——见 ADR 0003。

## 8. 配置与打包

| 项 | 事实 |
| --- | --- |
| 入口点 | `pyproject.toml:199` —— `lightrag-server = "lightrag.api.lightrag_server:main"` |
| 包发现 | `pyproject.toml:212-214` —— `[tool.setuptools.packages.find] include = ["lightrag*"]` |
| 部署参数 | `lightrag/api/config.py` —— `--host`(:364 默认 `0.0.0.0`)、`--port`(:369 默认 9621)、`--working-dir`(:377 默认 `./rag_storage`)、`LIGHTRAG_DEFAULT_UI`(:504 默认 `webui`) |
| 镜像 | `Dockerfile.lite`（只装 `api` extra）与 `Dockerfile`（另装 spaCy 模型与 `libcairo2`），基础镜像、`CMD`、`EXPOSE 9621` 相同 |
| Python 要求 | `pyproject.toml:14` —— `requires-python = ">=3.10"` |

**唯一的打包改动**：`include` 需要加上 `enterprise*`，否则 `pip install .` 之后
`import enterprise` 会失败——这是一个"本地跑得通、装完就挂"的典型陷阱。

## 9. 测试约定

| 项 | 事实 |
| --- | --- |
| 框架 | `pyproject.toml:57-64` —— `pytest>=8.4.2` / `pytest-asyncio>=1.2.0` / `pytest-xdist>=3.6.0` |
| 配置 | `pyproject.toml:225-246` —— `asyncio_mode="auto"`、`testpaths=["tests"]`、`python_files=["test_*.py"]` |
| 标记 | `pyproject.toml:240-246` —— `offline` / `integration` / `requires_db` / `requires_api` / `pg_smoke`；`integration` 默认跳过 |
| 运行方式 | `scripts/test.sh`（`Makefile` **没有** test 目标） |

**结论**：本项目的测试放在 `tests/enterprise/`，用 `offline` 语义（零上游依赖）书写，
并用 `--confcutdir=tests/enterprise` 避免被上游 `tests/conftest.py` 的夹具拖累。

## 未能确认 / 不存在

- 上游**没有** `@app.middleware("http")` 装饰器写法（只有 `add_middleware`）。
- 上游 `LightRAG` **没有**手写 `__init__`（dataclass 自动生成）。
- `operate.py` 层面**不存在**覆盖所有写入路径的唯一收口函数（见 §5）。
- `Makefile` **没有** test 目标。
