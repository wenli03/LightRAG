---
title: LightRAG 企业治理层演示
emoji: 🛡️
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# LightRAG 企业落地治理层 · 演示

上游 [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG)（MIT）负责检索与知识图谱；
本 Space 演示的是加在其上的**治理层**：证据门槛拒答、引用强制、图谱写入护栏、
四级降级链、四角色成本归因。

## 部署方式

把本仓库的 `Dockerfile.enterprise` 作为 Space 的 Dockerfile 使用：

```bash
# 在 Space 仓库里
cp ../../Dockerfile.enterprise ./Dockerfile
cp -r ../../enterprise ./
cp -r ../../data ./
cp -r ../../lightrag ./
cp ../../pyproject.toml ../../setup.py ../../LICENSE ./
```

然后在 Space 的 **Settings → Variables and secrets** 里配置：

| 名称 | 类型 | 值 |
| --- | --- | --- |
| `ENTERPRISE_API_KEY` | secret | `python -c "import secrets;print(secrets.token_urlsafe(32))"` 生成 |
| `ENTERPRISE_HOST` | variable | `0.0.0.0` |
| `ENTERPRISE_PORT` | variable | `7860` |
| `ENTERPRISE_WHITELIST_PATHS` | variable | `/health` |
| `ENTERPRISE_DAILY_CALLS` | variable | `200` |

> ⚠️ **未实测**：`huggingface.co` 在本机网络下超时不可达（`hf-mirror.com` 可达），
> 因此本 Space 的创建流程**没有实际执行过**。上面的 frontmatter 与 Secret 名称按平台的通行约定书写，
> 请以平台当前文档为准。容器内的行为（启动自检、鉴权、限流、降级）与平台无关，是实测过的。

## 为什么不会"演示到一半挂掉"

降级链的末级是**确定性摘录**：不调模型、不联网，直接从上一步检索到的原文里按词元重叠摘句子，
并强制带引用。所以断网、没 Key、额度用完时，演示仍然走得完全流程，
输出第一行会明确标注 `[无模型模式·确定性摘录]`。
