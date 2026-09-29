"""真实上游冒烟：跑一遍「装配 -> 入库 -> 检索 -> 护栏 -> 记账」，并把事实证明打印出来。

这个脚本存在的意义不是"再跑一次 demo"，而是**证明三个挂载点真的接上了**：

1. 图谱护栏确实拦在写入路径上（打印真实的写入尝试数与被拒数，而不是打印"已启用"）；
2. ``only_need_context`` 确实做到了只检索不生成（打印检索回来的来源文件）；
3. 记账确实按四角色分开（打印 extract / query 两个角色各自的调用数）。

用法::

    python scripts/lightrag_smoke.py --env-file <包含模型配置的 .env> --doc data/corpus/03-enterprise-rag-guardrails.md

模型配置从 ``--env-file`` 或进程环境读取，键名：
``ENTERPRISE_PRIMARY_BASE_URL`` / ``ENTERPRISE_PRIMARY_API_KEY`` / ``ENTERPRISE_PRIMARY_MODEL``。
**密钥只走环境变量，不落盘、不进日志。**
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from enterprise.answer_guard import AnswerGuardPolicy, EvidenceGate  # noqa: E402
from enterprise.graph_guard import GraphGuardPolicy  # noqa: E402
from enterprise.lightrag_integration import build_lightrag, ingest  # noqa: E402
from enterprise.metering import MeteringLedger  # noqa: E402
from enterprise.offline import DeterministicExtractive  # noqa: E402
from enterprise.retrievers import LightRAGRetriever  # noqa: E402


def load_env_file(path: str | None) -> int:
    if not path or not os.path.exists(path):
        return 0
    loaded = 0
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and value and key not in os.environ:
                os.environ[key] = value
                loaded += 1
    return loaded


class _Settings:
    """最小设置对象：只提供 ``build_lightrag`` 需要的字段，避免把网关配置一起拖进来。"""

    def __init__(self, working_dir: str, rejects_path: str) -> None:
        self.working_dir = working_dir
        self.rejects_path = rejects_path


async def run(args: argparse.Namespace) -> int:
    loaded = load_env_file(args.env_file)
    print(f"环境变量：从文件载入 {loaded} 项；模型 = {os.environ.get('ENTERPRISE_PRIMARY_MODEL', '(未配置)')}")

    ledger = MeteringLedger()
    settings = _Settings(args.working_dir, args.rejects)
    policy = GraphGuardPolicy(allowed_entity_types=None, on_violation=args.guard_mode)
    rag, guard = await build_lightrag(settings, ledger=ledger, guard_policy=policy)
    print(f"上游装配完成，护栏模式 = {guard.policy.on_violation}")

    text = Path(args.doc).read_text(encoding="utf-8")
    print(f"入库 {args.doc}（{len(text)} 字），这一步会真实调用抽取模型……")
    await ingest(rag, [text], file_paths=[Path(args.doc).name])

    stats = guard.stats()
    print(
        f"图谱写入尝试 {stats['node_attempts']} 点 / {stats['edge_attempts']} 边，"
        f"被拒 {stats['node_rejected']} 点 / {stats['edge_rejected']} 边，"
        f"拒绝原因 {stats['rejected_by_reason'] or '无'}"
    )
    if stats.get("observed_entity_types_top"):
        top = "、".join(f"{name}({count})" for name, count in stats["observed_entity_types_top"][:8])
        print(f"观测到的实体类型 Top8：{top}")

    retriever = LightRAGRetriever(rag, mode=args.mode)
    gate = EvidenceGate(AnswerGuardPolicy(min_coverage=args.min_coverage))
    extractive = DeterministicExtractive()
    for question in args.question:
        raw = await retriever.retrieve(question, top_k=args.top_k)
        data = raw.get("data") or {}
        references = [item.get("file_path") for item in data.get("references") or []]
        verdict = gate.check({"question": question, "raw_data": raw})
        print(f"\nQ: {question}")
        print(f"   检索到 {len(data.get('chunks') or [])} 个切片 / 来源 {references}")
        if verdict.rejected:
            print(f"   护栏判定：拒答（{verdict.reason.value}）{verdict.detail.get('coverage')=}")
            continue
        answer = extractive(question, context=data.get("chunks") or [], question=question)
        first_line = answer.splitlines()[0]
        print(f"   护栏判定：放行，覆盖率 {verdict.detail.get('coverage')}，阈值 {verdict.detail.get('threshold')}")
        print(f"   作答（离线摘录，仅用于证明链路）：{first_line}")

    print("\n四角色记账：")
    snapshot = ledger.snapshot()["by_role_layer"]
    for key, bucket in sorted(snapshot.items()):
        print(f"   {key}: 调用 {bucket['calls']} 次，失败 {bucket['failures']} 次，tokens_in={bucket['tokens_in']}")
    print(f"成本：{ledger.cost()}")
    if guard is not None and guard.policy.on_violation == "record":
        print(f"\n建议按观测结果收口词表（--guard-mode reject 配合 allowed_entity_types）：{guard.derive_vocabulary()}")
    return 0


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="真实上游冒烟（会真实调用模型，注意额度）")
    parser.add_argument("--env-file", default="", help="模型配置的 .env 文件路径")
    parser.add_argument("--doc", default="data/corpus/03-enterprise-rag-guardrails.md")
    parser.add_argument("--question", action="append", default=[], help="可重复；默认问两个")
    parser.add_argument("--mode", default="mix")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-coverage", type=float, default=0.4)
    parser.add_argument("--guard-mode", default="record", choices=["record", "reject"])
    parser.add_argument("--working-dir", default="data/lightrag_storage")
    parser.add_argument("--rejects", default="data/audit/graph_rejects.jsonl")
    args = parser.parse_args()
    if not args.question:
        args.question = [
            "拒答阈值为什么不用 BM25 的绝对分",
            "上海今天适合穿什么衣服",
        ]
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
