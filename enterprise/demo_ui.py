"""演示面板：把护栏的判定过程摊开给人看。

一个只显示"最终答案"的演示，看不出这套东西和普通 RAG 有什么区别。
所以这个面板刻意把**中间判定**全部暴露出来：覆盖率是多少、阈值是多少、
实际生效的是降级链的第几层、有没有降级、引用是怎么补上的、被护栏拒了几条写入。

这既是演示，也是排障界面——面试里被追问"你怎么知道护栏真的生效了"，
答案就是这一屏。

    python -m enterprise.demo_ui        # 单端口同时提供 REST 网关与演示面板
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from .app import GatewaySettings, build_gateway, create_app, serve_uvicorn
from .auth import enforce_exposure


def _ui_headers(settings: GatewaySettings) -> dict[str, str]:
    """面板与网关同源，用的是服务端自己的密钥；外部访问仍然必须带凭据。"""
    if settings.api_key:
        return {"authorization": f"Bearer {settings.api_key}"}
    return {}


def _load_eval_report(settings: GatewaySettings) -> dict[str, Any]:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "eval_report.json")
    path = os.path.normpath(path)
    if not os.path.exists(path):
        return {"available": False, "hint": "运行 `python -m enterprise.eval.run_eval --corpus data/corpus --golden data/golden/golden.jsonl --out data/eval_report.json` 生成"}
    with open(path, encoding="utf-8") as handle:
        report = json.load(handle)
    return {
        "available": True,
        "chosen_threshold": report.get("chosen_threshold"),
        "selection_rule": report.get("selection_rule"),
        "metrics": report.get("metrics"),
        "deterministic": report.get("deterministic"),
        "runs_compared": report.get("runs_compared"),
        "sweep": report.get("threshold_sweep"),
    }


def _render_answer(result: dict[str, Any]) -> str:
    badge = "🟢 正常作答" if not result.get("refused") else "🟠 已拒答"
    if result.get("degraded"):
        badge += f"（降级到 {result.get('layer')}）"
    lines = [f"### {badge}", "", result.get("answer") or "（无内容）"]
    detail = result.get("detail") or {}
    if detail:
        lines += ["", "---", "**判定细节**", "", "```json", json.dumps(detail, ensure_ascii=False, indent=2), "```"]
    return "\n".join(lines)


def build_blocks(gateway: Any, settings: GatewaySettings) -> Any:
    import gradio as gr

    async def ask(question: str) -> tuple[str, str]:
        question = (question or "").strip()
        if not question:
            return "请输入问题。", "{}"
        result = await gateway.answer(
            question=question,
            client_id="demo-ui",
            path="/v1/query",
            headers=_ui_headers(settings),
        )
        meta = {
            "reason": result.get("reason"),
            "refused": result.get("refused"),
            "layer": result.get("layer"),
            "degraded": result.get("degraded"),
            "latency_ms": result.get("latency_ms"),
            "retrieval": result.get("retrieval"),
        }
        return _render_answer(result), json.dumps(meta, ensure_ascii=False, indent=2)

    def panel() -> str:
        return "```json\n" + json.dumps(gateway.health(), ensure_ascii=False, indent=2) + "\n```"

    def guard_panel() -> str:
        return "```json\n" + json.dumps(gateway.graph_stats(), ensure_ascii=False, indent=2) + "\n```"

    def metering_panel() -> str:
        return "```json\n" + json.dumps(gateway.metering(), ensure_ascii=False, indent=2) + "\n```"

    def eval_panel() -> str:
        return "```json\n" + json.dumps(_load_eval_report(settings), ensure_ascii=False, indent=2) + "\n```"

    examples = [
        "图谱写入为什么必须带来源，不带会怎样",
        "拒答阈值为什么不用 BM25 的绝对分",
        "自然语言转 SQL 的只读护栏有哪三条",
        "上海今天适合穿什么衣服",
    ]

    with gr.Blocks(title="LightRAG 治理层演示") as blocks:
        gr.Markdown(
            "## LightRAG 企业落地治理层 · 演示\n"
            "上游负责检索与知识图谱；本层负责**证据门槛、引用强制、降级兜底、成本归因**。\n\n"
            "试着问一个知识库里**没有答案**的问题（例如最后一个示例），"
            "看它拒答，以及拒答理由里列出了哪些没命中的关键词。"
        )
        with gr.Tab("问答"):
            with gr.Row():
                box = gr.Textbox(label="问题", lines=2, scale=4)
                with gr.Column(scale=1):
                    submit = gr.Button("提问", variant="primary")
                    clear = gr.Button("清空")
            answer = gr.Markdown()
            meta = gr.Code(label="判定元数据（层级 / 是否降级 / 覆盖率 / 耗时）", language="json")
            gr.Examples(examples=examples, inputs=box)
            submit.click(ask, inputs=box, outputs=[answer, meta])
            box.submit(ask, inputs=box, outputs=[answer, meta])
            clear.click(lambda: ("", "", "{}"), outputs=[box, answer, meta])

        with gr.Tab("评测数字（可复跑）"):
            gr.Markdown("由 `python -m enterprise.eval.run_eval` 产出，两次运行逐字段一致才算过。")
            gr.Code(value=eval_panel, language="json", label="eval_report.json")

        with gr.Tab("运行状态"):
            gr.Code(value=panel, language="json", label="/health")
            gr.Code(value=guard_panel, language="json", label="图谱写入护栏统计")
            gr.Code(value=metering_panel, language="json", label="四角色成本归因（不内置单价）")

    return blocks


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    settings = GatewaySettings.from_env()
    for finding in enforce_exposure(**settings.exposure_kwargs()):
        print(f"[警告] {finding.code}: {finding.message}")
    gateway = build_gateway(settings)
    app = create_app(settings, gateway=gateway)
    blocks = build_blocks(gateway, settings)
    import gradio as gr

    combined = gr.mount_gradio_app(app, blocks, path="/")
    print(f"[启动] 演示面板与 REST 网关共用端口 {settings.host}:{settings.port}")
    serve_uvicorn(combined, settings)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
