#!/usr/bin/env bash
# 治理层的门禁脚本：跑评测 + 跑测试 + 断言"两次运行逐字段一致"。
#
# 这个脚本的用处是让"护栏有没有真的拦住"变成一条命令的结论，
# 而不是靠人去点页面、看日志。CI 里跑的就是它。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "== 1/3 离线评测（零模型依赖） =="
python -m enterprise.eval.run_eval \
    --corpus data/corpus \
    --golden data/golden/golden.jsonl \
    --sweep 0.3,0.4,0.5,0.6,0.7,0.8 \
    --repeat 2 \
    --out data/eval_report.json

echo "== 2/3 断言两次运行逐字段一致 =="
python - <<'PY'
import json
import sys

report = json.load(open("data/eval_report.json", encoding="utf-8"))
if not report["deterministic"]:
    print("不一致：")
    for difference in report["differences"]:
        print(" -", difference)
    sys.exit(1)
if report["metrics"]["over_answer_rate"] != 0.0:
    print(f"无答案样本被作答：over_answer_rate={report['metrics']['over_answer_rate']}")
    sys.exit(1)
print(f"通过：阈值 {report['chosen_threshold']}，Recall@3={report['metrics']['recall@3']}，"
      f"误拒率={report['metrics']['false_refusal_rate']}，引用覆盖率={report['metrics']['citation_rate']}")
PY

echo "== 3/3 治理层测试 =="
# --confcutdir：不要加载上游 tests/conftest.py，让这一套测试保持"零上游依赖"。
pytest tests/enterprise --confcutdir=tests/enterprise -q

echo "全部通过。"
