from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chat_reply_agent.services.evaluation_service import (  # noqa: E402
    DEFAULT_DATASET_PATH,
    load_evaluation_dataset,
    run_evaluation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 chat_reply_agent v0.12 评测集")
    parser.add_argument(
        "--case",
        action="append",
        dest="case_ids",
        help="只运行指定样本，可重复传入；默认运行全部样本。",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="自定义 YAML 评测集路径。",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="输出完整 JSON，便于保存为基线或接入 CI。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _, _, cases = load_evaluation_dataset(args.dataset)
    case_ids = args.case_ids or [case.id for case in cases]
    result = run_evaluation(case_ids, args.dataset)

    if args.json:
        print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
    else:
        print(
            f"{result.dataset_name} {result.version}: "
            f"平均分：{result.average_score if result.average_score is not None else '未评分'}，"
            f"通过 {result.passed_cases}/{result.total_cases}"
        )
        for item in result.results:
            state = "PASS" if item.passed else "FAIL"
            detail = f"error={item.error}" if item.error else "；".join(item.issues)
            score = f"{item.score} 分" if item.score is not None else "未评分"
            print(f"[{state}] {item.case_id}: {score} {detail}".rstrip())
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
