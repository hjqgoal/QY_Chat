"""Run the v0.20 deterministic before/after comparison without model calls."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chat_reply_agent.services.regression_service import compare_reply_regressions


if __name__ == "__main__":
    report = compare_reply_regressions()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 1)
