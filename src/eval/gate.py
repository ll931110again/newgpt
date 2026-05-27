from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _extract_metric(results: Dict[str, Any], task: str, metric: str) -> float:
    res = results.get("results", {})
    t = res.get(task, {})
    if metric in t:
        return float(t[metric])
    # lm-eval often outputs both `acc` and `acc_norm` depending on task
    raise KeyError(f"Missing metric {metric} for task {task}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="Path to runs/eval/summary.json")
    ap.add_argument("--min_arc_easy_acc", type=float, default=0.55)
    ap.add_argument("--min_hellaswag_acc", type=float, default=0.45)
    args = ap.parse_args()

    p = Path(args.summary)
    data = json.loads(p.read_text())

    arc = _extract_metric(data, "arc_easy", "acc")
    hs = _extract_metric(data, "hellaswag", "acc")

    failures = []
    if arc < args.min_arc_easy_acc:
        failures.append(f"arc_easy acc {arc:.4f} < {args.min_arc_easy_acc:.4f}")
    if hs < args.min_hellaswag_acc:
        failures.append(f"hellaswag acc {hs:.4f} < {args.min_hellaswag_acc:.4f}")

    if failures:
        raise SystemExit("GATING_FAILED\n" + "\n".join(failures))

    print("GATING_PASSED")


if __name__ == "__main__":
    main()

