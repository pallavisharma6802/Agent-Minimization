"""Run the ablation sweep on a pilot system and write a JSON report.

Usage:
  python experiments/run_pilot.py math_committee
  AGENTSLIM_BACKEND=openai python experiments/run_pilot.py math_committee
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentslim import evaluate, propose  # noqa: E402


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "math_committee"
    repeats = int(os.environ.get("AGENTSLIM_REPEATS", "5"))
    mod = importlib.import_module(f"systems.{name}")
    sysm, tasks = mod.build(), mod.tasks()

    print(f"== {name}  backend={os.environ.get('AGENTSLIM_BACKEND', 'mock')}  repeats={repeats}")
    base = evaluate(sysm, tasks, repeats)
    print(f"baseline: acc={base.accuracy:.3f}±{base.acc_std:.3f}  "
          f"calls/task={base.avg_calls:.1f}  cost=${base.cost_usd:.5f}  tokens={base.total_tokens:.0f}")

    t0 = time.time()
    report = propose(sysm, tasks, repeats=repeats)
    report["meta"] = {"backend": os.environ.get("AGENTSLIM_BACKEND", "mock"),
                      "repeats": repeats, "wall_s": round(time.time() - t0, 1)}

    print("\nverdicts:")
    for v in report["verdicts"]:
        print(f"  {v['agent']:<10} {v['kind']:<11} -> {v['classification']:<22} "
              f"move={v['best_move']:<16} dAcc={v['delta_accuracy']:+.3f} save={v['cost_saving_frac']:.2f}")
    print("\nheadline:", json.dumps(report["headline"], indent=2))

    os.makedirs("results", exist_ok=True)
    path = f"results/{name}.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
