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

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

# minimal .env loader (no dependency)
_envf = os.path.join(_ROOT, ".env")
if os.path.exists(_envf):
    for _line in open(_envf):
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

from agentslim import evaluate, greedy_minimize, propose  # noqa: E402
from agentslim.llm import METER  # noqa: E402


def _estimate(sysm, tasks, repeats, max_steps):
    n_ag = len(sysm.agents)
    n_cap = sum(1 for a in sysm.agents if a.kind == "capability")
    n_edge = sum(1 for a in sysm.agents for s in a.inputs if s != "__task__")
    per_eval = n_ag * len(tasks) * repeats
    propose_evals = 1 + n_cap * (3 + 1) + n_edge          # remove/identity/downgrade + merges
    minimize_evals = max_steps * (n_cap * 3 + n_edge)
    return per_eval * (propose_evals + minimize_evals)


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "math_committee"
    backend = os.environ.get("AGENTSLIM_BACKEND", "mock")
    repeats = int(os.environ.get("AGENTSLIM_REPEATS", "2" if backend != "mock" else "5"))
    max_steps = int(os.environ.get("AGENTSLIM_MAX_STEPS", "12"))
    mod = importlib.import_module(f"systems.{name}")
    sysm, tasks = mod.build(), mod.tasks()
    if os.environ.get("AGENTSLIM_MAX_TASKS"):
        tasks = tasks[:int(os.environ["AGENTSLIM_MAX_TASKS"])]

    if backend != "mock":
        upper = _estimate(sysm, tasks, repeats, max_steps)
        rpm = float(os.environ.get("AGENTSLIM_RPM", "8"))
        model = os.environ.get("AGENTSLIM_MODEL", "?")
        # gemini-flash ballpark: ~1.2k in + 0.3k out per call
        est_usd = upper * (1200 * 0.30 + 300 * 2.50) / 1_000_000
        print(f"PLAN  system={name} backend={backend} model={model} repeats={repeats} "
              f"max_steps={max_steps}")
        print(f"      <= {upper} LLM calls (pre-cache upper bound); "
              f"~{upper / max(rpm,1) / 60:.0f}h at {rpm} rpm; ~${est_usd:.2f} worst case")
        print(f"      cost ceiling this run: ${METER.cap:.2f}  (AGENTSLIM_MAX_USD)")
        if not os.environ.get("AGENTSLIM_YES"):
            print("      set AGENTSLIM_YES=1 to proceed.")
            return

    print(f"== {name}  backend={backend}  repeats={repeats}")
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

    mr = greedy_minimize(sysm, tasks, repeats=repeats)
    report["minimize"] = mr.ledger()
    print("\ngreedy minimize:")
    for s in mr.steps:
        print(f"  {s.move:<22} acc={s.accuracy:.3f} (d{s.delta_vs_original:+.3f}) "
              f"agents={s.n_agents} calls/task={s.n_calls}")
    led = report["minimize"]
    print(f"  => {led['original']['avg_calls']:.1f} -> {led['final']['avg_calls']:.1f} calls/task, "
          f"cost -{led['cost_saving_frac']*100:.0f}%, acc {led['accuracy_change']:+.3f}, "
          f"final agents: {led['final_agents']}")

    report["spend"] = {"usd": round(METER.spent, 5), "calls": METER.calls}
    os.makedirs("results", exist_ok=True)
    path = f"results/{name}.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nspent ${METER.spent:.4f} over {METER.calls} live calls")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
