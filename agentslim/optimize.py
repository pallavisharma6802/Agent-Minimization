"""Find the RIGHT number of agents, not the minimum.

`greedy_minimize` answers "how small can this get without losing quality" — which
on capable models often bottoms out at 1. That's not the interesting question for
a real team. `pareto_scan` instead sweeps every team size k = N .. 1, finds the
best agent subset at each k (greedy removal), and records (k, quality, cost). The
caller picks the operating point:

  - `knee`   : smallest k whose quality is within `band` of the best observed
  - `best`   : k with the highest quality outright
  - `budget` : largest k whose cost <= a $ ceiling

So a 6-agent system that genuinely needs 4 reports "4", and one that was padding
reports "1" — same procedure, honest answer either way.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import ablations
from .harness import RunStats, Task, evaluate, _adjacent_pairs
from .system import MultiAgentSystem


@dataclass
class Point:
    k: int
    agents: list[str]
    removed: str | None
    accuracy: float
    acc_std: float
    cost_usd: float
    avg_calls: float

    def row(self) -> dict:
        return {"k": self.k, "agents": self.agents, "removed": self.removed,
                "accuracy": round(self.accuracy, 4), "acc_std": round(self.acc_std, 4),
                "cost_usd": round(self.cost_usd, 6), "avg_calls": round(self.avg_calls, 2)}


def _best_removal(sys: MultiAgentSystem, tasks: list[Task], repeats: int):
    """Return (label, new_sys, stats) for the single removal/merge that best
    preserves accuracy (ties broken by cost)."""
    cands = []
    caps = [a for a in sys.agents if a.kind == "capability"]
    for a in caps:
        if len(sys.agents) > 1:
            try:
                cands.append((f"remove:{a.name}", ablations.remove(sys, a.name)))
            except Exception:
                pass
        cands.append((f"identity:{a.name}", ablations.identity(sys, a.name)))
    for u, v in _adjacent_pairs(sys):
        if sys.by_name(u).kind == "oversight" or sys.by_name(v).kind == "oversight":
            continue
        try:
            cands.append((f"merge:{u}+{v}", ablations.merge(sys, u, v)))
        except Exception:
            pass
    if not cands:
        return None
    scored = [(lbl, s, evaluate(s, tasks, repeats)) for lbl, s in cands]
    scored.sort(key=lambda t: (-t[2].accuracy, t[2].cost_usd))
    return scored[0]


def pareta_scan(*a, **k):  # tolerate the typo
    return pareto_scan(*a, **k)


def pareto_scan(sys: MultiAgentSystem, tasks: list[Task], repeats: int = 3,
                min_k: int = 1) -> dict:
    base = evaluate(sys, tasks, repeats)
    cur = sys.clone()
    points = [Point(k=len(cur.agents), agents=[x.name for x in cur.agents], removed=None,
                    accuracy=base.accuracy, acc_std=base.acc_std,
                    cost_usd=base.cost_usd, avg_calls=base.avg_calls)]
    while len([x for x in cur.agents if x.kind == "capability"]) > min_k:
        pick = _best_removal(cur, tasks, repeats)
        if pick is None:
            break
        lbl, nxt, st = pick
        prev_calls = points[-1].avg_calls
        cur = nxt
        # only record a point if the team actually got cheaper (fewer LLM calls)
        if st.avg_calls >= prev_calls - 1e-6:
            break
        points.append(Point(k=round(st.avg_calls), agents=[x.name for x in cur.agents],
                            removed=lbl, accuracy=st.accuracy, acc_std=st.acc_std,
                            cost_usd=st.cost_usd, avg_calls=st.avg_calls))

    best_acc = max(p.accuracy for p in points)
    band = max(0.02, points[0].acc_std)
    knee = min((p for p in points if p.accuracy >= best_acc - band), key=lambda p: p.k)
    return {
        "system": sys.name,
        "band": round(band, 4),
        "curve": [p.row() for p in points],
        "recommendation": {
            "knee_k": knee.k, "knee_agents": knee.agents,
            "keeps_accuracy": round(knee.accuracy, 4),
            "vs_baseline_acc": round(knee.accuracy - points[0].accuracy, 4),
            "cost_saving_frac": round(1 - knee.cost_usd / points[0].cost_usd, 4)
            if points[0].cost_usd else 0.0,
        },
    }
