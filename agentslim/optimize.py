"""Right-size a multi-agent system — WITHOUT the "collapse to 1" trap.

Read WHY_MULTIAGENT.md first. The short version:

  Task accuracy is only ONE reason multi-agent systems exist, and the weakest
  one. Companies also split agents for context capacity, permission / blast-radius
  scoping, parallel throughput, org ownership, model heterogeneity, compliance
  gating and reliability. An accuracy-only sweep is blind to all of that and will
  always say "you only need 1". It is wrong about real systems.

So this module does two things:

  1. `structural_report(sys)` — a per-agent verdict based on the agent's METADATA
     (tools, permissions, data_scope, owner, model, role_type). Agents that hold a
     distinct tool/permission/scope/owner, or that are oversight/gate/parallel/
     router, are marked KEEP with the reason. They are never sent to the ablation
     sweep.

  2. `pareto_scan(sys, tasks)` — sweeps team sizes, but ONLY over the agents that
     survive step 1 (pure capability decomposition, no structural justification).
     It reports the quality/cost curve and the knee, plus which agents it was not
     allowed to touch and why.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import ablations
from .ablations import BoundaryViolation
from .eval import Task, evaluate, adjacent_pairs as _adjacent_pairs
from .system import MultiAgentSystem


# --------------------------------------------------------------------------
# 1. Structural verdict — metadata only, no LLM calls
# --------------------------------------------------------------------------
def structural_report(sys: MultiAgentSystem) -> dict:
    verdicts = []
    for a in sys.agents:
        block = sys.structural_block(a.name)
        if a.role_type != "capability":
            verdicts.append({"agent": a.name, "verdict": "KEEP",
                             "reason": f"role_type={a.role_type} — protected"})
        elif block:
            verdicts.append({"agent": a.name, "verdict": "KEEP",
                             "reason": block})
        else:
            verdicts.append({"agent": a.name, "verdict": "SWEEP",
                             "reason": "pure capability decomposition, no distinct "
                                       "tools/permissions/scope/owner/model — "
                                       "eligible for the ablation sweep"})
    sweepable = [v["agent"] for v in verdicts if v["verdict"] == "SWEEP"]
    return {
        "system": sys.name,
        "agents_total": len(sys.agents),
        "structurally_protected": len(sys.agents) - len(sweepable),
        "sweep_candidates": sweepable,
        "verdicts": verdicts,
    }


# --------------------------------------------------------------------------
# 2. Ablation sweep — only over structurally-unjustified agents
# --------------------------------------------------------------------------
@dataclass
class Point:
    k: int
    agents: list
    removed: str | None
    accuracy: float
    acc_std: float
    cost_usd: float
    avg_calls: float

    def row(self) -> dict:
        return {"agents": self.agents, "removed": self.removed,
                "accuracy": round(self.accuracy, 4), "acc_std": round(self.acc_std, 4),
                "cost_usd": round(self.cost_usd, 6), "avg_calls": round(self.avg_calls, 2)}


def _candidates(sys: MultiAgentSystem):
    """Legal one-step reductions, respecting every structural boundary."""
    for a in sys.agents:
        if sys.structural_block(a.name) is not None:
            continue
        if len(sys.agents) > 1:
            try:
                yield f"remove:{a.name}", ablations.remove(sys, a.name)
            except BoundaryViolation:
                pass
        try:
            yield f"identity:{a.name}", ablations.identity(sys, a.name)
        except BoundaryViolation:
            pass
    for u, v in _adjacent_pairs(sys):
        if sys.merge_allowed(u, v) is not None:
            continue
        try:
            yield f"merge:{u}+{v}", ablations.merge(sys, u, v)
        except BoundaryViolation:
            pass


def _best_removal(sys: MultiAgentSystem, tasks: list[Task], repeats: int):
    cands = list(_candidates(sys))
    if not cands:
        return None
    scored = [(lbl, s, evaluate(s, tasks, repeats)) for lbl, s in cands]
    scored.sort(key=lambda t: (-t[2].accuracy, t[2].cost_usd))
    return scored[0]


def pareta_scan(*a, **k):  # tolerate the historical typo
    return pareto_scan(*a, **k)


def pareto_scan(sys: MultiAgentSystem, tasks: list[Task], repeats: int = 3) -> dict:
    struct = structural_report(sys)
    base = evaluate(sys, tasks, repeats)
    cur = sys.clone()
    points = [Point(k=len(cur.agents), agents=[x.name for x in cur.agents], removed=None,
                    accuracy=base.accuracy, acc_std=base.acc_std,
                    cost_usd=base.cost_usd, avg_calls=base.avg_calls)]

    while True:
        pick = _best_removal(cur, tasks, repeats)
        if pick is None:
            break
        lbl, nxt, st = pick
        if st.avg_calls >= points[-1].avg_calls - 1e-6:
            break
        cur = nxt
        points.append(Point(k=round(st.avg_calls), agents=[x.name for x in cur.agents],
                            removed=lbl, accuracy=st.accuracy, acc_std=st.acc_std,
                            cost_usd=st.cost_usd, avg_calls=st.avg_calls))

    best_acc = max(p.accuracy for p in points)
    band = max(0.02, points[0].acc_std)
    knee = min((p for p in points if p.accuracy >= best_acc - band),
              key=lambda p: len(p.agents))

    removed = [p.removed for p in points[1:]]
    # per-agent final disposition, so "kept" doesn't hide a neutralized agent
    disposition = {a.name: "kept" for a in sys.agents}
    for mv in removed:
        op, _, who = mv.partition(":")
        if op == "remove":
            disposition[who] = "removed"
        elif op == "identity":
            disposition[who] = "neutralized (no LLM call — passthrough)"
        elif op == "merge":
            x, _, y = who.partition("+")
            disposition[y] = f"merged into {x}"
    for name, block in ((a.name, sys.structural_block(a.name)) for a in sys.agents):
        if disposition[name] == "kept" and block:
            disposition[name] = f"kept (protected: {block})"

    return {
        "system": sys.name,
        "structural": struct,
        "band": round(band, 4),
        "curve": [p.row() for p in points],
        "recommendation": {
            "disposition": disposition,
            "agents_making_llm_calls_after": round(knee.avg_calls, 1),
            "agents_making_llm_calls_before": round(points[0].avg_calls, 1),
            "removed": removed,
            "note": (f"{struct['structurally_protected']} of {struct['agents_total']} "
                     f"agents are structurally load-bearing (see verdicts) and were "
                     f"never touched. Of the {len(struct['sweep_candidates'])} "
                     f"capability-decomposition agents, {len(removed)} are removable "
                     f"with no measured accuracy loss."),
            "accuracy_change": round(knee.accuracy - points[0].accuracy, 4),
            "cost_saving_frac": round(1 - knee.cost_usd / points[0].cost_usd, 4)
            if points[0].cost_usd else 0.0,
        },
    }
