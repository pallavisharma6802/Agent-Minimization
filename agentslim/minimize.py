"""Greedy minimizer: repeatedly apply the cheapest ablation that stays within the
accuracy noise band, until nothing more can be removed. Returns the reduced
system plus a before/after ledger.

This is the artifact a user actually wants: not "agent X looks redundant" but
"here is the smaller system, and here is the measured cost of getting there."
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import ablations
from .harness import RunStats, Task, evaluate, _adjacent_pairs
from .system import MultiAgentSystem


@dataclass
class Step:
    move: str
    accuracy: float
    delta_vs_original: float
    cost_usd: float
    n_agents: int
    n_calls: float


@dataclass
class MinimizeResult:
    original: RunStats
    final: RunStats
    final_system: MultiAgentSystem
    steps: list[Step] = field(default_factory=list)

    def ledger(self) -> dict:
        return {
            "original": self.original.as_dict(),
            "final": self.final.as_dict(),
            "final_agents": [a.name for a in self.final_system.agents],
            "removed_calls_frac": round(1 - self.final.avg_calls / self.original.avg_calls, 3)
            if self.original.avg_calls else 0.0,
            "cost_saving_frac": round(1 - self.final.cost_usd / self.original.cost_usd, 3)
            if self.original.cost_usd else 0.0,
            "accuracy_change": round(self.final.accuracy - self.original.accuracy, 4),
            "steps": [s.__dict__ for s in self.steps],
        }


def _candidates(sys: MultiAgentSystem):
    """Yield (label, new_system) for every legal one-step reduction."""
    cap = [a for a in sys.agents if a.kind == "capability"]
    for a in cap:
        if len(sys.agents) > 1:
            try:
                yield f"remove:{a.name}", ablations.remove(sys, a.name)
            except Exception:
                pass
        yield f"identity:{a.name}", ablations.identity(sys, a.name)
        yield f"downgrade:{a.name}", ablations.downgrade(sys, a.name, "mock-small")
    for u, v in _adjacent_pairs(sys):
        if sys.by_name(u).kind == "oversight" or sys.by_name(v).kind == "oversight":
            continue
        try:
            yield f"merge:{u}+{v}", ablations.merge(sys, u, v)
        except Exception:
            pass


def greedy_minimize(sys: MultiAgentSystem, tasks: list[Task], repeats: int = 5,
                    tol_sigma: float = 1.0, abs_tol: float = 0.02,
                    max_steps: int = 12) -> MinimizeResult:
    import os as _os
    max_steps = int(_os.environ.get("AGENTSLIM_MAX_STEPS", max_steps))
    original = evaluate(sys, tasks, repeats)
    band = max(abs_tol, tol_sigma * original.acc_std)
    cur, cur_stats = sys.clone(), original
    res = MinimizeResult(original=original, final=original, final_system=cur)

    for _ in range(max_steps):
        best = None  # (cost, label, new_sys, stats)
        for label, cand in _candidates(cur):
            st = evaluate(cand, tasks, repeats)
            if (st.accuracy - original.accuracy) >= -band:
                key = st.cost_usd
                if best is None or key < best[0]:
                    best = (key, label, cand, st)
        if best is None:
            break
        _, label, cand, st = best
        # accept only if it actually shrinks something (cost strictly down)
        if st.cost_usd >= cur_stats.cost_usd - 1e-9:
            break
        cur, cur_stats = cand, st
        res.steps.append(Step(
            move=label, accuracy=round(st.accuracy, 4),
            delta_vs_original=round(st.accuracy - original.accuracy, 4),
            cost_usd=round(st.cost_usd, 6), n_agents=len(cand.agents),
            n_calls=round(st.avg_calls, 2),
        ))

    res.final, res.final_system = cur_stats, cur
    return res
