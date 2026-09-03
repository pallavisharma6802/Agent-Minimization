"""Evaluate a system on a task set, then sweep ablations and classify each agent."""
from __future__ import annotations

import statistics as stats
from dataclasses import asdict, dataclass, field
from typing import Callable

from . import ablations
from .system import TASK, MultiAgentSystem


@dataclass
class Task:
    id: str
    text: str
    score: Callable[[str], float]  # returns 0..1


@dataclass
class RunStats:
    accuracy: float
    acc_std: float
    cost_usd: float
    total_tokens: float
    latency_s: float
    avg_calls: float
    repeats: int

    def as_dict(self) -> dict:
        return {k: round(v, 6) if isinstance(v, float) else v for k, v in asdict(self).items()}


def evaluate(sys: MultiAgentSystem, tasks: list[Task], repeats: int = 3) -> RunStats:
    per_run_acc, cost, tok, lat, calls = [], [], [], [], []
    for _ in range(repeats):
        correct = 0.0
        rc = tc = lt = cl = 0.0
        for t in tasks:
            out, tr = sys.run(t.text, t.id)
            correct += t.score(out)
            rc += tr.cost_usd; tc += tr.total_tokens; lt += tr.latency_s; cl += tr.n_calls
        per_run_acc.append(correct / len(tasks))
        cost.append(rc); tok.append(tc); lat.append(lt); calls.append(cl / len(tasks))
    return RunStats(
        accuracy=stats.fmean(per_run_acc),
        acc_std=stats.pstdev(per_run_acc) if repeats > 1 else 0.0,
        cost_usd=stats.fmean(cost),
        total_tokens=stats.fmean(tok),
        latency_s=stats.fmean(lat),
        avg_calls=stats.fmean(calls),
        repeats=repeats,
    )


def _adjacent_pairs(sys: MultiAgentSystem) -> list[tuple[str, str]]:
    pairs = []
    for a in sys.agents:
        for src in a.inputs:
            if src != TASK and any(x.name == src for x in sys.agents):
                pairs.append((src, a.name))
    return pairs


@dataclass
class AgentVerdict:
    agent: str
    kind: str
    classification: str
    best_move: str
    delta_accuracy: float
    cost_saving_frac: float
    detail: dict = field(default_factory=dict)


def propose(sys: MultiAgentSystem, tasks: list[Task], repeats: int = 3,
            tol_sigma: float = 1.0, abs_tol: float = 0.02) -> dict:
    base = evaluate(sys, tasks, repeats)
    band = max(abs_tol, tol_sigma * base.acc_std)

    def keeps(r: RunStats) -> bool:
        return (r.accuracy - base.accuracy) >= -band

    verdicts: list[AgentVerdict] = []
    for ag in sys.agents:
        if ag.kind == "oversight":
            verdicts.append(AgentVerdict(ag.name, ag.kind, "protected (oversight)", "keep", 0.0, 0.0))
            continue

        trials: list[tuple[str, RunStats]] = []
        try:
            trials.append(("remove", evaluate(ablations.remove(sys, ag.name), tasks, repeats)))
        except Exception as e:
            trials.append((f"remove:err:{e}", base))
        trials.append(("identity", evaluate(ablations.identity(sys, ag.name), tasks, repeats)))
        trials.append(("downgrade", evaluate(ablations.downgrade(sys, ag.name, "mock-small"), tasks, repeats)))
        for u, v in _adjacent_pairs(sys):
            if ag.name in (u, v):
                other = v if ag.name == u else u
                if sys.by_name(other).kind == "oversight":
                    continue
                try:
                    trials.append((f"merge:{u}+{v}", evaluate(ablations.merge(sys, u, v), tasks, repeats)))
                except Exception:
                    pass

        passing = [(m, r) for m, r in trials if not m.startswith("remove:err") and keeps(r)]
        if passing:
            move, r = max(passing, key=lambda mr: base.cost_usd - mr[1].cost_usd)
            cls = {
                "remove": "redundant",
                "identity": "reasoning-not-needed",
                "downgrade": "model-overspecified",
            }.get(move, "mergeable" if move.startswith("merge") else "reducible")
            verdicts.append(AgentVerdict(
                ag.name, ag.kind, cls, move,
                round(r.accuracy - base.accuracy, 4),
                round(1 - r.cost_usd / base.cost_usd, 4) if base.cost_usd else 0.0,
                {"ablated": r.as_dict()},
            ))
        else:
            verdicts.append(AgentVerdict(ag.name, ag.kind, "load-bearing", "keep", 0.0, 0.0,
                                         {"trials": {m: r.as_dict() for m, r in trials}}))

    return {
        "system": sys.name,
        "baseline": base.as_dict(),
        "noise_band": round(band, 4),
        "verdicts": [asdict(v) for v in verdicts],
        "headline": _headline(sys, verdicts),
    }


def _headline(sys: MultiAgentSystem, verdicts: list[AgentVerdict]) -> dict:
    reducible = [v for v in verdicts if v.classification not in ("load-bearing", "protected (oversight)")]
    n_cap = sum(1 for a in sys.agents if a.kind == "capability")
    return {
        "agents_total": len(sys.agents),
        "capability_agents": n_cap,
        "reducible_agents": len(reducible),
        "reducible_frac_of_capability": round(len(reducible) / n_cap, 3) if n_cap else 0.0,
        "moves": {v.agent: v.best_move for v in reducible},
    }
