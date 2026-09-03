"""Run a multi-agent system on a task set and collect quality + cost stats.

This is deliberately small. It measures ONE axis (task-metric accuracy) — see
WHY_MULTIAGENT.md for why that is necessary but not sufficient, and why
`optimize.pareto_scan` gates it behind the structural report.
"""
from __future__ import annotations

import statistics as stats
from dataclasses import asdict, dataclass
from typing import Callable

from .system import TASK, MultiAgentSystem


@dataclass
class Task:
    id: str
    text: str
    score: Callable[[str], float]  # final output -> 0..1


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
        correct = rc = tc = lt = cl = 0.0
        for t in tasks:
            out, tr = sys.run(t.text, t.id)
            correct += t.score(out)
            rc += tr.cost_usd; tc += tr.total_tokens; lt += tr.latency_s; cl += tr.n_calls
        per_run_acc.append(correct / len(tasks))
        cost.append(rc); tok.append(tc); lat.append(lt); calls.append(cl / len(tasks))
    return RunStats(
        accuracy=stats.fmean(per_run_acc),
        acc_std=stats.pstdev(per_run_acc) if repeats > 1 else 0.0,
        cost_usd=stats.fmean(cost), total_tokens=stats.fmean(tok),
        latency_s=stats.fmean(lat), avg_calls=stats.fmean(calls), repeats=repeats,
    )


def adjacent_pairs(sys: MultiAgentSystem) -> list[tuple[str, str]]:
    pairs = []
    for a in sys.agents:
        for src in a.inputs:
            if src != TASK and any(x.name == src for x in sys.agents):
                pairs.append((src, a.name))
    return pairs
