"""Pilot system #2: a 5-agent debate/ensemble ("More Agents Is All You Need"
style) — 4 parallel solvers + 1 aggregator that majority-votes.

Known ground truth: on these easy tasks a single solver already gets most right,
so 3 of the 4 solvers should be removable for near-zero accuracy loss. Tests
whether the minimizer recovers the "ensemble is over-provisioned" result.
"""
from __future__ import annotations

import re

from agentslim import Agent, MultiAgentSystem, Task

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def build() -> MultiAgentSystem:
    solvers = [
        Agent(f"solver{i}", "You are an arithmetic solver. Produce the numeric answer.",
              kind="capability", model="mock-large", inputs=["__task__"])
        for i in range(1, 5)
    ]
    agg = Agent("aggregator", "You take candidate answers and return the majority numeric answer.",
                kind="capability", model="mock-small",
                inputs=["solver1", "solver2", "solver3", "solver4"])
    return MultiAgentSystem(name="debate", sink="aggregator", agents=[*solvers, agg])


_RAW = [
    ("Each box has 6 apples and there are 4 boxes. How many apples in total?", 24),
    ("A train travels 30 miles per hour for 3 hours. How many miles total?", 90),
    ("There are 7 red balls, 8 blue balls and 5 green balls. How many balls altogether?", 20),
    ("Each basket holds 9 eggs and there are 3 baskets. How many eggs in total?", 27),
    ("Each pack has 8 pens and there are 5 packs. How many pens total?", 40),
    ("A garden has 10 roses, 6 tulips and 4 daisies. How many flowers altogether?", 20),
    ("Each pack has 3 juice boxes and there are 7 packs. How many juice boxes?", 21),
    ("A movie runs 2 hours. It plays 4 times a day. How many hours of screening per day?", 8),
    ("There are 11 cats and 9 dogs at the shelter. How many animals altogether?", 20),
    ("Each row has 12 seats and there are 5 rows. How many seats total?", 60),
]


def _scorer(gold: float):
    def score(output: str) -> float:
        got = _NUM.findall(output or "")
        if not got:
            return 0.0
        try:
            return 1.0 if abs(float(got[-1]) - gold) < 1e-6 else 0.0
        except ValueError:
            return 0.0
    return score


def tasks() -> list[Task]:
    return [Task(id=f"db{i:02d}", text=q, score=_scorer(a)) for i, (q, a) in enumerate(_RAW)]
