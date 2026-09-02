"""Pilot system #1: a 4-agent "math committee" for grade-school word problems.

Designed with a known ground truth about which agents matter, so we can check
whether the ablation harness recovers it:

  planner    (capability)  -> expected REDUNDANT     (decomposition adds nothing here)
  solver     (capability)  -> expected LOAD-BEARING
  checker    (oversight)   -> PROTECTED, and genuinely helps on 3-quantity problems
  formatter  (capability)  -> expected REDUNDANT / reasoning-not-needed (regex does it)
"""
from __future__ import annotations

import re

from agentslim import Agent, MultiAgentSystem, Task

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def build() -> MultiAgentSystem:
    return MultiAgentSystem(
        name="math_committee",
        sink="formatter",
        agents=[
            Agent("planner", "You decompose a math word problem into ordered steps. Do not solve it.",
                  kind="capability", model="mock-small", inputs=["__task__"]),
            Agent("solver", "You are a careful arithmetic solver. Use the plan and produce the numeric answer.",
                  kind="capability", model="mock-large", inputs=["__task__", "planner"]),
            Agent("checker", "You are an oversight verifier. Independently recompute and correct the solver.",
                  kind="oversight", model="mock-large", inputs=["__task__", "solver"]),
            Agent("formatter", "You extract just the final number from the verified answer.",
                  kind="capability", model="mock-small", inputs=["checker"]),
        ],
    )


# --- eval set -----------------------------------------------------------
_RAW = [
    ("Each box has 6 apples and there are 4 boxes. How many apples in total?", 24),
    ("A shelf has 5 books. Someone adds 3 more books, then adds 2 additional books. How many books are on the shelf?", 10),
    ("Tom has 12 marbles. He gives away 4 and then gives away 3. How many marbles are left?", 5),
    ("A train travels 30 miles per hour for 3 hours. How many miles total?", 90),
    ("There are 7 red balls, 8 blue balls and 5 green balls. How many balls altogether?", 20),
    ("Each basket holds 9 eggs and there are 3 baskets. How many eggs in total?", 27),
    ("Sara had 20 dollars. She spends 6 dollars and then spends 5 dollars. How much is left?", 9),
    ("A room has 4 tables. Each table has 5 chairs. Then 2 more chairs are added. How many chairs?", 22),
    ("A garden has 10 roses, 6 tulips and 4 daisies. How many flowers altogether?", 20),
    ("A bus carries 15 people. 4 get off and then 3 get off. How many people remain?", 8),
    ("Each pack has 8 pens and there are 5 packs. How many pens total?", 40),
    ("A jar holds 14 candies. 5 are eaten and then 6 are eaten. How many candies are left?", 3),
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
    return [Task(id=f"mc{i:02d}", text=q, score=_scorer(a)) for i, (q, a) in enumerate(_RAW)]
