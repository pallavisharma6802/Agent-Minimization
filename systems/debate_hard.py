"""debate on HARD problems — the case where an ensemble is supposed to help.

Same shape as systems/debate (4 independent solvers -> majority-vote aggregator),
but the tasks are competition-style multi-step problems where a single
`gemini-3.5-flash` pass is unreliable. If "More Agents Is All You Need" holds,
the 4-solver vote should beat 1 solver here, and the minimizer should KEEP the
solvers (unlike on the easy set, where it stripped them).
"""
from __future__ import annotations

import re

from agentslim import Agent, MultiAgentSystem, Task

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def build() -> MultiAgentSystem:
    solvers = [
        Agent(f"solver{i}",
              "You solve a hard math problem. Show brief working, then end with "
              "'ANSWER: <number>'.",
              kind="capability", model="mock-large", inputs=["__task__"])
        for i in range(1, 5)
    ]
    agg = Agent("aggregator",
                "You are given four candidate solutions. Pick the answer that "
                "appears most often (majority vote). End with 'ANSWER: <number>'.",
                kind="capability", model="mock-small",
                inputs=["solver1", "solver2", "solver3", "solver4"])
    return MultiAgentSystem(name="debate_hard", sink="aggregator", agents=[*solvers, agg])


# integer-answer, multi-step. answers verified by hand.
_RAW = [
    ("A number leaves remainder 2 when divided by 3, remainder 3 when divided by 4, "
     "and remainder 4 when divided by 5. What is the smallest positive such number?", 59),
    ("How many positive divisors does 360 have?", 24),
    ("What is the sum of all positive integers less than 100 that are divisible by 3 or 5?", 2318),
    ("A fair coin is flipped 4 times. In how many of the 16 outcomes are there more "
     "heads than tails?", 5),
    ("The product of two consecutive even integers is 168. What is their sum?", 26),
    ("How many trailing zeros are in 25! (25 factorial)?", 6),
    ("If x + 1/x = 3, what is x^3 + 1/x^3?", 18),
    ("A right triangle has legs 9 and 12. What is the length of the altitude to the "
     "hypotenuse, times 5?", 36),
    ("How many 3-digit numbers have digits that sum to 5?", 15),
    ("What is the remainder when 2^40 is divided by 100?", 76),
    ("In how many ways can 7 identical balls be placed into 3 distinct boxes so that "
     "no box is empty?", 15),
    ("The sequence a1=2, a_{n+1}=2 a_n - 1. What is a6?", 33),
]


def _scorer(gold: float):
    def score(output: str) -> float:
        m = re.findall(r"ANSWER:\s*(-?\d+(?:\.\d+)?)", output or "")
        got = m[-1] if m else (_NUM.findall(output or "") or [None])[-1]
        if got is None:
            return 0.0
        try:
            return 1.0 if abs(float(got) - gold) < 1e-6 else 0.0
        except ValueError:
            return 0.0
    return score


def tasks() -> list[Task]:
    return [Task(id=f"dh{i:02d}", text=q, score=_scorer(a)) for i, (q, a) in enumerate(_RAW)]
