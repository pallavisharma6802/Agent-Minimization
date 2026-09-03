"""Real framework #2 (working): a LangGraph `StateGraph` multi-agent pipeline in
the "multi-agent collaboration" style from the langchain-ai/langgraph docs —
explicit nodes, explicit edges, each node an LLM agent. No tool-calling / routing
(see langgraph_supervisor_team.py for the supervisor variant, which needs a full
function-calling client our shim doesn't provide).

Graph:  researcher -> analyst -> writer  (linear), state carries the running text.

Ablations: drop a node (rewire the edge), or collapse to a single node.
Metric: a separate Gemini judge scores the final answer 0-10 vs a reference.
"""
from __future__ import annotations

import os
import re
import sys
from typing import TypedDict

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
_envf = os.path.join(_ROOT, ".env")
if os.path.exists(_envf):
    for _l in open(_envf):
        _l = _l.strip()
        if _l and not _l.startswith("#") and "=" in _l:
            k, v = _l.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
os.environ.setdefault("AGENTSLIM_MAX_TOKENS", "2048")

from agentslim.llm import LLM, METER  # noqa: E402
from agentslim.trace import Trace  # noqa: E402

NODES = {
    "researcher": "Recall the relevant facts and context for the question. Do not answer yet.",
    "analyst": "Reason step by step from the notes so far. Check for traps. Do not write the final answer.",
    "writer": "Write the final, clear, correct, concise answer for the user.",
}

QA = [
    ("A bat and a ball cost $1.10. The bat costs $1.00 more than the ball. "
     "How much is the ball?", "$0.05"),
    ("If 5 machines take 5 minutes to make 5 widgets, how long for 100 machines "
     "to make 100 widgets?", "5 minutes"),
    ("A farmer has 17 sheep and all but 9 die. How many are left?", "9"),
    ("What is heavier: a kilogram of steel or a kilogram of feathers?",
     "Equal — one kilogram each."),
    ("In a race you overtake the person in 2nd place. What place are you in now?",
     "2nd place"),
    ("A clerk says two coins total 30 cents and one is not a nickel. "
     "What are they?", "A quarter and a nickel."),
]

HARD = [
    ("What is the remainder when 7^100 is divided by 13?", "9"),
    ("How many positive integers less than 1000 are divisible by neither 3 nor 7?",
     "571"),
    ("Alice, Bob, Carol each have a different pet (cat, dog, fish) and live in "
     "houses 1-3 left to right. The dog owner is directly left of Carol. Alice is "
     "in house 3. Bob does not own the fish. Who owns the cat?", "Bob owns the cat."),
    ("A snail climbs 3 m up a 10 m well each day and slips 2 m each night. "
     "On which day does it reach the top?", "Day 8"),
    ("Three people check into a hotel room costing $30, paying $10 each. The clerk "
     "later refunds $5; the bellhop keeps $2 and returns $1 to each guest. Each "
     "guest paid $9 (total $27) plus the bellhop's $2 is $29. Where is the missing "
     "dollar?", "There is no missing dollar; the $27 already includes the bellhop's $2."),
    ("If today is Wednesday, what day of the week will it be 100 days from now?",
     "Friday"),
]


def build_graph(keep: list[str]):
    from langgraph.graph import END, START, StateGraph

    class S(TypedDict):
        question: str
        notes: str
        answer: str

    def make(node_name):
        def fn(state: S):
            sysmsg = NODES[node_name]
            user = f"QUESTION: {state['question']}\n\nNOTES SO FAR:\n{state.get('notes','')}"
            out = LLM(model=os.environ["AGENTSLIM_MODEL"]).complete(
                sysmsg, user, agent=node_name)
            if node_name == "writer":
                return {"answer": out}
            return {"notes": state.get("notes", "") + f"\n[{node_name}] {out}"}
        return fn

    g = StateGraph(S)
    order = [n for n in ("researcher", "analyst", "writer") if n in keep]
    if "writer" not in order:
        order.append("writer")  # always need something that produces `answer`
    for n in order:
        g.add_node(n, make(n))
    g.add_edge(START, order[0])
    for a, b in zip(order, order[1:]):
        g.add_edge(a, b)
    g.add_edge(order[-1], END)
    return g.compile()


def judge(q, ref, ans) -> float:
    out = LLM(model=os.environ["AGENTSLIM_MODEL"]).complete(
        "You are a strict grader. Reply with only an integer 0-10.",
        f"Question: {q}\nReference: {ref}\nCandidate: {ans}\nScore 0-10 for correctness.",
        agent="judge")
    m = re.search(r"\d+", out)
    return max(0.0, min(1.0, int(m.group(0)) / 10.0)) if m else 0.0


def run(keep, repeats=1, dataset=None):
    scores, traces = [], []
    for _ in range(repeats):
        for q, ref in (dataset or QA):
            app = build_graph(keep)
            tr = Trace(task_id=q[:24])
            with tr:
                res = app.invoke({"question": q, "notes": "", "answer": ""})
                ans = res.get("answer", "")
                tr.final_output = ans
            scores.append(judge(q, ref, ans))
            traces.append(tr)
    return scores, traces


if __name__ == "__main__":
    import json
    repeats = int(os.environ.get("AGENTSLIM_REPEATS", "2"))
    configs = {
        "researcher+analyst+writer (3)": ["researcher", "analyst", "writer"],
        "analyst+writer (2)": ["analyst", "writer"],
        "researcher+writer (2)": ["researcher", "writer"],
        "writer only (1)": ["writer"],
    }
    out = {}
    for dsname, ds in (("easy", QA), ("hard", HARD)):
        for name, keep in configs.items():
            sc, traces = run(keep, repeats, dataset=ds)
            calls = sum(t.n_calls for t in traces) / len(traces)
            key = f"[{dsname}] {name}"
            out[key] = {"score_mean": round(sum(sc) / len(sc), 3),
                        "scores": [round(x, 2) for x in sc], "avg_calls": round(calls, 2)}
            print(f"{key:<42} score={out[key]['score_mean']:.3f} calls/q={calls:.1f} "
                  f"(real ${METER.spent:.3f})")
    with open(os.path.join(_ROOT, "results", "langgraph_pipeline.json"), "w") as f:
        json.dump({"framework": "langgraph StateGraph (linear multi-agent pipeline)",
                   "configs": out, "model": os.environ.get("AGENTSLIM_MODEL"),
                   "spend": {"live_usd": round(METER.spent, 5), "live_calls": METER.calls,
                             "modeled_usd": round(METER.modeled, 5)}}, f, indent=2)
    print(f"\nreal API spend ${METER.spent:.4f} / {METER.calls} live calls "
          f"-> results/langgraph_pipeline.json")
