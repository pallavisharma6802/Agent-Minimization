"""Real framework #2: langchain-ai/langgraph + langchain-ai/langgraph-supervisor
(`create_supervisor`, the canonical supervisor-multi-agent pattern).

A supervisor routes a question to specialist agents and combines their replies.
We build the standard 3-specialist team (researcher / analyst / writer), all
tool-free, and ask general-knowledge/reasoning questions. Quality is judged by a
separate Gemini judge (0-10) against a reference answer.

Ablations:
  - full team (supervisor + 3 specialists)
  - drop one specialist at a time
  - no supervisor: a single generalist agent answers directly

Question: does the supervisor + specialist split beat one good agent?
"""
from __future__ import annotations

import os
import re
import sys

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

from agentslim.adapters.langchain_shim import GeminiChat  # noqa: E402
from agentslim.llm import LLM, METER  # noqa: E402
from agentslim.trace import Trace  # noqa: E402

SPECIALISTS = {
    "researcher": "You recall relevant facts, definitions and context for the question.",
    "analyst": "You reason step by step about the question and check the logic.",
    "writer": "You write the final clear, correct, concise answer for the user.",
}

QA = [
    ("A bat and a ball cost $1.10 total. The bat costs $1.00 more than the ball. "
     "How much is the ball?", "$0.05 (5 cents)"),
    ("If it takes 5 machines 5 minutes to make 5 widgets, how long for 100 machines "
     "to make 100 widgets?", "5 minutes"),
    ("Name the only U.S. state whose name is a single syllable.", "Maine"),
    ("A farmer has 17 sheep and all but 9 die. How many are left?", "9"),
    ("What is heavier: a kilogram of steel or a kilogram of feathers?",
     "They weigh the same (one kilogram each)."),
    ("I have two coins totaling 30 cents and one of them is not a nickel. "
     "What are the coins?", "A quarter and a nickel (the quarter is 'not a nickel')."),
]


def build_team(keep_specialists):
    from langgraph.prebuilt import create_react_agent
    from langgraph_supervisor import create_supervisor

    agents = []
    for name in keep_specialists:
        m = GeminiChat(model_name=os.environ["AGENTSLIM_MODEL"], agent_tag=name)
        agents.append(create_react_agent(m, tools=[], prompt=SPECIALISTS[name], name=name))
    sup_model = GeminiChat(model_name=os.environ["AGENTSLIM_MODEL"], agent_tag="supervisor")
    app = create_supervisor(
        agents, model=sup_model,
        prompt=("You manage a team: " + ", ".join(keep_specialists) +
                ". Route the question to whichever specialists help, then give the "
                "final answer yourself."),
    ).compile()
    return app


def _last_ai_text(result) -> str:
    msgs = result.get("messages", [])
    for m in reversed(msgs):
        if getattr(m, "type", "") == "ai" and getattr(m, "content", ""):
            return m.content
    return msgs[-1].content if msgs else ""


def judge(q, ref, ans) -> float:
    out = LLM(model=os.environ["AGENTSLIM_MODEL"]).complete(
        "You are a strict grader. Reply with only an integer 0-10.",
        f"Question: {q}\nReference answer: {ref}\nCandidate answer: {ans}\n"
        f"Score 0-10 for correctness (10 = matches the reference).",
        agent="judge")
    m = re.search(r"\d+", out)
    return max(0.0, min(1.0, int(m.group(0)) / 10.0)) if m else 0.0


def run_team(keep_specialists, repeats=1):
    scores, traces = [], []
    for _ in range(repeats):
        for q, ref in QA:
            app = build_team(keep_specialists)
            tr = Trace(task_id=q[:24])
            with tr:
                res = app.invoke({"messages": [{"role": "user", "content": q}]})
                ans = _last_ai_text(res)
                tr.final_output = ans
            scores.append(judge(q, ref, ans))
            traces.append(tr)
    return scores, traces


def run_single(repeats=1):
    """No supervisor, no team: one generalist agent."""
    scores, traces = [], []
    for _ in range(repeats):
        for q, ref in QA:
            tr = Trace(task_id=q[:24])
            with tr:
                ans = LLM(model=os.environ["AGENTSLIM_MODEL"]).complete(
                    "You are a careful expert. Answer clearly and correctly.", q,
                    agent="generalist")
                tr.final_output = ans
            scores.append(judge(q, ref, ans))
            traces.append(tr)
    return scores, traces


if __name__ == "__main__":
    import json
    repeats = int(os.environ.get("AGENTSLIM_REPEATS", "2"))
    configs = {
        "supervisor + 3 specialists": ["researcher", "analyst", "writer"],
        "supervisor + 2 (no researcher)": ["analyst", "writer"],
        "supervisor + 1 (writer only)": ["writer"],
    }
    out = {}
    for name, keep in configs.items():
        sc, traces = run_team(keep, repeats)
        calls = sum(t.n_calls for t in traces) / len(traces)
        out[name] = {"score_mean": round(sum(sc) / len(sc), 3),
                     "scores": [round(x, 2) for x in sc], "avg_calls": round(calls, 2)}
        print(f"{name:<34} score={out[name]['score_mean']:.3f} calls/q={calls:.1f} "
              f"(real ${METER.spent:.3f})")
    sc, traces = run_single(repeats)
    out["single generalist agent"] = {
        "score_mean": round(sum(sc) / len(sc), 3),
        "scores": [round(x, 2) for x in sc],
        "avg_calls": round(sum(t.n_calls for t in traces) / len(traces), 2)}
    print(f"{'single generalist agent':<34} score={out['single generalist agent']['score_mean']:.3f} "
          f"calls/q={out['single generalist agent']['avg_calls']:.1f} (real ${METER.spent:.3f})")

    with open(os.path.join(_ROOT, "results", "langgraph_supervisor_team.json"), "w") as f:
        json.dump({"framework": "langgraph + langgraph-supervisor (create_supervisor)",
                   "configs": out, "model": os.environ.get("AGENTSLIM_MODEL"),
                   "spend": {"live_usd": round(METER.spent, 5), "live_calls": METER.calls,
                             "modeled_usd": round(METER.modeled, 5)}}, f, indent=2)
    print(f"\nreal API spend ${METER.spent:.4f} / {METER.calls} live calls "
          f"-> results/langgraph_supervisor_team.json")
