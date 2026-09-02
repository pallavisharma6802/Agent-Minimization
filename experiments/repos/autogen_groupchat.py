"""Real framework #3: microsoft/autogen (AutoGen / AG2, ~40k+ ★) — the GroupChat
pattern (`SelectorGroupChat`: a manager LLM picks who speaks next; agents share
one conversation — a genuine team, not a sequential pipeline).

We build a 5-role team, run it on hard reasoning + coding tasks, and ablate by
dropping roles. Model: Groq `openai/gpt-oss-20b` (weak enough to leave headroom
so a real team could help). Judge: Groq `openai/gpt-oss-120b` (stronger, separate).

Question: does the manager + 5 specialists beat a 2-agent (solver+verifier) or a
single solver, on tasks where one weak agent is unreliable?
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
for _l in open(os.path.join(_ROOT, ".env")):
    if "=" in _l and not _l.startswith("#"):
        k, v = _l.strip().split("=", 1)
        os.environ.setdefault(k, v.strip().strip('"').strip("'"))

from autogen_agentchat.agents import AssistantAgent          # noqa: E402
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination  # noqa: E402
from autogen_agentchat.teams import SelectorGroupChat, RoundRobinGroupChat        # noqa: E402
from autogen_core.models import ModelInfo, UserMessage       # noqa: E402
from autogen_ext.models.openai import OpenAIChatCompletionClient  # noqa: E402

GROQ = dict(base_url="https://api.groq.com/openai/v1", api_key=os.environ["GROQ_API_KEY"],
            model_info=ModelInfo(vision=False, function_calling=True, json_output=False,
                                 family="unknown", structured_output=False))
WORKER = os.environ.get("AGENTSLIM_MODEL_CODER", "qwen/qwen3.6-27b")
JUDGE = os.environ.get("AGENTSLIM_JUDGE", "openai/gpt-oss-120b")


class _Retry(OpenAIChatCompletionClient):
    """Groq free tier is ~8k tokens/min — wrap create() with 429 backoff."""
    async def create(self, *a, **k):
        import asyncio as _aio
        delay = 3.0
        for i in range(8):
            try:
                return await super().create(*a, **k)
            except Exception as e:  # noqa: BLE001
                if "429" not in str(e) and "rate_limit" not in str(e).lower():
                    raise
                m = re.search(r"try again in ([\d.]+)s", str(e))
                await _aio.sleep(float(m.group(1)) + 1 if m else delay)
                delay = min(delay * 1.5, 30)
        return await super().create(*a, **k)


def _mc(model):
    return _Retry(model=model, temperature=0.0, max_tokens=700, **GROQ)


ROLES = {
    "planner": "You break the problem into steps. Be brief. Do not give the final answer.",
    "solver": "You carry out the plan and produce a candidate answer with working.",
    "critic": "You look for errors in the solver's work and point them out specifically.",
    "verifier": "You independently recompute / re-derive and check the candidate answer.",
    "finalizer": "You state the single final answer. Reply exactly 'FINAL: <answer>' and nothing else.",
}

TASKS = [
    # 6 items to fit Groq free-tier daily budget
    ("What is the remainder when 7^100 is divided by 13?", "9"),
    ("How many positive integers below 1000 are divisible by neither 3 nor 7?", "571"),
    ("A snail climbs 3m up a 10m well by day and slips 2m each night. Which day does it reach the top?", "8"),
    ("If today is Wednesday, what day is it 100 days from now?", "Friday"),
    ("Three people pay $30 for a room ($10 each). The clerk refunds $5; the bellhop pockets $2 and "
     "returns $1 to each. Each paid $9 = $27, plus $2 = $29. Where is the missing dollar?",
     "no missing dollar; the $27 already includes the bellhop's $2"),
    ("What is the 12th Fibonacci number if F(1)=F(2)=1?", "144"),
]

CONFIGS = {
    "team of 5 (planner/solver/critic/verifier/finalizer)":
        ["planner", "solver", "critic", "verifier", "finalizer"],
    "solver + verifier + finalizer (3)": ["solver", "verifier", "finalizer"],
    "solver + finalizer (2)": ["solver", "finalizer"],
    "solver only (1)": ["solver"],
}


def build_team(keep):
    agents = []
    for name in keep:
        agents.append(AssistantAgent(name=name, model_client=_mc(WORKER),
                                     system_message=ROLES[name]))
    if len(agents) == 1:
        return agents[0], None
    term = TextMentionTermination("FINAL:") | MaxMessageTermination(len(keep) + 1)
    team = RoundRobinGroupChat(agents, termination_condition=term)
    return None, team


async def _judge(q, ref, ans) -> float:
    mc = _mc(JUDGE)
    r = await mc.create([UserMessage(
        content=(f"Question: {q}\nReference answer: {ref}\nCandidate answer: {ans}\n"
                 "Is the candidate correct? Reply with only YES or NO."), source="user")])
    return 1.0 if "yes" in str(r.content).lower()[:5] else 0.0


async def run_config(keep, repeats=1):
    scores, msg_counts = [], []
    for _ in range(repeats):
        for q, ref in TASKS:
            solo, team = build_team(keep)
            if solo is not None:
                r = await solo.run(task=q)
                ans = r.messages[-1].content
                nmsg = sum(1 for m in r.messages if getattr(m, "source", "") in keep)
            else:
                r = await team.run(task=q)
                ans = r.messages[-1].content
                nmsg = sum(1 for m in r.messages
                           if getattr(m, "source", "") in keep or getattr(m, "source", "") == "selector")
            m = re.search(r"FINAL:\s*(.+)", str(ans), re.S)
            ans = m.group(1).strip() if m else str(ans)
            scores.append(await _judge(q, ref, ans))
            msg_counts.append(nmsg)
            await asyncio.sleep(4)  # stay under Groq free-tier TPM
    return scores, msg_counts


async def main():
    repeats = int(os.environ.get("AGENTSLIM_REPEATS", "1"))
    out = {}
    for name, keep in CONFIGS.items():
        sc, mc = await run_config(keep, repeats)
        out[name] = {"score_mean": round(sum(sc) / len(sc), 3),
                     "scores": [round(x, 2) for x in sc],
                     "avg_llm_calls": round(sum(mc) / len(mc), 1), "n_agents": len(keep)}
        print(f"{name:<52} score={out[name]['score_mean']:.3f} "
              f"calls/task={out[name]['avg_llm_calls']:.1f}")
    with open(os.path.join(_ROOT, "results", "autogen_groupchat.json"), "w") as f:
        json.dump({"framework": "microsoft/autogen RoundRobinGroupChat",
                   "worker_model": WORKER, "judge_model": JUDGE, "configs": out}, f, indent=2)
    print("-> results/autogen_groupchat.json")


if __name__ == "__main__":
    asyncio.run(main())
