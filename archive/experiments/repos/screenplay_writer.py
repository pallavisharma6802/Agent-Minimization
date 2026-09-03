"""Real repo #2: crewAIInc/crewA-examples :: crews/screenplay_writer

A 3-agent SEQUENTIAL transform pipeline (verbatim role/goal/backstory/task text
from the repo's YAML):

  analyst      -> distill the arguments from a discussion
  scriptwriter -> turn it into dialogue-only screenplay
  formatter    -> reformat into the "## (person):" template

The repo ships its own quality metric: a `scorer` agent that rates the final
script 1-10. We use exactly that as the objective (LLM-as-judge, the repo's own
rubric), averaged over several input discussions.

Hypotheses:
  - formatter is redundant (scriptwriter can emit the format directly)
  - analyst may or may not be load-bearing

Setup: pip install crewai ; repo cloned at $CREW_EXAMPLES ; Vertex .env.
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
os.environ.setdefault("AGENTSLIM_MAX_TOKENS", "4096")

from agentslim.adapters.crewai_shim import install as install_shim  # noqa: E402
from agentslim.llm import LLM, METER  # noqa: E402
from agentslim.trace import Trace  # noqa: E402

CREW_DIR = os.environ.get(
    "CREW_EXAMPLES",
    os.path.join(_ROOT, "..", "..", "scratchpad", "crewAI-examples"))
SW = os.path.join(CREW_DIR, "crews", "screenplay_writer", "config")

PIPELINE = [("analyst", "task1"), ("scriptwriter", "task2"), ("formatter", "task3")]
KIND = {"analyst": "capability", "scriptwriter": "capability", "formatter": "capability"}

# a few short discussion inputs (newsgroup-thread style, like the repo's sample)
DISCUSSIONS = [
    """A: I think remote work kills company culture; you lose the hallway chats.
B: Disagree. Culture is about trust and outcomes, not proximity. My team ships more remote.
A: But onboarding juniors is much harder without an office.
B: That's a real cost, I'll grant. We solved it with a mandatory first-month co-location.""",
    """X: The city should ban cars from downtown entirely.
Y: That would wreck small businesses that rely on drive-by customers.
X: Studies from Oslo and Ghent show foot traffic and sales actually rose after pedestrianization.
Y: Those cities had great transit first. Ours doesn't. Sequence matters.""",
    """P: Standardized testing is the fairest way to compare students across schools.
Q: It mostly measures family income and test-prep access, not ability.
P: Every alternative — GPA, essays, interviews — is even more subjective and gameable.
Q: Then fix the rubric and train reviewers; don't keep a broken proxy because grading is hard.""",
]


def _cfg():
    import yaml
    return (yaml.safe_load(open(os.path.join(SW, "agents.yaml"))),
            yaml.safe_load(open(os.path.join(SW, "tasks.yaml"))))


def build_crew(keep=None, merges=None):
    from crewai import Agent, Crew, Process, Task
    ac, tc = _cfg()
    keep = keep or [a for a, _ in PIPELINE]
    merged, drop = {}, set()
    for a, b in (merges or []):
        merged[a] = ac[a]["backstory"] + "\nAlso: " + ac[b]["goal"] + " " + ac[b]["backstory"]
        drop.add(b)
    objs, tasks = {}, []
    for akey, tkey in PIPELINE:
        if akey not in keep or akey in drop:
            continue
        cfg = dict(ac[akey])
        if akey in merged:
            cfg["backstory"] = merged[akey]
        objs.setdefault(akey, Agent(role=cfg["role"], goal=cfg["goal"],
                                    backstory=cfg["backstory"],
                                    allow_delegation=False, verbose=False))
        desc = tc[tkey]["description"].replace("{{discussion}}", "{discussion}")
        tasks.append(Task(description=desc, expected_output=tc[tkey]["expected_output"],
                          agent=objs[akey]))
    return Crew(agents=list(objs.values()), tasks=tasks,
               process=Process.sequential, verbose=False)


_SCORE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:/\s*10)?")


def repo_score(script: str) -> float:
    """Run the repo's own `scorer` agent; return score/10 in [0,1]."""
    ac, tc = _cfg()
    s = ac["scorer"]
    sysmsg = f"role: {s['role']}\n{s['goal']}\n{s['backstory']}"
    user = tc["task4"]["description"].replace("{{script}}", "{script}").format(script=script)
    out = LLM(model=os.environ["AGENTSLIM_MODEL"]).complete(sysmsg, user, agent="scorer")
    m = _SCORE_RE.search(out.strip())
    if not m:
        return 0.0
    return max(0.0, min(1.0, float(m.group(1)) / 10.0))


def run_config(keep=None, merges=None, repeats=1):
    install_shim()
    scores, traces = [], []
    for _ in range(repeats):
        for disc in DISCUSSIONS:
            crew = build_crew(keep, merges)
            tr = Trace(task_id=disc[:20])
            with tr:
                out = str(crew.kickoff(inputs={"discussion": disc}))
                tr.final_output = out
            scores.append(repo_score(out))
            traces.append(tr)
    return scores, traces


if __name__ == "__main__":
    import json
    repeats = int(os.environ.get("AGENTSLIM_REPEATS", "2"))
    configs = {
        "baseline (analyst+scriptwriter+formatter)": dict(keep=None),
        "no formatter (2)": dict(keep=["analyst", "scriptwriter"]),
        "no analyst (2)": dict(keep=["scriptwriter", "formatter"]),
        "scriptwriter only (1)": dict(keep=["scriptwriter"]),
        "merge analyst->scriptwriter (2)": dict(merges=[("scriptwriter", "analyst")]),
    }
    out = {}
    for name, kw in configs.items():
        sc, traces = run_config(repeats=repeats, **kw)
        calls = sum(t.n_calls for t in traces) / len(traces)
        cost = sum(t.cost_usd for t in traces) / len(traces)
        out[name] = {"score_mean": round(sum(sc) / len(sc), 3), "scores": [round(x, 2) for x in sc],
                     "avg_calls": round(calls, 2), "avg_cost_usd": round(cost, 6)}
        print(f"{name:<42} score={out[name]['score_mean']:.3f}  calls/run={calls:.1f}  "
              f"${cost:.5f}  (real ${METER.spent:.3f})")
    with open(os.path.join(_ROOT, "results", "screenplay_writer.json"), "w") as f:
        json.dump({"repo": "crewAIInc/crewA-examples :: crews/screenplay_writer",
                   "configs": out, "model": os.environ.get("AGENTSLIM_MODEL"),
                   "spend": {"live_usd": round(METER.spent, 5), "live_calls": METER.calls,
                             "modeled_usd": round(METER.modeled, 5)}}, f, indent=2)
    print(f"\nreal API spend ${METER.spent:.4f} / {METER.calls} live calls "
          f"-> results/screenplay_writer.json")
