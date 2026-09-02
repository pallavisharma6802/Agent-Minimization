"""Real repo #1: crewAIInc/crewAI-examples  ->  crews/game-builder-crew

A 3-agent SEQUENTIAL CrewAI pipeline, verbatim role/goal/backstory/task text
from the repo's YAML:

  senior_engineer_agent  -> code_task     (writes full python game code)
  qa_engineer_agent      -> review_task   (checks code, returns full code)
  chief_qa_engineer_agent-> evaluate_task (final review, returns full code)

Final output = last task output. We reconstruct the crew from the repo's config
files (not their @CrewBase class) so agents/tasks can be sliced for ablation;
the prompts are identical to what the repo ships.

Hypothesis: this is the canonical "redundant critic chain" — evaluate_task
(and possibly review_task) contribute little measurable quality over code_task.

Metric per run (0..1), averaged over the game specs:
  0.34  final answer contains a python code block that compiles
  0.33  AST has a callable entrypoint + a loop (game loop present)
  0.33  code references the spec's core nouns (food/snake/grid etc.)

Setup:
  pip install crewai
  repo cloned at $CREW_EXAMPLES  (default: scratchpad/crewAI-examples)
  .env with Vertex Gemini config (litellm shim routes all calls there)
"""
from __future__ import annotations

import ast
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

os.environ.setdefault("AGENTSLIM_MAX_TOKENS", "6144")  # code generation needs room

from agentslim.adapters.crewai_shim import install as install_shim  # noqa: E402
from agentslim.trace import Trace  # noqa: E402

CREW_DIR = os.environ.get(
    "CREW_EXAMPLES",
    os.path.join(_ROOT, "..", "..", "scratchpad", "crewAI-examples"),
)
GB = os.path.join(CREW_DIR, "crews", "game-builder-crew", "src", "game_builder_crew", "config")


def _cfg():
    import yaml
    agents = yaml.safe_load(open(os.path.join(GB, "agents.yaml")))
    tasks = yaml.safe_load(open(os.path.join(GB, "tasks.yaml")))
    games = yaml.safe_load(open(os.path.join(GB, "gamedesign.yaml")))
    return agents, tasks, games


# (agent_key, task_key) in repo order
PIPELINE = [
    ("senior_engineer_agent", "code_task"),
    ("qa_engineer_agent", "review_task"),
    ("chief_qa_engineer_agent", "evaluate_task"),
]
KIND = {"senior_engineer_agent": "capability",
        "qa_engineer_agent": "oversight",          # a reviewer -> protect by default
        "chief_qa_engineer_agent": "oversight"}


def build_crew(keep: list[str] | None = None, merges: list[tuple[str, str]] | None = None):
    """Return a crewai.Crew reconstructed from repo config, limited to `keep`
    agent keys (default all), with optional (a,b) merges folding b into a."""
    from crewai import Agent, Crew, Process, Task

    agents_cfg, tasks_cfg, _ = _cfg()
    keep = keep or [a for a, _ in PIPELINE]
    merged_backstory: dict[str, str] = {}
    drop: set[str] = set()
    for a, b in (merges or []):
        merged_backstory[a] = agents_cfg[a]["backstory"] + "\nAlso: " + agents_cfg[b]["backstory"]
        drop.add(b)

    agent_objs: dict[str, object] = {}
    tasks: list = []
    for akey, tkey in PIPELINE:
        if akey not in keep or akey in drop:
            continue
        cfg = dict(agents_cfg[akey])
        if akey in merged_backstory:
            cfg["backstory"] = merged_backstory[akey]
        if akey not in agent_objs:
            agent_objs[akey] = Agent(role=cfg["role"], goal=cfg["goal"],
                                     backstory=cfg["backstory"],
                                     allow_delegation=False, verbose=False)
        tasks.append(Task(description=tasks_cfg[tkey]["description"],
                          expected_output=tasks_cfg[tkey]["expected_output"],
                          agent=agent_objs[akey]))
    return Crew(agents=list(agent_objs.values()), tasks=tasks,
               process=Process.sequential, verbose=False)


# --------------------------- scoring ---------------------------------
_CODE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.S)
_OPEN_RE = re.compile(r"```(?:python)?\s*(.*)$", re.S)


def _extract_code(text: str) -> str:
    text = text or ""
    m = _CODE_RE.findall(text)
    if m:
        return max(m, key=len)
    m2 = _OPEN_RE.search(text)          # unclosed fence (truncated output)
    if m2:
        return m2.group(1)
    return text


def score(final_text: str, spec: str) -> float:
    code = _extract_code(final_text)
    s = 0.0
    tree = None
    try:
        tree = ast.parse(code)
        s += 0.34
    except SyntaxError:
        return s
    has_loop = any(isinstance(n, (ast.While, ast.For)) for n in ast.walk(tree))
    has_entry = any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(tree)) \
        or "__main__" in code
    if has_loop and has_entry:
        s += 0.33
    nouns = [w for w in re.findall(r"[a-z]{4,}", spec.lower())]
    core = [w for w in ("snake", "food", "grid", "score", "collision", "pygame",
                        "player", "board", "tile", "move") if w in nouns]
    if core and sum(1 for w in core if w in code.lower()) >= max(2, len(core) // 2):
        s += 0.33
    return round(s, 3)


# --------------------------- run + trace ----------------------------
def run_config(keep=None, merges=None, specs=("example3_snake",), repeats=1):
    install_shim()
    _, _, games = _cfg()
    accs, traces = [], []
    for _ in range(repeats):
        for sk in specs:
            spec = games[sk]
            crew = build_crew(keep, merges)
            tr = Trace(task_id=sk)
            with tr:
                out = crew.kickoff(inputs={"game": spec})
                tr.final_output = str(out)
            accs.append(score(str(out), spec))
            traces.append(tr)
    return accs, traces


if __name__ == "__main__":
    import json
    from agentslim.llm import METER

    specs = ("example3_snake", "example1_pacman", "example2_pacman")
    repeats = int(os.environ.get("AGENTSLIM_REPEATS", "2"))
    configs = {
        "baseline (3 agents)": dict(keep=None),
        "no chief_qa (2)": dict(keep=["senior_engineer_agent", "qa_engineer_agent"]),
        "engineer only (1)": dict(keep=["senior_engineer_agent"]),
        "merge qa->chief (2)": dict(merges=[("qa_engineer_agent", "chief_qa_engineer_agent")]),
    }
    out = {}
    for name, kw in configs.items():
        accs, traces = run_config(specs=specs, repeats=repeats, **kw)
        calls = sum(t.n_calls for t in traces) / len(traces)
        cost = sum(t.cost_usd for t in traces) / len(traces)
        out[name] = {"acc_mean": round(sum(accs) / len(accs), 3), "accs": accs,
                     "avg_calls": round(calls, 2), "avg_cost_usd": round(cost, 6)}
        print(f"{name:<24} acc={out[name]['acc_mean']:.3f}  calls/run={calls:.1f}  "
              f"${cost:.5f}  (spent ${METER.spent:.3f} total)")
    os.makedirs(os.path.join(_ROOT, "results"), exist_ok=True)
    with open(os.path.join(_ROOT, "results", "game_builder_crew.json"), "w") as f:
        json.dump({"configs": out, "spend": {"usd": round(METER.spent, 5), "calls": METER.calls},
                   "model": os.environ.get("AGENTSLIM_MODEL")}, f, indent=2)
    print(f"\ntotal spend ${METER.spent:.4f} / {METER.calls} calls -> results/game_builder_crew.json")
