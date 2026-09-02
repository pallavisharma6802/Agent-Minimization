"""Real repo #4 (5+ agents): OpenBMB/ChatDev (classic v1.1.6) — a 7-role virtual
software company (CEO, CPO, CTO, Programmer, Code Reviewer, Software Test
Engineer; + Chief Creative Officer for GUI, unused here).

We route every agent turn through Vertex Gemini (agentslim `chatdev_shim`), run
the company on small self-contained coding specs, and score the produced program
by actually executing it against a hidden functional check.

Ablation = which PHASES run (ChatChainConfig), which maps to which agents
participate:

  DemandAnalysis   CPO+CEO        LanguageChoose  CTO+CEO
  Coding           Programmer+CTO CodeCompleteAll Programmer+CTO
  CodeReview       Reviewer+Prog  Test            Tester+Prog
  EnvironmentDoc   Programmer+CTO Manual          CPO+CEO

Configs:
  full            all phases (6 agents)
  no_manual_docs  drop EnvironmentDoc + Manual
  no_review       drop CodeReview            -> removes Code Reviewer
  no_test         drop Test                  -> removes Test Engineer
  no_review_test  drop both                  -> removes Reviewer + Tester
  coding_only     LanguageChoose + Coding + CodeCompleteAll (CEO+CTO+Programmer)

Run with the ChatDev venv:
  .venv-chatdev/bin/python experiments/repos/chatdev_company.py
Env: CHATDEV_HOME=<clone of ChatDev@v1.1.6>, plus the Vertex .env.
"""
from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
_envf = os.path.join(_ROOT, ".env")
if os.path.exists(_envf):
    for _l in open(_envf):
        _l = _l.strip()
        if _l and not _l.startswith("#") and "=" in _l:
            k, v = _l.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
os.environ.setdefault("OPENAI_API_KEY", "sk-agentslim-dummy")
os.environ.setdefault("AGENTSLIM_MAX_TOKENS", "4096")

CHATDEV_HOME = os.environ.get(
    "CHATDEV_HOME",
    os.path.join(_ROOT, "..", "..", "scratchpad", "ChatDev"))
sys.path.insert(0, CHATDEV_HOME)

from agentslim.adapters.chatdev_shim import install as install_shim  # noqa: E402
from agentslim.llm import METER  # noqa: E402
from agentslim.trace import Trace  # noqa: E402

# --- phase sets -------------------------------------------------------
_ALL = ["DemandAnalysis", "LanguageChoose", "Coding", "CodeCompleteAll",
        "CodeReview", "Test", "EnvironmentDoc", "Manual"]
CONFIGS = {
    "full (6 agents)": _ALL,
    "no manual/docs": [p for p in _ALL if p not in ("EnvironmentDoc", "Manual")],
    "no CodeReview": [p for p in _ALL if p != "CodeReview"],
    "no Test": [p for p in _ALL if p != "Test"],
    "no Review+Test": [p for p in _ALL if p not in ("CodeReview", "Test")],
    "coding only (CEO+CTO+Prog)": ["LanguageChoose", "Coding", "CodeCompleteAll"],
}

# --- tasks: spec + hidden functional check ---------------------------
TASKS = [
    {
        "name": "ExprCalc",
        "prompt": ("Develop a command-line calculator. It takes a single arithmetic "
                   "expression string as sys.argv[1] (supporting + - * / parentheses and "
                   "integers) and prints only the numeric result to stdout. Entry point main.py."),
        "check": [(["2+3*4"], "14"), (["(10-2)/4"], "2"), (["7*7"], "49")],
    },
    {
        "name": "WordCount",
        "prompt": ("Develop a command-line tool. Given a filename as sys.argv[1], it prints "
                   "the number of words in that file to stdout (words separated by whitespace). "
                   "Entry point main.py."),
        "check": "wordcount",
    },
    {
        "name": "FizzBuzz",
        "prompt": ("Develop a program with entry point main.py that takes an integer N as "
                   "sys.argv[1] and prints the FizzBuzz sequence from 1 to N, one item per line "
                   "(Fizz for multiples of 3, Buzz for 5, FizzBuzz for both, else the number)."),
        "check": [(["5"], "1\n2\nFizz\n4\nBuzz")],
    },
    # --- bug-prone: a lone coder usually ships an off-by-one / missed edge case ---
    {
        "name": "RomanToInt",
        "prompt": ("Develop a program, entry point main.py, that takes a Roman numeral string "
                   "as sys.argv[1] and prints its integer value. Must handle subtractive forms "
                   "(IV=4, IX=9, XL=40, XC=90, CD=400, CM=900)."),
        "check": [(["IV"], "4"), (["XL"], "40"), (["MCMXciv".upper()], "1994"),
                  (["LVIII"], "58"), (["CDXLIV"], "444")],
    },
    {
        "name": "NthPrime",
        "prompt": ("Develop a program, entry point main.py, that takes a positive integer n as "
                   "sys.argv[1] and prints the n-th prime number (1-indexed, so n=1 -> 2)."),
        "check": [(["1"], "2"), (["6"], "13"), (["25"], "97"), (["100"], "541")],
    },
    {
        "name": "BalancedBrackets",
        "prompt": ("Develop a program, entry point main.py, that takes a string as sys.argv[1] "
                   "and prints 'True' if all (), [], {} brackets are balanced and correctly "
                   "nested, else 'False'. Ignore non-bracket characters."),
        "check": [(["(a[b]{c})"], "True"), (["([)]"], "False"), (["{{}}"], "True"),
                  (["(]"], "False"), (["a(b)c"], "True")],
    },
]


def _find_warehouse(name: str) -> str | None:
    wh = os.path.join(CHATDEV_HOME, "WareHouse")
    cands = [d for d in os.listdir(wh) if d.startswith(name + "_")]
    if not cands:
        return None
    return os.path.join(wh, sorted(cands)[-1])


def score_project(proj_dir: str, task: dict) -> float:
    if not proj_dir or not os.path.isdir(proj_dir):
        return 0.0
    pys = [f for f in os.listdir(proj_dir) if f.endswith(".py")]
    if not pys:
        return 0.0
    s = 0.2  # produced code
    for f in pys:
        try:
            ast.parse(open(os.path.join(proj_dir, f)).read())
        except SyntaxError:
            return s
    s = 0.4  # all compiles
    entry = "main.py" if "main.py" in pys else pys[0]

    checks = task["check"]
    passed = tot = 0
    with tempfile.TemporaryDirectory() as tmp:
        for f in pys:
            shutil.copy(os.path.join(proj_dir, f), tmp)
        if task["check"] == "wordcount":
            open(os.path.join(tmp, "in.txt"), "w").write("the quick brown fox jumps")
            checks = [(["in.txt"], "5")]
        for argv, expected in checks:
            tot += 1
            try:
                r = subprocess.run([sys.executable, entry, *argv], cwd=tmp,
                                   capture_output=True, text=True, timeout=20)
                if expected.strip() in r.stdout.strip() or r.stdout.strip() == expected.strip():
                    passed += 1
            except Exception:
                pass
    if tot:
        s = 0.4 + 0.6 * (passed / tot)
    return round(s, 3)


def run_config(phases: list[str], task: dict, tag: str) -> tuple[float, Trace]:
    # ChatDev uses relative paths + bare `from utils import ...` everywhere;
    # it only works with CWD == its repo root and the root first on sys.path.
    os.chdir(CHATDEV_HOME)
    if sys.path[0] != CHATDEV_HOME:
        sys.path.insert(0, CHATDEV_HOME)
    from camel.typing import ModelType
    from chatdev.chat_chain import ChatChain

    install_shim()
    cfgroot = os.path.join(CHATDEV_HOME, "CompanyConfig", "Default")
    base_chain = json.load(open(os.path.join(cfgroot, "ChatChainConfig.json")))
    base_chain["chain"] = [p for p in base_chain["chain"] if p["phase"] in phases]
    base_chain["recruitments"] = base_chain.get("recruitments", [])

    tmpcfg = os.path.join(CHATDEV_HOME, "CompanyConfig", f"_ab_{tag}")
    os.makedirs(tmpcfg, exist_ok=True)
    json.dump(base_chain, open(os.path.join(tmpcfg, "ChatChainConfig.json"), "w"))

    name = task["name"] + "_" + tag
    cc = ChatChain(
        config_path=os.path.join(tmpcfg, "ChatChainConfig.json"),
        config_phase_path=os.path.join(cfgroot, "PhaseConfig.json"),
        config_role_path=os.path.join(cfgroot, "RoleConfig.json"),
        task_prompt=task["prompt"], project_name=name,
        org_name="agentslim", model_type=ModelType.GPT_4O_MINI, code_path="")
    import logging
    os.makedirs(os.path.dirname(cc.log_filepath), exist_ok=True)
    open(cc.log_filepath, "a").close()
    for h in list(logging.root.handlers):
        logging.root.removeHandler(h)
    logging.basicConfig(filename=cc.log_filepath, level=logging.INFO,
                        format="[%(asctime)s %(levelname)s] %(message)s",
                        datefmt="%Y-%d-%m %H:%M:%S", encoding="utf-8", force=True)
    cc.pre_processing()
    cc.make_recruitment()
    tr = Trace(task_id=name)
    with tr:
        cc.execute_chain()
    try:
        cc.post_processing()
    except Exception:
        pass
    sc = score_project(_find_warehouse(name), task)
    tr.final_output = f"score={sc}"
    return sc, tr


if __name__ == "__main__":
    repeats = int(os.environ.get("AGENTSLIM_REPEATS", "1"))
    out = {}
    for cname, phases in CONFIGS.items():
        scores, calls = [], []
        for r in range(repeats):
            for task in TASKS:
                tag = f"{cname.split()[0].lower()}{r}"
                try:
                    sc, tr = run_config(phases, task, tag)
                except Exception as e:
                    print(f"  ! {cname}/{task['name']}: {e}")
                    sc, tr = 0.0, Trace()
                scores.append(sc)
                calls.append(tr.n_calls)
        out[cname] = {"score_mean": round(sum(scores) / len(scores), 3),
                      "scores": [round(x, 2) for x in scores],
                      "avg_calls": round(sum(calls) / len(calls), 1),
                      "n_phases": len(phases)}
        print(f"{cname:<28} score={out[cname]['score_mean']:.3f} "
              f"calls/build={out[cname]['avg_calls']:.1f} "
              f"phases={len(phases)} (real ${METER.spent:.2f})")

    with open(os.path.join(_ROOT, "results", "chatdev_company.json"), "w") as f:
        json.dump({"repo": "OpenBMB/ChatDev @ v1.1.6 (7-role software company)",
                   "configs": out, "model": os.environ.get("AGENTSLIM_MODEL"),
                   "spend": {"live_usd": round(METER.spent, 5), "live_calls": METER.calls,
                             "modeled_usd": round(METER.modeled, 5)}}, f, indent=2)
    print(f"\nreal API spend ${METER.spent:.4f} / {METER.calls} live calls "
          f"-> results/chatdev_company.json")
