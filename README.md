# Agent right-sizing

**Which agents in a multi-agent system exist for no structural reason — and can be
removed with proof that the outcome is unchanged?**

> Not "shrink to 1." Read **[WHY_MULTIAGENT.md](WHY_MULTIAGENT.md)** first — it explains
> why an accuracy-only sweep always (wrongly) collapses to one agent, and what to
> measure instead.

Companies run multi-agent for **context capacity, least-privilege / blast-radius
scoping, parallel throughput, org ownership, model heterogeneity, compliance gating,
reliability, and human-in-the-loop gates** — not for task accuracy. A right-sizing
tool must respect those boundaries and only touch agents that are pure
task-decomposition with no structural justification.

---

## Repo layout

```
WHY_MULTIAGENT.md   the reframe — read this first
RESEARCH.md         literature synthesis: why / when / how (Google 2512.08296,
                    Tran & Kiela, MAST, error-compounding)
SURVEY.md           methods survey (AgentPrune, MaAS, AFlow, ADAS, …) + critique

agentslim/
  system.py         MultiAgentSystem DAG + Agent with STRUCTURAL METADATA
                    (role_type, tools, permissions, data_scope, owner, parallel_group)
  optimize.py       structural_report()  -> per-agent KEEP/SWEEP from metadata alone
                    pareto_scan()        -> sweeps ONLY unjustified agents, reports
                                            per-agent disposition + why each was kept
  ablations.py      remove / identity / heuristic / downgrade / merge — all REFUSE
                    to cross a structural boundary (BoundaryViolation) unless force=True
  eval.py           run a system on a task set -> quality + cost stats
  trace.py          per-LLM-call trace + causal graph
  llm.py            backends: Vertex Gemini (free-trial credits), Groq, mock;
                    disk cache + rate limiter + hard $ ceiling
  adapters/         route real repos through our transport, keep their orchestration:
                    crewai_shim, chatdev_shim, langchain_shim, autogen_shim, litellm_shim

systems/
  enterprise_support.py   demo: 7-agent support system where pareto_scan correctly
                          KEEPS 5 (tool/permission boundaries + a compliance gate)

archive/            the earlier accuracy-only experiments and their results. Kept
                    for reference (they show how to instrument ChatDev / CrewAI /
                    LangGraph / AutoGen). Their "collapse to 1/3" numbers are the
                    measurement artifact WHY_MULTIAGENT.md §0 is about — do not cite
                    them as findings.
```

## Quick start

```bash
python -m venv .venv && .venv/bin/pip install -e .
AGENTSLIM_BACKEND=mock .venv/bin/python -c "
from agentslim import pareto_scan
import systems.enterprise_support as es
r = pareto_scan(es.build(), es.tasks(), repeats=2)
print(r['recommendation']['note'])
for a, d in r['recommendation']['disposition'].items(): print(f'  {a}: {d}')
"
```

Real backends: put `GOOGLE_CLOUD_PROJECT` / `GOOGLE_CLOUD_LOCATION=global` /
`AGENTSLIM_MODEL=gemini-3.5-flash` (Vertex, ADC auth) and/or `GROQ_API_KEY` in `.env`.
See `agentslim/llm.py` for the full env contract and the spend ceiling.

## Status

Foundation + reframe done. **Next:** adapters must extract `tools` / `permissions` /
`owner` from real frameworks (WHY_MULTIAGENT.md §4 has the field→source table), then
run `structural_report` + `pareto_scan` on real repos and report per-agent verdicts —
not aggregate agent counts.
