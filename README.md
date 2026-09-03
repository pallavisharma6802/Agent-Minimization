# Agent Minimization

**Which agents in a multi-agent system exist for no structural reason?**

> ⚠️ **Read [WHY_MULTIAGENT.md](WHY_MULTIAGENT.md) first.** Early runs of this project
> kept "shrinking to 1 agent" — that was a **measurement artifact**: the tool only
> scored task accuracy, and companies don't build multi-agent for accuracy. They build
> it for context capacity, permission scoping, parallel throughput, org ownership, model
> heterogeneity, compliance gating and reliability. The tool now respects those
> boundaries and only sweeps agents that are pure task-decomposition with no structural
> justification.

Multi-agent LLM systems mix agents that are load-bearing for hard structural reasons
with agents that are cargo-cult sequential decomposition. The second kind is common,
adds cost and latency, and — per UC Berkeley's MAST and our ChatDev weak-coder run —
sometimes *degrades* quality through error compounding. This project tells the two apart
on **real open-source systems**, respecting every structural boundary, and backs each
"removable" call with an ablation.

---

## What we're building

Three artifacts:

### 1. MA-Bloat corpus
A public dataset of 30–50 actively-maintained open-source multi-agent systems (LangGraph +
CrewAI first), each with:
- a minimal task eval with programmatic scoring
- full call-graph instrumentation (every LLM call: agent, role prompt, input context, tools, output, tokens, cost, latency)
- ablation results (remove / merge / downgrade / replace-with-heuristic for each agent)
- a hand label for **why each agent exists**: real specialization · context isolation · redundant · demo theater

### 2. `agentslim` — a refactoring assistant
Point it at a multi-agent repo. It instruments the system, runs shadow ablations against
the repo's eval, and emits:
- ranked refactor suggestions with **measured metric deltas** vs a noise band
- a code diff for each suggestion
- cost / latency / token savings

Semi-automated first (human confirms each suggestion), more autonomous later.

### 3. Risk-surface report
For each system, before/after a refactor:
- number of LLM calls, total context tokens, $ / task
- number of inter-agent trust edges
- prompt-injection propagation reachability (inject into one tool output, measure spread)
- privilege union per agent
- collusion capacity (can any two agents coordinate to defeat a guardrail that binds each)

Shows the Pareto frontier of capability vs risk surface, and demonstrates that most
reductions cost little or no capability.

---

## Method: causal agent ablation

The core primitive. Instead of asking "does this agent's output *correlate* with the final
answer", we intervene:

- **remove** — route the agent's input straight to its consumer
- **identity** — replace its output with its input passed through
- **heuristic** — replace it with a non-LLM rule (regex router, switch statement)
- **merge** — fold its role prompt and tools into an adjacent agent
- **downgrade** — swap to a cheaper model

We measure the task-metric delta against a noise band (baseline run 5×, since multi-agent
systems are high-variance) plus the cost/latency delta. Each agent is then classified:
**redundant · mergeable · context-isolation-only · load-bearing**.

**Capability agents** (do work) are candidates for reduction.
**Oversight agents** (monitors, safety critics) are kept or strengthened — never cut.

---

## Hypotheses (pre-registered)

- **H1** — >40% of agents in OSS multi-agent repos can be merged/removed with <2% task-metric change.
- **H2** — the dominant *real* reason for multiple agents is context-window isolation, not capability specialization.
- **H3** — agent count correlates with failure rate more strongly than with task success.
- **H4** — reducing agents shrinks prompt-injection blast radius roughly linearly with removed inter-agent trust edges.

---

## Plan

| Phase | Weeks | Output |
|---|---|---|
| **0 — Scoping** | 1–2 | Lock frameworks (LangGraph + CrewAI, maybe AutoGen). Repo selection criteria: ≥N stars, commit in last 90 days, runnable entrypoint, ≥3 LLM-calling agents, checkable output. Write H1–H4 pre-registration. |
| **1 — Instrumentation harness** | 3–4 | Framework adapters that intercept every LLM call via callbacks / client monkeypatch. Build the causal graph (which outputs feed which inputs, which reach the final answer). Static pass for degenerate agents. |
| **2 — Minimal evals** | 4–6 | 20–50 scored task instances per repo. Baseline metric + noise band. *This is the expensive part.* |
| **3 — Ablations** | 3–4 | Run remove/identity/heuristic/merge/downgrade per agent. Classify each. Build ground-truth labels. |
| **4 — Tool** | 3–4 | Package Phases 1+3 into `agentslim <path>`. Validate: does its ranking match the hand analysis? Report precision/recall of "safe to remove". |
| **5 — Risk surface** | 2–3 | Implement the risk metrics. Recompute pre/post refactor. Run the injection-propagation test. |

~4–5 months solo; 2–3 with help on evals.

### Immediate next steps
1. Lit-check whether "causal mediation / intervention analysis for agent necessity in LLM multi-agent systems" is genuinely unpublished — if so, that's the method contribution.
2. Hand-instrument **5 pilot repos** (LangGraph + CrewAI) and get real numbers on 1–2 before scaling anything.
3. Draft the risk-surface metric list; get one safety researcher to sanity-check it.

---

## Deliverables

- The corpus — public
- `agentslim` — public
- A measurement paper (arXiv + an agents workshop)
- A write-up with the headline numbers: *"median OSS multi-agent system — X% of LLM calls removable, $Y/1k tasks saved, Z fewer injection paths, no accuracy loss."*

---

## Prior work this builds on / around

- **AgentPrune** (ICLR'25) — one-shot communication-graph pruning, 28–73% token cut. Prunes *edges on a standardized abstraction*, not agents in real codebases.
- **MaAS** (ICML'25 Oral) — learns a distribution over architectures, samples per query, 6–45% of baseline cost. Generates from scratch; doesn't refactor existing systems.
- **AFlow / ADAS / GPTSwarm** — automated agentic workflow search. Same limitation: their own DSL, academic benchmarks.
- **UC Berkeley MAST** — "Why Do Multi-Agent LLM Systems Fail?" 7 frameworks, 1,642 traces, 41–86.7% failure rates, ~37% rooted in system/agent-design.
- **Cognition, "Don't Build Multi-Agents"** — fan-out breaks shared-state tasks; single agent + context engineering often wins.
- **Cooperative AI, "Multi-Agent Risks from Advanced AI"** — collusion, cascading failure, emergent coordination as risks absent in single-agent systems.

**The gap we fill:** all efficiency work is on ~6 academic benchmarks. Nobody has measured
bloat on real deployed systems, built a tool that refactors them in their own framework, or
accounted for the safety surface that reduction removes.

See [`SURVEY.md`](SURVEY.md) for the full literature review and critique.
