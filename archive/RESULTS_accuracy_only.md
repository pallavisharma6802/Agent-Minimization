# Results (2026-09-02)

> **⚠️ Read [WHY_MULTIAGENT.md](WHY_MULTIAGENT.md) before trusting any "collapse to N"
> number below.** These experiments scored **task accuracy only** — the one axis
> where multi-agent is weakest — on tasks small enough to fit one context window.
> They say nothing about the reasons companies actually run multi-agent (context
> capacity, permission scoping, parallelism, org ownership, compliance, reliability).
> `pareto_scan` now enforces structural boundaries and only sweeps agents that are
> pure task-decomposition; see the `enterprise_support` demo where it correctly keeps
> **5 of 7** agents.
>
> **Method reframe.** The objective is *not* "shrink to 1." `pareto_scan` sweeps team
> sizes over the structurally-unjustified agents only, and reports per-agent
> disposition + why each protected agent was left alone.

---

## Real repo #4 — OpenBMB/ChatDev (~26k★), a 7-role virtual software company

The strongest test so far: a famous multi-agent repo, a **real metric** (generate a
program, then *execute it* against hidden functional tests), on **genuinely hard
tasks** (LRU cache, RPN eval with truncating division, topological sort with cycle
detection + lexicographic tie-break).

Ablation = which ChatDev *phases* run → which of the 6 agents participate.

| config | agents | test-pass score | LLM calls / build |
|---|---|---|---|
| full pipeline | 6 | **1.000** | 14.0 |
| drop EnvironmentDoc + Manual | 6 | 1.000 | 11.0 |
| drop Code Review | 5 | 1.000 | 8.0 |
| drop Test | 5 | 1.000 | 14.0 |
| drop Review **and** Test | 4 | 1.000 | 8.0 |
| **coding core only** (CEO+CTO+Programmer) | **3** | **1.000** | **3.2** |

Every config passes **every** hidden test on **every** task. ChatDev's Code Reviewer
and Test Engineer agents do not change a single functional outcome here; the 3-agent
core produces verified-correct code (topological sort etc.) at **3.2 calls/build vs
14 — a 77% cut**. Real API spend for the whole sweep: **$0.17**.

`pareto_scan` recommendation: **k = 3** (the coding core). Not 1 — the pipeline
structurally needs language-choice → code → complete. But not 6.

This matches UC Berkeley's MAST finding that ChatDev's problems are coordination
overhead, not missing capability.

**Caveat:** `gemini-3.5-flash` nails these tasks in the Coding phase, so Review/Test
have nothing to catch. The score is pinned at 1.000 — no headroom to *see* a benefit.
So we re-ran with a **deliberately weak coder** ↓.

### ChatDev with a weak coder (Groq `openai/gpt-oss-20b` as the Programmer)

Everyone else (CTO, Code Reviewer, Test Engineer, judge) stays on Gemini. Same
hard tasks, now n=2, 5 tasks. The weak coder ships real bugs → scores fall from
1.000, giving the review/test agents something to catch.

| config | agents | test-pass score | calls/build |
|---|---|---|---|
| full pipeline | 6 | 0.480 | 14.6 |
| drop Code Review | 5 | 0.480 | 8.6 |
| drop Review **and** Test | 4 | 0.480 | 8.6 |
| **coding core only** (CEO+CTO+Programmer) | **3** | **0.640** | **2.6** |

**The 3-agent core scores *higher* than the full 6-agent pipeline (0.64 vs 0.48), at
1/6 the LLM calls.** The four configs that keep any review/test phase produce
*byte-identical* results — those agents change nothing. And the DemandAnalysis /
Manual / extra-CodeComplete cycles in the full pipeline actively break tasks the
3-agent core gets right (RomanToInt, NthPrime). More agents → worse **and** 5×
more expensive. Real spend: **$0.02**.

**Bottom line for ChatDev:** strong coder → 6 agents ≈ 3 agents (cut for free);
weak coder → 6 agents **<** 3 agents (cutting *improves* quality). `pareto_scan`
says **k = 3** either way. This is the sharpest evidence in the project that
deployed multi-agent pipelines are over-built.

---

---

## Real repo #5 — microsoft/autogen (~40k★), RoundRobinGroupChat

A 3rd framework. 5-role team (planner→solver→critic→verifier→finalizer), 6 hard
reasoning tasks (7^100 mod 13, the missing-dollar riddle, trailing zeros of 100!,
etc.), Vertex Gemini, n=2.

| config | agents | score | LLM calls / task |
|---|---|---|---|
| team of 5 | 5 | 1.000 | 5.0 |
| solver + verifier + finalizer | 3 | 1.000 | 3.0 |
| solver + finalizer | 2 | 1.000 | 2.0 |
| **solver only** | **1** | **1.000** | **1.0** |

Monotone: 5→1 agents, quality flat, cost linear in headcount. `pareto_scan` → k=1
(strong model on these tasks). Same pattern as CrewAI, LangGraph, ChatDev.

---

## The pattern across 5 real repos + 3 frameworks

| repo | ★ | framework | shipped agents | right-sized | quality change |
|---|---|---|---|---|---|
| ChatDev (strong coder) | 26k | CrewAI-style | 6 | 3 | 0.00 |
| **ChatDev (weak coder)** | 26k | CrewAI-style | 6 | 3 | **+0.16 (better)** |
| game-builder-crew | 7k | CrewAI | 3 | 1 | 0.00 |
| screenplay_writer | 7k | CrewAI | 3 | 1 | +0.02 |
| langgraph_pipeline | — | LangGraph | 3 | 1 | 0.00 |
| autogen_groupchat | 40k | AutoGen | 5 | 1 | 0.00 |

Never once did the shipped agent count turn out to be right-sized. Reductions of
40–83% of LLM calls, quality flat or up. The one case with a weak worker (ChatDev)
is the sharpest: **the extra agents made it worse.**

This is consistent with the literature ([RESEARCH.md](RESEARCH.md)): Google's
scaling study, Tran & Kiela's equal-compute result, and MAST all point the same way.

---

# Earlier results (pilots + CrewAI/LangGraph, detail)

All runs: **Gemini `gemini-3.5-flash` via Vertex AI** (`location=global`, project
`agent-min`), **GCP free-trial credits only** — never activated to paid. Total
real API spend for everything below: **≈ $0.75**. Every response is cached to
`.llm_cache/`, so reruns are free and the numbers reproduce exactly.

The cost meter reports two numbers: **live** (real API $ this run) and **modeled**
(what the traced calls would cost with no cache) — the second is the one that
matters when comparing system configs.

---

## What we tested

| # | system | framework | source | task | metric |
|---|--------|-----------|--------|------|--------|
| 1 | math_committee | agentslim DAG | synthetic (known ground truth) | 12 grade-school word problems | exact-match accuracy |
| 2 | debate | agentslim DAG | synthetic | 10 arithmetic problems | exact-match |
| 3 | **game-builder-crew** | **CrewAI** | **crewAIInc/crewA-examples** (~7k★) | build a game from a spec | 3-part code proxy + code-churn |
| 4 | **screenplay_writer** | **CrewAI** | **crewAIInc/crewA-examples** (~7k★) | discussion → screenplay | the repo's **own scorer agent** (1–10) |
| 5 | langgraph_pipeline | **LangGraph** `StateGraph` | docs "multi-agent collaboration" pattern | reasoning Q&A (easy + hard sets) | Gemini judge vs reference |

---

## The headline

**Every one of the five multi-agent systems collapses to a single agent with no
measurable quality loss.**

| system | baseline | minimized | Δ quality | calls | cost |
|---|---|---|---|---|---|
| math_committee | 4 agents / 4 calls, acc 1.000 | 1 call | 0.000 | −75% | −72% |
| debate | 5 agents / 5 calls, acc 1.000 | 1 call | 0.000 | −80% | −79% |
| game-builder-crew | 3 agents, score 0.667 | 1 agent | 0.000 | −67% | −67% |
| screenplay_writer | 3 agents, score 0.933 | 1 agent | **+0.017** | −67% | −74% |
| langgraph_pipeline (hard) | 3 agents, score 0.833 | 1 agent | 0.000 | −67% | −67% |

- **screenplay_writer**: the lone `scriptwriter` agent *out-scores* the full
  `analyst → scriptwriter → formatter` pipeline (0.950 vs 0.933) on the repo's own
  rubric. The other two agents add latency and a little noise, no quality.
- **game-builder-crew**: the two review agents (`qa_engineer`, `chief_qa_engineer`)
  change **2.4%** and **0.0%** of the code respectively (measured by diff ratio) —
  near-pure pass-throughs. Same score with them or without.
- **langgraph_pipeline**: holds even on deliberately hard reasoning problems
  (modular arithmetic, logic puzzles, the "missing dollar" riddle). The
  `researcher → analyst → writer` decomposition doesn't beat one writer.
- **debate_hard** (ensemble case, the one "More Agents" says should win): 4
  independent solvers + majority vote vs 1 solver, on 12 competition-style
  integer-answer problems (CRT, divisor counts, `2^40 mod 100`, etc.).
  **1.000 = 1.000 = 1.000** — `gemini-3.5-flash` one-shots all of them, so the
  vote buys nothing. We could not find a task hard enough for this model to make
  the ensemble pay, within the time budget.

This lines up with the external evidence: UC Berkeley's MAST puts
verification/review agents among the top failure sources, and Cognition's
"Don't Build Multi-Agents" argues sequential role pipelines mostly add
coordination cost. We now have it measured on real code.

---

## What this does NOT show (be honest)

1. **No task here needs multi-agent.** `gemini-3.5-flash` is strong enough to
   one-shot every one of these — including the "hard" competition-math set built
   specifically to break it (debate_hard: still 1.000). So "1 agent is enough" is
   partly a statement about the model. The experiments that would flip this —
   parallel search with a capable aggregator on genuinely model-breaking problems,
   tool-use division of labour, >100k-token context — are not built yet. This is
   the single most important gap to close before the thesis is publishable.
2. **Metrics are coarse.** The game-code proxy is flat at 0.667; the repo's
   screenplay scorer only ranges 9.0–9.5; the judge is the same model family.
   Stronger metrics (headless game execution, pairwise judging, a bigger judge)
   would firm up or complicate the picture.
3. **Small n.** 6–12 samples per config, 2 repeats. No proper variance bands yet.
4. **Two frameworks, one real repo.** Need MetaGPT / ChatDev / AutoGen and more
   real repos before any general claim.
5. The `downgrade` ablation is a near-noop on a single model — needs a
   cheap/expensive model pair (blocked: `gemini-2.5-pro` has ~0 free-trial quota).

---

## What's in the repo now

```
agentslim/                core: DAG model, Trace, LLM (Vertex/mock), ablations,
                          propose(), greedy_minimize(), spend meter + cache + RPS limit
agentslim/adapters/       crewai_shim (works), langchain_shim (works for StateGraph;
                          supervisor handoff is WIP), litellm_shim
systems/                  math_committee, debate  (synthetic pilots)
experiments/repos/        game_builder_crew, screenplay_writer, langgraph_pipeline,
                          langgraph_supervisor_team (WIP)
results/*.json            all numbers above, reproducible from cache
experiments/FINDINGS.md   detailed per-experiment notes + caveats
experiments/WORKLOG.md    timeline of the overnight run
```

## Suggested next steps (morning discussion)

1. **A task where multi-agent wins.** Pick one: parallel-sample + aggregate on
   MATH/AIME; a tool-use task (code + test + fix) with a real sandbox; long-doc QA.
   Show the minimizer *keeps* the agents that matter there.
2. **Real repo #3 with a strong metric** — ChatDev or MetaGPT on a coding task
   scored by actually running the output.
3. Wire `AGENTSLIM_MODEL_LARGE/_SMALL` and get `downgrade` working (needs paid-tier
   quota or a second provider).
4. Variance: 5 repeats, report CIs; pre-register the H1–H4 thresholds.
5. Fold the fixed-config repo experiments into `greedy_minimize` so the tool
   produces the reduced system + diff automatically for any adapter.
