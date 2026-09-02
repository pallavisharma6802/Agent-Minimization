# Findings log

## 2026-09-01 — Pilot harness online (mock backend)

**What works:** DAG system model + per-call trace + ablation primitives
(remove / identity / heuristic / downgrade / merge) + `propose()` per-agent
classifier + `greedy_minimize()` end-to-end reducer. Two pilot systems run.

### math_committee (planner → solver → [checker] → formatter)
| metric | baseline | minimized |
|---|---|---|
| agents | 4 | 1 (checker) |
| LLM calls/task | 4.0 | 1.0 |
| cost | — | **-62%** |
| accuracy | 0.833 | 0.917 (+0.083) |

Recovered ground truth: planner and formatter are redundant; the solver's large
model is over-specified *because the oversight checker independently recomputes*
and masks solver errors. The minimizer collapsed the capability chain and left
the oversight agent doing the work — a real "the scaffolding wasn't load-bearing"
result.

### debate (4 parallel solvers → aggregator)
| metric | baseline | minimized |
|---|---|---|
| agents | 5 | 1 | 
| LLM calls/task | 5.0 | 1.0 |
| cost | — | **-99%** |
| accuracy | 0.800 | 0.900 (+0.100) |

Recovered "ensemble is over-provisioned on easy tasks" (the *More Agents Is All
You Need* regime only pays off when task difficulty is high).

## Known limitations (do not report these numbers as real)
1. **Mock backend has zero accuracy variance** → noise band collapses to `abs_tol`
   (0.02), so borderline agents look more removable than they would with a real
   model. Need OpenAI/Anthropic backend for credible deltas.
2. Mock large-vs-small models barely differ → `downgrade` verdicts are not
   trustworthy yet.
3. Provenance edge detection (`Trace.consumed_by`) uses a 60-char substring match;
   fine for pilots, will miss paraphrased hand-offs in real frameworks.
4. No real repos yet — next step is a LangGraph and a CrewAI adapter.

## Next
- [ ] Add a real LLM backend run (needs `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`)
- [ ] Third pilot: redundant critic chain (writer → c1 → c2 → c3) with LLM-judge scoring
- [ ] `collapse_all` ablation (the Cognition "just one agent" test) as an explicit move
- [ ] LangGraph adapter → lower a real `StateGraph` into `MultiAgentSystem`
- [ ] Risk-surface metrics (trust edges, injection reachability) pre/post minimize
