# Findings log

## 2026-09-02 — Real backend (Vertex / gemini-3.5-flash, `global`) + first real repos

Model: `gemini-3.5-flash` via Vertex AI, `location=global`, project `agent-min`
(GCP free-trial credits only). All runs cache to `.llm_cache/`; the cost meter
separates **live** (real API $) from **modeled** (what the traced calls would cost
uncached).

### Pilot 1 — math_committee (planner → solver → [checker] → formatter)
| metric | baseline | minimized |
|---|---|---|
| capability agents making an LLM call | 3 | 0 |
| LLM calls / task | 4.0 | 1.0 |
| modeled cost | — | **−72%** |
| accuracy (12 grade-school word problems) | 1.000 | 1.000 |

`greedy_minimize` turned `planner` and `solver` into no-LLM pass-throughs and
removed `formatter`; the only surviving call is the oversight `checker`, which
independently recomputes and is right on its own. Sweep cost ~$0.25 / 1608 calls.
**Caveat:** baseline is already 100% — an easy task where multi-agent was never
needed. The result is real but unsurprising; the sharp test needs harder tasks.

### Pilot 2 — debate (4 solvers → aggregator)
(running / see results/debate.json) — baseline acc 1.000; propose flags all 4
solvers `remove`. Expected: ensemble over-provisioned on easy tasks
("More Agents" only pays at high difficulty).

---

## Real repo #1 — crewAIInc/crewA-examples :: crews/game-builder-crew
3-agent **sequential** CrewAI pipeline, prompts verbatim from the repo's YAML:
`senior_engineer` (writes game code) → `qa_engineer` (review) → `chief_qa_engineer`
(final review). Metric: 3-part code proxy (compiles / has game loop+entrypoint /
mentions the spec's core nouns), 3 game specs × 2 repeats.

| config | score | calls/run | modeled $/run |
|---|---|---|---|
| baseline (3 agents) | 0.667 | 3.0 | $0.0253 |
| drop chief_qa (2) | 0.667 | 2.0 | $0.0160 |
| **engineer only (1)** | **0.667** | **1.0** | **$0.0076** |
| merge qa→chief (2) | 0.667 | 2.0 | $0.0160 |

**Downstream code churn** (fraction of the code the reviewer actually changed):
`chief_qa_engineer` = **0.000**, `qa_engineer` = **0.024**. The two review agents
are near-pure pass-throughs; dropping them is free on this metric (−67% calls/cost).
Real API spend for the whole experiment: **$0.39**.
Caveat: the code proxy is coarse (flat 0.667 across all configs; the `pacman`
specs fail the noun check regardless). A headless-execution metric would be
stronger. Still, the churn number is direct evidence.

## Real repo #2 — crewAIInc/crewA-examples :: crews/screenplay_writer
3-agent **sequential** transform pipeline: `analyst` (distill arguments) →
`scriptwriter` (dialogue screenplay) → `formatter` (apply template). Metric: the
repo's **own `scorer` agent** (1–10 rubric), 3 discussions × 2 repeats.

| config | score /1 | calls/run |
|---|---|---|
| baseline (analyst+scriptwriter+formatter) | 0.933 | 3.0 |
| drop formatter (2) | 0.917 | 2.0 |
| drop analyst (2) | 0.950 | 2.0 |
| **scriptwriter only (1)** | **0.950** | **1.0** |
| merge analyst→scriptwriter (2) | 0.950 | 2.0 |

The lone `scriptwriter` **scores higher** than the full pipeline (0.950 vs 0.933)
at 1/3 the calls. The extra two agents add no measurable quality and a little
noise (configs with them dip to 0.90 on some samples; configs without are a flat
0.95). Real API spend: **$0.036**.
Caveat: the repo's scorer is lenient/low-variance (9.0–9.5 band); a stronger or
pairwise judge would firm this up. n=6 per config.

---

## Cross-cutting read so far
Three real 3-agent pipelines (one synthetic, two from a ~7k-star repo), same
result each time: **~2 of every 3 agents carry no measurable quality**, and
removing them cuts calls and cost ~1:1 with agent count. Consistent with MAST
(review/verification agents are a top failure source) and Cognition's
"don't build multi-agent". What's still missing for a strong claim:
1. a **hard** task where multi-agent genuinely helps, to show the method keeps
   load-bearing agents (currently every baseline is near-ceiling);
2. stronger metrics (headless execution; pairwise LLM-judge);
3. a non-CrewAI framework (LangGraph adapter next);
4. more samples + variance bands.

## Known limitations (unchanged)
- `downgrade` ablation is a near-noop on a single-model backend (need
  `AGENTSLIM_MODEL_LARGE`/`_SMALL`; gemini-2.5-pro has ~0 free-trial quota).
- provenance edge detection is a 60-char substring match.
- CrewAI tool-calling turns are not instrumented (shim falls back to original).
