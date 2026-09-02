# Agent Minimization for Multi-Agent Systems — Research Survey & Brutal Critique

*Draft v0.1 — 2026-09-01. Purpose: decide whether "find open-source multi-agent projects and reduce agent count without degrading efficiency" is a real research contribution or a repackaging of existing work. Verdict-first, then evidence.*

---

## 0. TL;DR verdict (brutal version)

**The core intuition is correct and well-supported: most deployed multi-agent systems are over-provisioned, and reducing them is both a cost win and a safety win.**

**But the specific framing you proposed — "a tool that scans GitHub repos and prunes agents" — is, as stated, mostly already done, and the part that isn't done is hard for reasons that aren't about tooling.** Concretely:

1. **Communication/edge pruning is solved-ish.** AgentPrune (ICLR'25) already does one-shot pruning of the agent communication graph with 28–73% token reduction at equal accuracy. M³Prune, AGP (Adaptive Graph Pruning), and others extend it.
2. **Architecture/agent-count search is solved-ish.** MaAS (ICML'25 Oral, "agentic supernet"), AFlow (MCTS over workflows), ADAS, GPTSwarm all *automatically discover* smaller/cheaper agentic workflows. MaAS reports 6–45% of baseline inference cost at equal-or-better accuracy.
3. **The "just use one agent" position is a live, credible camp.** Cognition ("Don't Build Multi-Agents") argues fan-out is actively harmful for shared-state tasks; Anthropic's own multi-agent research system admits 10–15× token cost.
4. **The failure evidence is overwhelming.** UC Berkeley MAST: 7 frameworks, 1,600+ traces, 41–86.7% failure rates, ~37% of failures rooted in system/agent-design decisions (too many agents, unclear roles, bad handoffs).

**Where the gap actually is** (and where your project could live):
- Nobody has a **repo-level, static+dynamic analyzer** that ingests an *arbitrary existing* multi-agent codebase (LangGraph / CrewAI / AutoGen / OpenAI Swarm / bespoke) and outputs a *concrete refactor* ("these 3 agents are one agent with 3 tools; this critic never changes an output; this router is a regex"). Existing pruning work operates on *their own* benchmark harnesses and standardized graph abstractions, not on messy real codebases.
- Nobody frames **agent-count reduction as an AI-safety intervention** with measured reduction in attack surface / collusion capacity / cascading-failure paths. Cooperative AI has a $10M fund explicitly for this world; the framing is fundable but unclaimed at the "practical tooling" level.
- The empirical question **"how much of deployed multi-agent complexity is load-bearing?"** has not been answered on real-world (non-benchmark) systems.

So: **kill the "novel pruning algorithm" framing. Keep the "measurement study + refactoring assistant + safety framing" framing.** Details below.

---

## 1. Framing the research question

### 1.1 The vague version (what you said)
> "Find open-source multi-agent projects, reduce the number of agents without degrading efficiency."

Problems with this as a research question:
- "Efficiency" is undefined. Do you mean task accuracy, latency, token cost, dollar cost, or reliability? These trade off against each other. AgentPrune improves cost; it does not always improve accuracy.
- "Number of agents" is the wrong primitive. In LangGraph a "node" may or may not be an LLM call. In CrewAI an "agent" is a persona; the cost driver is *turns × context size*, not headcount. Reducing 5 agents to 3 can *increase* cost if the 3 now carry bloated context.
- "Without degrading" is unfalsifiable without a benchmark per project, and most OSS multi-agent repos ship **no eval harness at all**.

### 1.2 Sharper candidate research questions

Pick one as primary. My ranking:

**RQ1 (measurement — strongest, most defensible):**
*In real-world open-source multi-agent systems, what fraction of agents/edges/LLM-calls are "load-bearing" — i.e., removing or merging them measurably degrades a task metric — versus redundant?*
- Deliverable: a dataset of N (say 30–50) OSS multi-agent repos, each with a minimal eval, ablation results, and a taxonomy of "why this agent exists" (real specialization / context isolation / cargo-culting / demo theater / prompt-length workaround).
- This is a paper on its own. Nobody has done it on non-benchmark systems.

**RQ2 (method):**
*Can a static + dynamic analysis pipeline automatically identify safe agent-merge / agent-removal refactors in an arbitrary multi-agent codebase, and how does its accuracy compare to human expert judgment?*
- Deliverable: the tool + an evaluation against expert-labeled ground truth from RQ1.

**RQ3 (safety):**
*Does reducing agent count / inter-agent communication measurably shrink the security and systemic-risk surface (prompt-injection propagation paths, collusion capacity, cascading-failure reachability), and at what cost to capability?*
- Deliverable: a safety-metrics framework + before/after measurements on the RQ1 corpus. Aligns with Cooperative AI's multi-agent safety fund and BlueDot's AI-safety audience.

**Recommended scope for a first project:** RQ1 as the backbone, RQ2 as the artifact, RQ3 as the framing/impact section. That's one coherent story: *"We measured how bloated real multi-agent systems are, built a tool to fix it, and show the fix is also a safety win."*

### 1.3 Hypotheses to state up front (so you can be proven wrong)
- H1: >40% of agents in OSS multi-agent repos can be merged/removed with <2% task-metric change.
- H2: The dominant *real* reason for multiple agents is **context-window isolation**, not capability specialization. (If true, better context management dissolves most multi-agent designs — this is the Cognition thesis.)
- H3: Agent count correlates with failure rate (MAST data suggests this) more strongly than with task success.
- H4: Reducing agents reduces prompt-injection blast radius roughly linearly with the number of removed inter-agent trust edges.

---

## 2. Survey — what exists

### 2.1 Camp A: "Reduce the communication, keep the agents" (graph/edge pruning)

| Work | Venue | What it does | Reported result | Limitation for you |
|---|---|---|---|---|
| **AgentPrune** ("Cut the Crap") | ICLR 2025 | One-shot pruning of spatial-temporal message-passing graph via trainable low-rank mask; removes redundant/malicious messages | Equal accuracy at **$5.6 vs $43.7**; **28.1–72.8% token reduction**; also robustifies vs adversarial agents | Operates on *their* standardized graph abstraction over their benchmark suite (MMLU, HumanEval, GSM8K, etc.), not arbitrary repos. Prunes edges, not agents. |
| **AGP — Adaptive Graph Pruning** | 2506.02951 (2025) | Jointly optimizes hard-node pruning + soft-edge pruning, task-adaptive | Beats fixed topologies across benchmarks | Same: benchmark harness, not real codebases. |
| **M³Prune** | 2511.19969 (2025) | Hierarchical communication-graph pruning for multimodal multi-agent RAG | Efficiency gains on MM-RAG | Narrow domain. |
| **AgentDiet / trajectory reduction** | 2509.23586 (2025) | Removes useless/redundant/expired info *within* trajectories | **39.9–59.7% input-token cut, 21–36% total cost**, equal performance | Complementary, not competing — this is context hygiene, orthogonal to agent count. |

**Takeaway:** the "prune the graph" niche is crowded and has strong ICLR/ICML results. Do **not** pitch a new pruning algorithm.

### 2.2 Camp B: "Search for the right architecture automatically"

| Work | Venue | Mechanism | Result |
|---|---|---|---|
| **MaAS — Multi-agent Architecture Search via Agentic Supernet** | ICML 2025 **Oral** (~top 1%) | Learns a *distribution* over architectures; a controller samples a per-query subnetwork (query-difficulty-aware resource allocation) | **6–45% of baseline inference cost**, +0.54–11.82% accuracy, transfers across datasets |
| **AFlow** | ICLR 2025 | MCTS over code-represented workflow graphs; nodes/edges as code | Beats ADAS and hand-designed; near-frontier accuracy at fraction of cost |
| **ADAS** | 2024 | LLM "meta agent" writes new agent-system code in a loop, linear archive | Foundational but weak search (linear heuristic) |
| **GPTSwarm** | ICML 2024 | Agents as graph; RL + graph optimization on edges | Struggles with conditional/branching workflows |
| **EvoMAC / EvoFlow / A2Flow / FlowBank** | 2025–26 | Evolutionary or precompute-and-reuse variants | Incremental |

**Takeaway:** "automatically find a cheaper multi-agent design" is an established subfield with a canonical benchmark set (GSM8K, MATH, HumanEval, MBPP, MMLU, DROP). The **open flank is that all of them start from scratch or from a DSL** — none *refactor an existing production codebase in its original framework*.

### 2.3 Camp C: "Don't build multi-agent at all"

- **Cognition, "Don't Build Multi-Agents"** (Walden Yan, 2025): for shared-state tasks (coding), parallel subagents each act on a partial view, make conflicting implicit decisions, and you get incoherent output. Principles: (1) share full context/trace, (2) actions carry implicit decisions — avoid conflicting ones. Implication: single agent + good context management beats fan-out.
- **Anthropic, "How we built our multi-agent research system"** (2025): multi-agent *does* win for open-ended parallel research/breadth-first search — but costs **~15× the tokens of a chat**, and only pays off when task value is high.
- **LangChain, "How and when to build multi-agent systems"** (2025): walks back some earlier enthusiasm; "context engineering" over agent proliferation.
- **Walkthrough consensus (mid-2026):** single agent wins for sequential tasks under ~20K tokens; multi-agent only for genuinely independent parallel sub-problems.

**Takeaway:** there is a credible, industry-backed thesis that **your project's premise is right and the fix is often "collapse to one agent."** Use this as support, not competition. It also tells you the *mechanism* to look for: multiple agents are frequently a **workaround for context-window / attention limits**, not a real decomposition.

### 2.4 Camp D: "Multi-agent as a safety problem"

- **"Multi-Agent Risks from Advanced AI"** (Cooperative AI Foundation, 2025): new failure modes absent in single-agent — collusion, conflict, destabilizing dynamics, emergent agency, multi-agent security holes. Drivers: interaction topology, cognitive opacity, objective divergence.
- **Cooperative AI — $10M fund** "Scaling AI Safety for a Multi-Agent World" + a separate multi-agent-safety grants program. Explicitly wants work here.
- **Collusion literature:** "Institutional AI: Governing LLM Collusion in Multi-Agent Cournot Markets"; work showing colluding agents exceed isolated-agent capability via role distribution + info sharing.
- **UC Berkeley MAST (Multi-Agent System Failure Taxonomy)**, "Why Do Multi-Agent LLM Systems Fail?" (2025): 7 frameworks (AppWorld 86.7% fail, ChatDev 75%, HyperAgent 74.7%), 1,642 traces, 14 failure modes in 3 buckets — **Specification & System Design 37%, Inter-Agent Misalignment 31%, Verification & Termination 31%**. ~79% of failures involve agents misunderstanding each other / context lost in handoffs.

**Takeaway:** the safety framing is *strong and underserved at the practical-tooling level*. "Every agent you delete is one fewer trust boundary, one fewer injection vector, one fewer node in a cascade" is a clean, quantifiable thesis. Nobody owns "agent minimization as attack-surface reduction."

### 2.5 Adjacent: classical MAS minimization
Old multi-agent-systems / MARL literature has "team formation," "coalition structure generation," and "minimal agent teams," plus organizational-design theory (when does hierarchy help). Worth a paragraph for intellectual honesty, but the LLM setting differs because agents are stochastic, expensive, and share a substrate model.

---

## 3. What the field lacks (the actual gaps)

1. **No real-world corpus.** Every efficiency/pruning result is on ~6 academic benchmarks. We do not know how bloated *actual* deployed systems (the top LangGraph/CrewAI/AutoGen repos, YC companies' agent stacks) are. **This is measurable and unclaimed.**

2. **No "framework-native refactoring" tool.** MaAS/AFlow/ADAS emit systems in their own representation. A practitioner with a 2,000-line CrewAI codebase gets nothing actionable. The gap is a tool that:
   - parses the actual framework graph (LangGraph `StateGraph`, CrewAI `Crew`, AutoGen `GroupChat`, Swarm handoffs, or a bespoke orchestrator),
   - instruments it to log every LLM call, its input context, output, and downstream causal effect,
   - runs ablations (remove agent / merge two agents / replace critic with pass-through / replace router with heuristic),
   - reports concrete merge candidates with measured metric deltas and a diff.

3. **No agreed "load-bearing" definition or metric.** Need something like *causal contribution*: does this agent's output change the final answer on >X% of inputs? Does removing it change a metric beyond noise? This is a methodological contribution.

4. **No safety accounting.** No metric set for "systemic risk surface of a multi-agent config": count of inter-agent trust edges, injection-propagation reachability, privilege union per agent, collusion capacity (can any 2 agents coordinate to defeat a guardrail that binds each). Reducing agents should move these numbers; nobody measures it.

5. **Benchmark-Goodharting risk unaddressed.** "Reduce agents without degrading efficiency" on GSM8K is trivial and meaningless — most benchmark tasks never needed multi-agent. The hard cases are open-ended, long-horizon, tool-heavy tasks where evals barely exist.

---

## 4. Brutal critique of your idea as originally stated

**What's wrong / risky:**

1. **"Scan any GitHub repo" is a research-scope trap.** Multi-agent repos have near-zero API uniformity. You'd spend 80% of effort on parser plumbing for LangGraph vs CrewAI vs bespoke, and 20% on the actual science. Pick **one or two frameworks** (LangGraph + CrewAI cover most of the OSS long tail) and say so.

2. **No evals = no claim.** Most OSS multi-agent repos are demos with no test suite. "Without degrading efficiency" requires you to *build an eval per repo*, which is the real cost. Budget for it or the project is unfalsifiable.

3. **"Reduce agent count" optimizes a vanity metric.** The thing that costs money and creates risk is **LLM calls, context tokens, and trust edges**, which are correlated with but not equal to agent count. Reframe around those.

4. **The algorithmic contribution is likely already published.** If you pitch "a pruning method," reviewers cite AgentPrune/MaAS/AFlow and desk-reject. You must pitch measurement + tooling + safety, or a *genuinely* new method primitive (e.g., causal-mediation-based agent ablation — possibly novel, needs a lit check).

5. **"AI safety" needs to be earned, not asserted.** Fewer agents ≠ safer by fiat. You need the metric framework in §3.4 and actual before/after numbers, or BlueDot / safety reviewers will read it as cost-optimization with a safety sticker.

6. **Efficiency might already be near-optimal in good repos, and terrible only in bad ones.** Plausible outcome: the well-engineered systems (Anthropic-style) are lean, and the bloated ones are abandoned demos nobody runs. Then your corpus is "we optimized code nobody uses." Mitigate by weighting toward **actively-maintained, actually-deployed** systems (stars + recent commits + used-in-production signals).

7. **Single-model substrate confound.** If all agents are GPT-4o-mini, merging them is easy. If they're heterogeneous (a cheap router + expensive worker + a fine-tuned critic), merging destroys the cost structure. Your method must detect and respect heterogeneity.

**What's right / defensible:**
- The premise that deployed multi-agent systems are over-built is supported by MAST, Cognition, and every cost analysis.
- "Measure it on real systems" is a genuine, unclaimed contribution.
- The safety framing is fundable (Cooperative AI) and audience-appropriate (BlueDot).
- A refactoring assistant that outputs concrete diffs is a real product with a real user (every team running agents in prod and watching the bill).

**Honest probability assessment:**
- As "new pruning algorithm": ~10% chance of being novel/publishable. Don't.
- As "measurement study + open dataset + refactoring tool + safety framing": ~70% chance of a solid workshop/short paper, ~35% of a strong main-conference paper if the corpus is big and the tool works.
- As a YC pitch ("AI agent cost/safety audit"): viable *if* you can show a demo that saves a real company real money on their real stack. Crowded-ish (LLM observability players — LangSmith, Langfuse, AgentOps, Braintrust — will add this), so speed and a wedge matter.

---

## 5. Proposed reframed project

**Title (working):** *How Much of Your Multi-Agent System Is Load-Bearing? A Measurement Study and Refactoring Assistant.*

**Three artifacts:**
1. **MA-Bloat corpus:** 30–50 actively-maintained OSS multi-agent systems (LangGraph + CrewAI first), each with (a) a minimal task eval, (b) full call-graph instrumentation, (c) ablation results, (d) hand-labeled "reason each agent exists."
2. **`agentslim` tool:** instruments a target system, runs merge/remove/downgrade ablations, emits ranked refactor suggestions with measured metric deltas + a code diff. Start semi-automated (human confirms each suggestion).
3. **Safety-surface report:** for each system, before/after on: # LLM calls, total context tokens, $/task, # inter-agent trust edges, injection-propagation reachability, privilege union, collusion capacity. Show the Pareto frontier of capability vs risk-surface.

**Key methodological primitive to develop:** *causal agent ablation* — measure each agent's causal contribution to the final metric via intervention (remove / replace with identity / replace with cheap heuristic), not just correlation. Check whether "causal mediation analysis for agent necessity" is genuinely unpublished (quick lit search; I didn't find it).

---

## 6. Test-on-open-source plan (concrete, phased)

**Phase 0 — scoping (1–2 weeks)**
- Lock scope to LangGraph + CrewAI (+ maybe AutoGen). Justify with GitHub prevalence.
- Build the repo selection criteria: ≥N stars, commit in last 90 days, has a runnable entrypoint, uses ≥3 agents/nodes with LLM calls, task has a checkable output.
- Candidate sources: `langgraph` examples + community repos, CrewAI `crewAI-examples`, `awesome-llm-agents` lists, GAIA/AppWorld/AssistantBench-style task repos, ChatDev, MetaGPT, AgentVerse, AutoGen `GroupChat` samples.

**Phase 1 — instrumentation harness (3–4 weeks)**
- Write framework adapters: intercept every LLM call (monkeypatch the client / use framework callbacks — LangChain callbacks, CrewAI step callbacks), log {agent id, role prompt, input context, tools available, output, wall-clock, tokens, cost}.
- Build the causal graph: which agent outputs feed which agent inputs; which reach the final answer.
- Static pass: detect degenerate agents (router that's a switch statement; critic whose output is never conditioned on; agent with one tool and no reasoning = a function call).

**Phase 2 — minimal evals (4–6 weeks, the expensive part)**
- For each repo, define 20–50 task instances with programmatic scoring (exact match, unit tests, LLM-judge with rubric + human spot-check).
- Establish baseline metric + noise band (run baseline 5×; multi-agent systems are high-variance).

**Phase 3 — ablations**
- For each agent: (a) remove (route its input straight to consumer), (b) merge with an adjacent agent (concatenate role prompts, union tools), (c) downgrade model, (d) replace with heuristic.
- Measure metric delta vs noise band, plus cost/latency delta.
- Classify: redundant / mergeable / load-bearing / context-isolation-only.

**Phase 4 — the tool**
- Package Phases 1+3 into `agentslim <path>` → report + suggested diffs.
- Validate: does the tool's ranking match your hand analysis from Phase 3? Report precision/recall of "safe to remove."

**Phase 5 — safety surface**
- Implement the risk-surface metrics. Recompute pre/post refactor.
- Run a prompt-injection propagation test: inject into one agent's tool output, see how far it spreads pre vs post.

**Deliverables:** the corpus (public), the tool (public), a paper, a blog post with the money numbers ("median OSS multi-agent system: X% of LLM calls removable, $Y/1k tasks saved, Z fewer injection paths").

**Rough timeline:** ~4–5 months for a serious first version solo; 2–3 months with help on evals.

---

## 7. Pitch framing — YC vs BlueDot

### 7.1 YC (product/company)
- **One-liner:** "We audit your AI agent stack and cut the LLM bill 30–60% without losing accuracy — and show you the security holes you closed doing it."
- **Wedge:** teams running multi-agent in prod, bill is spiking, nobody knows which agents matter. You plug into their traces (LangSmith/Langfuse/OTel), run shadow ablations, deliver a refactor PR.
- **Why now:** agent frameworks exploded in 2024–25, bills are landing in 2026, MAST-style unreliability is public, budget scrutiny is real.
- **Moat concern (be honest in the pitch):** observability incumbents (LangSmith, Langfuse, Braintrust, AgentOps) can bolt this on. Your edge = the causal-ablation method + the safety report + being the specialist. Land-grab speed matters.
- **Business model:** audit engagement → continuous "agent efficiency + safety" monitoring SaaS.
- **Risk to flag:** market may prefer "make agents work" over "make agents cheaper" until reliability is solved. Consider leading with reliability (fewer agents = fewer MAST failure modes) and cost as the ROI proof.

### 7.2 BlueDot Impact / AI-safety framing
- **Thesis:** multi-agent proliferation expands systemic risk surface (collusion capacity, cascading failure, injection propagation, emergent coordination — per Cooperative AI's "Multi-Agent Risks from Advanced AI"). **Minimization is a concrete, deployable safety intervention** — the multi-agent analogue of least-privilege / attack-surface reduction.
- **Contribution:** a metric framework for "multi-agent risk surface" + empirical evidence that most deployed multi-agent complexity is not load-bearing, so the safety cost is largely free.
- **Funding fit:** Cooperative AI's multi-agent safety grants + $10M "Scaling AI Safety for a Multi-Agent World" fund explicitly target this.
- **What makes it not just cost-cutting:** you must (a) define and measure the risk metrics, (b) show cases where a removed agent was a real vulnerability (injection relay, privilege escalation path, unmonitored autonomous loop), (c) discuss where minimization *hurts* safety (removing a dedicated safety monitor/critic is the wrong cut — your method must protect those).
- **Nuance to preempt:** "isn't a monitor agent a good multi-agent pattern?" Yes. Distinguish **capability agents** (minimize) from **oversight agents** (keep/strengthen). This distinction is itself a contribution.

---

## 8. Immediate next steps

1. **Lit-check the one possibly-novel primitive:** search for "causal mediation / intervention analysis for agent necessity in LLM multi-agent systems." If truly absent, that's your method contribution.
2. **Pick 5 pilot repos** across LangGraph + CrewAI and hand-instrument them this week — get real numbers on 1–2 before scaling. Ground truth beats more surveying.
3. **Draft the risk-surface metric list** and get one safety researcher (BlueDot network) to sanity-check it.
4. **Decide primary venue:** measurement paper (arXiv + agents workshop) vs YC vs safety grant. They're compatible but the emphasis order differs; pick the lead.
5. Write the H1–H4 hypotheses into a pre-registration doc so results are credible.

---

## Sources

- [Cut the Crap / AgentPrune (ICLR 2025)](https://arxiv.org/abs/2410.02506) · [ICLR page](https://iclr.cc/virtual/2025/poster/29978)
- [Adaptive Graph Pruning for Multi-Agent Communication](https://arxiv.org/pdf/2506.02951)
- [M³Prune: Hierarchical Communication Graph Pruning](https://arxiv.org/pdf/2511.19969)
- [Reducing Cost of LLM Agents with Trajectory Reduction (AgentDiet)](https://arxiv.org/abs/2509.23586)
- [Multi-agent Architecture Search via Agentic Supernet (MaAS, ICML 2025 Oral)](https://arxiv.org/abs/2502.04180) · [code](https://github.com/bingreeky/MaAS)
- [AFlow: Automating Agentic Workflow Generation (ICLR 2025)](https://arxiv.org/abs/2410.10762)
- [More Agents Is All You Need](https://arxiv.org/abs/2402.05120)
- [Why Do Multi-Agent LLM Systems Fail? / MAST — UC Berkeley](https://futureagi.substack.com/p/why-do-multi-agent-llm-systems-fail) · [analysis](https://www.hakunamatatatech.com/our-resources/blog/why-do-multi-agent-llm-systems-fail)
- [Cognition — Don't Build Multi-Agents / single vs multi debate](https://www.philschmid.de/single-vs-multi-agents) · [Anthropic multi-agent research architecture](https://theaiengineer.substack.com/p/how-anthropic-built-multi-agent-deep)
- [LangChain — How and when to build multi-agent systems](https://www.langchain.com/blog/how-and-when-to-build-multi-agent-systems)
- [Multi-Agent Risks from Advanced AI — Cooperative AI Foundation](https://www.cs.toronto.edu/~nisarg/papers/Multi-Agent-Risks-from-Advanced-AI.pdf)
- [Cooperative AI — Scaling AI Safety for a Multi-Agent World ($10M)](https://www.cooperativeai.com/calls-for-proposals/scaling-ai-safety-for-a-multi-agent-world) · [multi-agent safety grants](https://www.cooperativeai.com/grants/multi-agent-safety)
- [Multi-Agent AI Risks: Mapping the Emerging Coordination Challenge](https://medium.com/@gema-parreno-piqueras/multi-agent-ai-risks-mapping-the-emerging-coordination-challenge-42da86d2d173)
- [Institutional AI: Governing LLM Collusion in Multi-Agent Cournot Markets](https://arxiv.org/pdf/2601.11369)
- [Redis — Why Multi-Agent LLM Systems Fail & How to Fix Them](https://redis.io/blog/why-multi-agent-llm-systems-fail/)
