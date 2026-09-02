# Why teams build multi-agent systems, when it helps, and how to right-size them

*Literature synthesis for the agent-minimization project. Updated 2026-09-02.*
*Companion to [SURVEY.md](SURVEY.md) (methods) and [RESULTS.md](RESULTS.md) (our experiments).*

---

## 1. Why people reach for multi-agent (the real reasons)

Not all of these are about capability. Sorted by how well they hold up.

| # | Reason | Holds up? | Note |
|---|--------|-----------|------|
| 1 | **Context-window / attention limits** — one agent with 15 tools and a huge prompt degrades; split the work so each agent has a focused context | **Strong** | Agent quality drops once the toolkit passes ~8–12 tools (context-window overload + "cognitive interference"). This is the single most common *legitimate* driver. |
| 2 | **Genuinely parallel, independent sub-tasks** — breadth-first search, "research these 10 companies" | **Strong** | Throughput win. Anthropic's research system is this; it costs ~15× the tokens and only pays when task value is high. |
| 3 | **Distinct / adversarial expert roles** — generator vs critic, red-team vs blue-team | **Situational** | Real gains on *hard* tasks; on easy tasks the critic has nothing to catch (we saw exactly this — [RESULTS.md](RESULTS.md)). |
| 4 | **Modularity / separation of concerns** — test, upgrade, fine-tune, roll back one agent without touching the rest | **Strong, but it's an *engineering* benefit, not a capability one** | You can keep this benefit while collapsing capability-redundant agents. A 3-agent system is still modular. |
| 5 | **Failure isolation** — a bad agent's blast radius is one module | **Weak in practice** | The opposite is usually observed: errors *propagate* across agents (see §3). |
| 6 | **"It's how humans organize"** — mirror a software company / newsroom | **Weak** | Anthropomorphic, not evidence-based. ChatDev's 7-role company is the archetype and MAST clocks it at ~75% failure. |
| 7 | **Cargo-culting / demo aesthetics** — the framework tutorial had 5 agents | **Not a reason** | A large fraction of the OSS long tail. |

**Takeaway for the project:** reasons 1, 2, 4 are the ones a minimizer must respect.
Reason 1 dissolves with better context management. Reason 2 means "keep the parallel
fan-out." Reason 4 means "don't merge two agents that are independently owned/tested
even if you technically could."

---

## 2. When multi-agent actually beats a strong single agent

The most rigorous datapoint is **Google Research, "Towards a Science of Scaling Agent
Systems"** (arXiv:2512.08296) — 180–260 configurations, 6 benchmarks, 5 canonical
architectures (Single, Independent, Centralized, Decentralized, Hybrid), 3 model
families:

- **Structured / parallelizable tasks** (financial reasoning, tool workbenches):
  centralized multi-agent **+20–81%** over single-agent. Real, large gains.
- **Open-ended tasks** (PlanCraft): uncoordinated multi-agent **−35%**. Coordination
  overhead exceeds any benefit; a single agent's coherent reasoning wins.
- **Model-family-dependent**: OpenAI/Google models scale cooperatively; Anthropic
  models often do better with *less* coordination.

**"Single-Agent LLMs Outperform Multi-Agent Systems on Multi-Hop Reasoning Under
Equal Thinking-Token Budgets"** (Tran & Kiela, Stanford, arXiv:2604.02460):

- Across Qwen3, DeepSeek-R1-Distill, Gemini 2.5, **single agents match or exceed
  multi-agent once compute is held equal**.
- Information-theoretic argument (Data Processing Inequality): under a fixed
  reasoning budget with perfect context use, a single agent is strictly more
  information-efficient. Multi-agent only competes when the single agent's *effective
  context utilization degrades* — i.e. reason #1 above.
- **Implication:** many published multi-agent "wins" are **uncounted compute** —
  N agents = N× the tokens, and the baseline wasn't given the same budget.

**Agent-scaling / debate literature** (e.g. "Understanding Agent Scaling via
Diversity" 2602.03794; "More Agents Is All You Need" 2402.05120):

- Performance vs agent count shows **strong diminishing returns**. The big jump is
  **1 → 2**; **3–5 agents is the practical ceiling** for most tasks.
- **Homogeneous scaling** (clones) plateaus fast. **Heterogeneous** scaling
  (different models / prompts / tools) keeps paying — diversity, not headcount, is
  the lever.
- Hard, decomposable tasks (GSM-Plus, competition math) keep improving to ~7 agents
  *for large models*; easy tasks saturate at 1–2.

---

## 3. Why the extra agents so often *don't* help (or hurt)

1. **Compute confound** — the apparent gain is just more tokens (Tran & Kiela).
2. **Error compounding in sequential pipelines** — "errors propagate like rumors."
   Google Research: independent agents amplify errors **17.2×** vs **4.4×** with
   centralized coordination. A faulty planner's mistake is *irrecoverable* downstream.
3. **Lost-in-handoff** — UC Berkeley MAST: ~79% of multi-agent failures involve an
   agent misunderstanding another's output; ~37% trace to system/agent-design
   choices (too many agents, unclear roles).
4. **Coordination cost scales with headcount, not difficulty** — every inter-agent
   message is an LLM call. A 6-agent pipeline on an easy task pays 6× for nothing.
5. **Reviewer/critic agents frequently don't catch bugs** — our ChatDev runs: the
   Code Reviewer changed **0%** of the code; with a weak coder the review/test
   phases produced **byte-identical** output and the full pipeline scored *lower*
   than the 3-agent core. (Contrast: a *well-placed* verification agent right after
   each primary agent caught **96.4%** of injected errors in one study — placement
   and design matter enormously.)

---

## 4. How to right-size a multi-agent system (the levers)

Ordered roughly by how much they move the needle.

1. **Give the single agent better tools first.** A lone agent wired to a code
   interpreter + search + DB beats a 4-agent prompt-only pipeline. Only fan out
   when one agent genuinely can't hold the context.
2. **Control for compute when you compare.** Benchmark the single-agent baseline at
   the *same total token budget* as the multi-agent system. Most "multi-agent wins"
   evaporate here.
3. **Prefer heterogeneity over headcount.** 2 different models/prompts > 4 clones.
4. **Ablate for causal contribution.** For each agent: remove / replace-with-identity
   / replace-with-heuristic / merge-with-neighbour, and measure the task-metric
   delta against a noise band. Keep only agents that move it. *(This project's method.)*
5. **Right-size to the knee, not the minimum.** Sweep every team size, plot
   quality vs cost, take the smallest team still at peak quality. Often that's 3,
   sometimes 1, occasionally 5 — but almost never the shipped number.
6. **Separate agent *types*:**
   - **Capability agents** (do the work) → minimize aggressively.
   - **Oversight agents** (monitors, safety critics, verifiers) → keep/strengthen,
     but *place them well* (right after each primary, not as a final rubber-stamp).
   - **Parallel workers** (independent fan-out) → keep the fan-out; the count is set
     by the task's natural parallelism, not by a topology search.
7. **Make it difficulty-adaptive.** MaAS-style: route easy queries to a 1-agent
   path, spend the full team only on hard ones. 6–45% of the static cost.
8. **Collapse sequential role pipelines.** researcher→analyst→writer chains are the
   highest-risk, lowest-benefit pattern (error compounding, no parallelism). If the
   roles aren't independently owned, make them one agent with a structured prompt.

---

## 5. Where this project sits

The literature says: *multi-agent helps on structured/parallel/hard tasks with
centralized coordination and heterogeneous agents; it hurts on sequential,
open-ended, or easy tasks; and most published wins are compute artifacts.*

Nobody has:
- **measured the load-bearing fraction on real deployed OSS systems** (all the
  scaling papers use synthetic benchmark harnesses);
- shipped a **tool that ablates an arbitrary repo in its own framework** and emits
  the right-sized system + a diff;
- connected right-sizing to the **safety surface** (fewer trust edges, smaller
  injection blast radius, less collusion capacity).

That's the gap. Our ChatDev result (6→3 agents, quality flat or *up*, −77% cost, on
a 26k-star repo with a real execution metric) is the first brick.

---

## Sources

- [Towards a Science of Scaling Agent Systems (Google Research, arXiv:2512.08296)](https://arxiv.org/abs/2512.08296) · [blog](https://research.google/blog/towards-a-science-of-scaling-agent-systems-when-and-why-agent-systems-work/)
- [Single-Agent LLMs Outperform Multi-Agent Systems Under Equal Thinking-Token Budgets (Tran & Kiela, arXiv:2604.02460)](https://arxiv.org/html/2604.02460v1)
- [Understanding Agent Scaling in LLM-Based Multi-Agent Systems via Diversity (arXiv:2602.03794)](https://www.alphaxiv.org/abs/2602.03794)
- [Scaling Behavior of Single LLM-Driven Multi-Agent Systems (arXiv:2606.00655)](https://arxiv.org/html/2606.00655v1)
- [More Agents Is All You Need (arXiv:2402.05120)](https://arxiv.org/abs/2402.05120)
- [Why Do Multi-Agent LLM Systems Fail? / MAST — UC Berkeley](https://futureagi.substack.com/p/why-do-multi-agent-llm-systems-fail)
- [Why Your Multi-Agent System is Failing: the 17x Error Trap of the "Bag of Agents" (Towards Data Science)](https://towardsdatascience.com/why-your-multi-agent-system-is-failing-escaping-the-17x-error-trap-of-the-bag-of-agents/)
- [The Compounding Errors Problem (Zartis)](https://www.zartis.com/the-compounding-errors-problem-why-multi-agent-systems-fail-and-the-architecture-that-fixes-it/)
- [On the Resilience of LLM-Based Multi-Agent Collaboration with Faulty Agents (arXiv:2408.00989)](https://arxiv.org/pdf/2408.00989)
- [Single vs Multi-Agent — philschmid](https://www.philschmid.de/single-vs-multi-agents) · [LangChain: how and when to build multi-agent systems](https://www.langchain.com/blog/how-and-when-to-build-multi-agent-systems)
- [Multi-agent LLM applications review — Victor Dibia](https://newsletter.victordibia.com/p/multi-agent-llm-applications-a-review)
