# Why the answer is almost never "1" — and the reframe the project needs

*Written 2026-09-02 as a course-correction and a handoff to the next project.*
*Read this before touching `pareto_scan` or reading `RESULTS.md`.*

---

## 0. The mistake we made

Every experiment in [RESULTS.md](RESULTS.md) collapsed the system to **1 agent** (or,
for ChatDev, to 3 — its structural floor). That looks like a strong result. It is
actually a **measurement artifact**, and taking it at face value would make the
project wrong.

Here is the artifact:

> `pareto_scan` removes an agent, re-runs the task, and keeps the removal if the
> **task-accuracy score** doesn't drop. Task accuracy is the **only** signal it
> uses. So it can only ever discover one thing: *"decomposing this task across
> agents did not improve the answer."*

For the tasks we used — grade-school math, a Roman-numeral converter, an
8-line screenplay, "7^100 mod 13" — that is true and unsurprising. A single
modern LLM call fits the whole task in one context window and solves it. The
literature agrees (Tran & Kiela: single agents match multi-agent at equal
compute; [RESEARCH.md](RESEARCH.md) §2).

But **companies do not build multi-agent systems to improve task accuracy.** They
build them for reasons our metric is structurally blind to. If you only measure
accuracy, you will always conclude "you don't need the agents," and you will
always be wrong about real systems.

---

## 1. Why companies actually run multi-agent (nine reasons, none of them accuracy)

### 1. Context capacity — the task doesn't fit in one window
A 300-page contract review, triaging 10,000 support tickets, a market report
synthesised from 60 sources, a codebase-wide refactor. One agent physically
cannot hold the inputs. You split the work so each agent owns a slice and an
orchestrator composes the results. This is the single most common legitimate
driver, and it is a hard limit, not a preference. Our tasks were ~200 tokens —
they could never surface this.
*("The New Conway's Law: context windows shape enterprise architecture.")*

### 2. Least privilege / blast-radius containment (security)
The published industry rule (AWS, Microsoft, Zscaler):
> **"Agents that retrieve data must never be the same agents that write to
> production systems."**

A single agent holding database read, email send, payment execution, and deploy
rights is a catastrophe waiting for one prompt injection. Enterprises split
agents precisely so each one carries the **minimum, scoped, often ephemeral**
permission set. The split *is* the security control. Merging two agents that
have different permission scopes is not an optimization — it's a vulnerability.

### 3. Parallelism / throughput / latency
10,000 tickets cannot go sequentially through one agent within an SLA. You fan
out for wall-clock time. The agent count here is set by the task's natural
parallelism, not by any topology search. Our single-shot tasks have no
parallelism to exploit.

### 4. Conway's Law — organizational ownership
A bank's fraud team and its customer-service team **will never ship one agent.**
Different teams own them, deploy on different cadences, pick different models,
carry different SLAs and on-call rotations, answer to different compliance
owners. "Merge these two agents" is really "merge these two departments." A
centralized agent registry exists in mature orgs precisely to track per-agent
ownership, approved models, authorized tools, and risk tier.

### 5. Heterogeneous models & cost tiering
A cheap classifier routes → an expensive reasoner does the hard step → a
fine-tuned domain model handles the specialist bit → a small guardrail model
screens the output. Collapsing to "one agent" forces one model for all of it:
worse on the hard parts, wasteful on the easy parts. Diversity across agents is
a feature, not redundancy ([RESEARCH.md](RESEARCH.md) §2: heterogeneous scaling
keeps paying where homogeneous scaling plateaus).

### 6. Governance, auditability, compliance
Regulated decisions (credit, insurance, hiring, medical) must be **attributable
and individually gated**: "the underwriting agent decided X on these inputs; the
fairness agent checked Y; a human approved Z." A monolith's reasoning is one
opaque blob you cannot audit or gate mid-stream. "Governance enforced
architecturally consistently outperforms governance enforced through policy" —
so the boundaries are deliberately in the architecture.

### 7. Reliability & monoculture risk
Monolith = single point of failure. And running the *same* base model for every
step **correlates failures** — when it's wrong, it's wrong everywhere at once.
Sometimes different models per agent is a deliberate hedge, not waste.

### 8. Human-in-the-loop checkpoints
"A manager approves before the email sends." "Legal signs off before publish."
That gate is naturally its own node with its own state and its own interface.

### 9. Data / tenancy boundaries
An agent that may see raw customer PII vs one that may only see aggregates. An
agent scoped to tenant A's data vs tenant B's. Data governance forces the split
regardless of what a task-accuracy benchmark says.

---

## 2. What our benchmarks measured, and why "1" fell out

| axis | does multi-agent help? | did our benchmark test it? |
|---|---|---|
| task accuracy on a context-sized problem | usually **no** (matches literature) | **yes — this is all we tested** |
| context capacity | often the whole reason | no (tiny tasks) |
| permission / blast-radius scoping | yes, hard requirement | no (no tools, no permissions) |
| parallel throughput / latency | yes | no (single-shot) |
| org ownership / independent deploy | yes | no (we authored one system) |
| model heterogeneity / cost tiering | yes | no (one model everywhere) |
| auditability / compliance gating | yes | no |
| reliability / graceful degradation | yes | no |
| human-in-the-loop gates | yes | no |

We measured the one axis where multi-agent is *weakest*, on tasks chosen so a
single call would suffice, and then the tool reported "you only need one agent."
Of course it did. **That finding does not generalize to systems that exist for
reasons 1–9.**

---

## 3. The correct reframe

> **The tool must not ask "can this collapse to 1 agent?" It must ask: "which
> agents exist for no structural reason *and* also don't affect the outcome?"**

An agent is a candidate for removal/merge **only if it is redundant on every
axis**:

- [ ] it does **not** expand context capacity — its inputs comfortably fit in a
      sibling agent's window
- [ ] it has **no distinct tool set** — same tools as the agent it would merge into
- [ ] it has **no distinct permission / data scope** — merging would not widen any
      agent's access
- [ ] it is **not a parallel worker** — removing it does not reduce achievable
      throughput
- [ ] it is **not an oversight / compliance / HITL gate**
- [ ] it is **not independently owned / deployed** by a different team
- [ ] it uses the **same model** as the agent it would merge into (or a strictly
      weaker one)

If an agent fails **any** box, it is load-bearing for a reason the accuracy
metric cannot see — leave it. Only agents that pass **all** boxes go into the
ablation sweep, and only *those* can be reported as "removable" — and even then,
only with the ablation evidence that the outcome is unchanged.

What survives this filter is the real target: **pure sequential
capability-decomposition with no structural justification** — the
`researcher → analyst → writer` chains, the rubber-stamp reviewers, the
"planner" that just rephrases the task. Those are the cargo-cult agents. The
honest pitch is *"here are the N agents that exist for no reason, with proof
they don't change the result"* — not *"you have too many agents."*

---

## 4. Agent-boundary taxonomy (what makes an agent un-mergeable)

The next project's data model should carry, per agent:

| field | why it matters | how an adapter fills it |
|---|---|---|
| `tools: set[str]` | distinct tools ⇒ distinct capability ⇒ don't merge | from the framework's tool bindings |
| `permissions: set[str]` / `data_scope` | distinct scope ⇒ security boundary ⇒ never merge | IAM role, scopes, allow-lists |
| `owner: str` | distinct owner ⇒ org boundary ⇒ don't merge | CODEOWNERS, agent registry, repo path |
| `model: str` | distinct model ⇒ heterogeneity is intentional | framework config |
| `role_type` | `capability` \| `oversight` \| `gate` \| `parallel_worker` \| `router` | prompt/role heuristics + registry |
| `context_budget` | how much its inputs would add to a merged agent's window | measured from the trace |
| `parallel_group` | agents in the same group are fan-out workers; keep the fan-out | topology |

`merge(a, b)` must **refuse** unless
`a.tools == b.tools and a.permissions == b.permissions and a.owner == b.owner
and a.model == b.model and both role_types are 'capability'`.
`remove(a)` must refuse if `a.role_type in {'oversight','gate','parallel_worker'}`
or `a` is the only holder of a permission any consumer needs.

---

## 5. What the tool should output

Not a single number. A per-agent verdict:

```
planner      REMOVABLE      reason: no tools, no distinct scope, same model as solver;
                            ablation: -0.00 accuracy, -18% cost. Recommend: merge into solver.
solver       LOAD-BEARING   reason: does the core work; ablation -0.42 accuracy.
db_reader    KEEP           reason: distinct permission scope (db:read) not held by any
                            other agent — merging would widen write-agent's access.
critic       KEEP (weak)    reason: role_type=oversight; ablation shows -0.00 accuracy on
                            this task set, but oversight agents are protected. Flag for
                            human review: is this critic ever changing an output?
publisher    KEEP           reason: role_type=gate (human-in-the-loop approval).
researcher_2 KEEP           reason: parallel_worker in group 'research' — removing reduces
                            fan-out breadth.
```

Plus a system-level summary: *"3 of 9 agents are structurally unjustified
sequential decomposition; removing them cuts LLM calls 34% with no measured
accuracy change. The other 6 are load-bearing for context/permissions/ownership
and should stay."*

---

## 6. The research contribution, restated honestly

**Not:** "multi-agent systems are bloated, shrink them to 1."

**Yes:** "Deployed multi-agent systems mix agents that exist for hard structural
reasons (context, security, ownership, compliance, throughput) with agents that
are pure task-decomposition cargo-culting. The second kind is common, adds cost
and latency and — per MAST and our ChatDev weak-coder run — sometimes *degrades*
quality through error compounding. We give you a tool that tells the two apart,
respects the structural boundaries, and backs every 'removable' call with an
ablation."

That is a defensible, useful, and honest claim. "Shrink to 1" is none of those.

---

## Sources
- [AWS: least-privilege authorization in multi-agent AI chains (Cedar)](https://aws.amazon.com/blogs/security/enforce-least-privilege-authorization-in-multi-agent-ai-chains-using-cedar/)
- [Microsoft: least privilege for AI agents — identity, access, tool binding](https://www.microsoft.com/en-us/security/blog/2026/07/16/least-privilege-for-ai-agents-identity-access-and-tool-binding/)
- [The CASE Framework: control architecture for governing enterprise agentic AI (arXiv:2608.10153)](https://arxiv.org/html/2608.10153)
- [The New Conway's Law: how AI context windows shape enterprise architecture](https://www.cioreview.com/leadership-perspectives/the-new-conways-law-how-ai-context-windows-shape-enterprise-architecture-nid-42587-cid-175.html)
- [Towards a Science of Scaling Agent Systems (Google Research, arXiv:2512.08296)](https://arxiv.org/abs/2512.08296)
- [Single-Agent LLMs Outperform Multi-Agent Under Equal Thinking-Token Budgets (Tran & Kiela, arXiv:2604.02460)](https://arxiv.org/html/2604.02460v1)
- [Why Do Multi-Agent LLM Systems Fail? / MAST](https://futureagi.substack.com/p/why-do-multi-agent-llm-systems-fail)
- Full literature synthesis: [RESEARCH.md](RESEARCH.md)
