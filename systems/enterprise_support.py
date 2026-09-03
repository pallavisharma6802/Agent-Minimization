"""Demo pilot: an enterprise support-triage system — the case where "collapse to
1" is obviously wrong, and the structural gate in `pareto_scan` proves it.

Agents and why each exists (see WHY_MULTIAGENT.md):

  intake          capability   – classify the ticket (no tools, no scope)   -> SWEEPABLE
  paraphraser     capability   – restate the ticket for the drafter          -> SWEEPABLE (cargo-cult)
  kb_retriever    capability   – tools={kb_search}, perms={kb:read}          -> KEEP (sole tool holder)
  db_reader       capability   – tools={db_query}, perms={db:read}, pii      -> KEEP (security + data boundary)
  drafter         capability   – write the reply                             -> SWEEPABLE
  compliance_gate gate         – policy / HITL approval                      -> KEEP (role_type=gate)
  sender          capability   – tools={email_send}, perms={email:send}      -> KEEP (write boundary — must
                                                                                never merge with a reader)

A naive accuracy-only minimizer would try to collapse all seven into one. The
structural report blocks five of them for reasons the task metric cannot see; the
sweep is only allowed to touch `intake`, `paraphraser`, `drafter`.
"""
from __future__ import annotations

import re

from agentslim import Agent, MultiAgentSystem, Task, TASK


def build() -> MultiAgentSystem:
    return MultiAgentSystem(
        name="enterprise_support",
        sink="sender",
        agents=[
            Agent("intake", "Classify the support ticket into one category and restate the ask.",
                  role_type="capability", model="mock-small", inputs=[TASK]),
            Agent("paraphraser", "Rephrase the classified ticket for the drafter. Add nothing.",
                  role_type="capability", model="mock-small", inputs=["intake"]),
            Agent("kb_retriever", "Return the relevant knowledge-base article for the ticket.",
                  role_type="capability", model="mock-small", inputs=["intake"],
                  tools={"kb_search"}, permissions={"kb:read"}),
            Agent("db_reader", "Look up the customer's account status.",
                  role_type="capability", model="mock-small", inputs=["intake"],
                  tools={"db_query"}, permissions={"db:read"}, data_scope="pii"),
            Agent("drafter", "Write the customer reply using the KB article and account status.",
                  role_type="capability", model="mock-large",
                  inputs=["paraphraser", "kb_retriever", "db_reader"]),
            Agent("compliance_gate", "Approve the reply only if it follows refund and privacy policy.",
                  role_type="gate", model="mock-large", inputs=["drafter"]),
            Agent("sender", "Send the approved reply to the customer.",
                  role_type="capability", model="mock-small", inputs=["compliance_gate"],
                  tools={"email_send"}, permissions={"email:send"}),
        ],
    )


_RAW = [
    ("I was charged twice for my subscription this month, please refund one charge.", "refund"),
    ("How do I reset my password?", "password"),
    ("My account says suspended but I paid — can you check?", "account"),
    ("I want to cancel and get a prorated refund for the unused period.", "refund"),
    ("The mobile app keeps crashing on login.", "bug"),
    ("Please update the email address on my account.", "account"),
]


def _scorer(gold: str):
    def score(output: str) -> float:
        return 1.0 if gold in (output or "").lower() else 0.0
    return score


def tasks() -> list[Task]:
    return [Task(id=f"es{i:02d}", text=q, score=_scorer(a)) for i, (q, a) in enumerate(_RAW)]
