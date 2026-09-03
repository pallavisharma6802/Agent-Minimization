"""Framework-neutral representation of a multi-agent system as a DAG of agents.

Real repos (LangGraph, CrewAI, AutoGen) get adapters that lower into this form
later. For pilots we author systems directly here so the causal graph is exact.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Optional

import os

from .llm import LLM
from .trace import Trace


def _resolve_model(tag: str) -> str:
    """On a real backend, map the pilot's abstract size tag to a concrete model.
    'mock-large' -> AGENTSLIM_MODEL_LARGE (falls back to AGENTSLIM_MODEL)."""
    if os.environ.get("AGENTSLIM_BACKEND", "mock") == "mock":
        return tag
    base = os.environ.get("AGENTSLIM_MODEL", "gemini-2.5-flash")
    if "large" in tag:
        return os.environ.get("AGENTSLIM_MODEL_LARGE", base)
    return os.environ.get("AGENTSLIM_MODEL_SMALL", base)

TASK = "__task__"


# role_type drives what the minimizer is even allowed to touch. See WHY_MULTIAGENT.md.
#   capability      – does task work; the ONLY type eligible for removal/merge
#   oversight       – monitor / critic / verifier; protected (place well, don't cut)
#   gate            – human-in-the-loop / compliance approval; never cut
#   parallel_worker – one of a fan-out group; cutting reduces throughput/breadth
#   router          – dispatches to others; cutting collapses the topology
ROLE_TYPES = ("capability", "oversight", "gate", "parallel_worker", "router")


@dataclass
class Agent:
    name: str
    system_prompt: str
    kind: str = "capability"           # legacy alias for role_type; kept in sync
    model: str = "mock-small"
    inputs: list[str] = field(default_factory=lambda: [TASK])
    # non-LLM replacement installed by an ablation; takes (task_text, upstream) -> str
    fn: Optional[Callable[[str, dict], str]] = None

    # --- structural metadata: what makes this agent un-mergeable / un-removable ---
    role_type: str = "capability"
    tools: frozenset = field(default_factory=frozenset)      # distinct tools => distinct capability
    permissions: frozenset = field(default_factory=frozenset)  # distinct scope => security boundary
    data_scope: str = ""                                     # e.g. "pii", "tenant:A", "aggregate"
    owner: str = ""                                          # team / CODEOWNERS / registry entry
    parallel_group: str = ""                                 # agents sharing this are fan-out workers

    def __post_init__(self):
        # keep the legacy `kind` field and `role_type` consistent
        if self.role_type == "capability" and self.kind != "capability":
            self.role_type = "oversight" if self.kind == "oversight" else self.kind
        if self.kind == "capability" and self.role_type != "capability":
            self.kind = self.role_type
        self.tools = frozenset(self.tools)
        self.permissions = frozenset(self.permissions)

    def boundary_key(self) -> tuple:
        """Two agents may only be merged if their boundary keys match."""
        return (self.role_type, self.model, self.tools, self.permissions,
                self.data_scope, self.owner, self.parallel_group)

    def protected(self) -> bool:
        """True => the minimizer must not remove or merge this agent at all."""
        return self.role_type in ("oversight", "gate", "parallel_worker", "router")

    def render_user(self, task_text: str, upstream: dict[str, str]) -> str:
        parts = [f"PROBLEM:\n{task_text}"]
        for src in self.inputs:
            if src != TASK and src in upstream:
                parts.append(f"\n[{src} said]:\n{upstream[src]}")
        return "\n".join(parts)


@dataclass
class MultiAgentSystem:
    name: str
    agents: list[Agent]
    sink: str

    def clone(self) -> "MultiAgentSystem":
        return copy.deepcopy(self)

    def by_name(self, n: str) -> Agent:
        return next(a for a in self.agents if a.name == n)

    # -- structural gating: what the minimizer is allowed to touch ----------
    def structural_block(self, name: str) -> str | None:
        """Return a human-readable reason this agent must NOT be removed/merged,
        or None if it is a legitimate ablation candidate. See WHY_MULTIAGENT.md."""
        a = self.by_name(name)
        if a.protected():
            return f"role_type={a.role_type} (protected — not a capability agent)"
        if a.tools:
            sole = a.tools - frozenset().union(
                *[x.tools for x in self.agents if x.name != name]) if self.agents else a.tools
            if sole:
                return f"sole holder of tools {set(sole)} — removing loses that capability"
        if a.permissions:
            others = frozenset().union(*[x.permissions for x in self.agents if x.name != name]) \
                if len(self.agents) > 1 else frozenset()
            sole_perm = a.permissions - others
            if sole_perm:
                return (f"sole holder of permissions {set(sole_perm)} — a security boundary; "
                        f"merging would widen another agent's access")
        if a.data_scope and a.data_scope not in ("", "aggregate", "public"):
            return f"data_scope={a.data_scope!r} — a data-governance boundary"
        if a.owner and len({x.owner for x in self.agents if x.owner}) > 1:
            return f"owner={a.owner!r} — independently owned; merging crosses an org boundary"
        return None

    def merge_allowed(self, a_name: str, b_name: str) -> str | None:
        """None if a and b may be merged; else the reason they may not."""
        a, b = self.by_name(a_name), self.by_name(b_name)
        if a.role_type != "capability" or b.role_type != "capability":
            return "one side is not a capability agent"
        if a.model != b.model:
            return f"different models ({a.model} vs {b.model}) — heterogeneity is intentional"
        if a.tools != b.tools:
            return "different tool sets"
        if a.permissions != b.permissions or a.data_scope != b.data_scope:
            return "different permission / data scope — security boundary"
        if a.owner != b.owner and (a.owner or b.owner):
            return f"different owners ({a.owner!r} vs {b.owner!r}) — org boundary"
        return None

    def _order(self) -> list[Agent]:
        names = {a.name for a in self.agents}
        done: list[str] = []
        out: list[Agent] = []
        guard = 0
        while len(out) < len(self.agents):
            guard += 1
            if guard > 500:
                raise RuntimeError("cycle or missing input in agent graph")
            for a in self.agents:
                if a.name in done:
                    continue
                deps = [s for s in a.inputs if s != TASK and s in names]
                if all(d in done for d in deps):
                    out.append(a); done.append(a.name)
        return out

    def run(self, task_text: str, task_id: str = "?") -> tuple[str, Trace]:
        tr = Trace(task_id=task_id)
        with tr:
            outputs: dict[str, str] = {}
            for step, a in enumerate(self._order()):
                upstream = {s: outputs[s] for s in a.inputs if s in outputs}
                if a.fn is not None:
                    outputs[a.name] = a.fn(task_text, upstream)
                    continue
                llm = LLM(model=_resolve_model(a.model))
                user = a.render_user(task_text, upstream)
                outputs[a.name] = llm.complete(a.system_prompt, user, agent=a.name, step=step)
            final = outputs.get(self.sink, "")
            tr.final_output = final
        return final, tr
