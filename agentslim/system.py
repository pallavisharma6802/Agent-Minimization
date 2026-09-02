"""Framework-neutral representation of a multi-agent system as a DAG of agents.

Real repos (LangGraph, CrewAI, AutoGen) get adapters that lower into this form
later. For pilots we author systems directly here so the causal graph is exact.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Optional

from .llm import LLM
from .trace import Trace

TASK = "__task__"


@dataclass
class Agent:
    name: str
    system_prompt: str
    kind: str = "capability"           # "capability" (minimize) | "oversight" (protect)
    model: str = "mock-small"
    inputs: list[str] = field(default_factory=lambda: [TASK])
    # non-LLM replacement installed by an ablation; takes (task_text, upstream) -> str
    fn: Optional[Callable[[str, dict], str]] = None

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
                llm = LLM(model=a.model)
                user = a.render_user(task_text, upstream)
                outputs[a.name] = llm.complete(a.system_prompt, user, agent=a.name, step=step)
            final = outputs.get(self.sink, "")
            tr.final_output = final
        return final, tr
