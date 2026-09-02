"""Execution trace: every LLM call made during one task run, plus helpers to
derive the causal graph (which agent's output fed which agent's input)."""
from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from typing import Optional

_ACTIVE: contextvars.ContextVar[Optional["Trace"]] = contextvars.ContextVar("trace", default=None)


def current_trace() -> Optional["Trace"]:
    return _ACTIVE.get()


@dataclass
class CallRecord:
    agent: str
    step: int
    model: str
    system: str
    user: str
    output: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_s: float


@dataclass
class Trace:
    task_id: str = "?"
    calls: list[CallRecord] = field(default_factory=list)
    # names of agents whose output text appeared verbatim in a later agent's input
    consumed_by: dict[str, set[str]] = field(default_factory=dict)
    final_output: str = ""

    def __enter__(self):
        self._token = _ACTIVE.set(self)
        return self

    def __exit__(self, *exc):
        _ACTIVE.reset(self._token)
        return False

    def record(self, rec: CallRecord) -> None:
        # attribute upstream provenance: did any prior agent's output land in this user prompt?
        for prior in self.calls:
            if prior.output and prior.output.strip() and prior.output.strip()[:60] in rec.user:
                self.consumed_by.setdefault(prior.agent, set()).add(rec.agent)
        self.calls.append(rec)

    # -- aggregates -----------------------------------------------------
    @property
    def n_calls(self) -> int:
        return len(self.calls)

    @property
    def total_tokens(self) -> int:
        return sum(c.input_tokens + c.output_tokens for c in self.calls)

    @property
    def cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    @property
    def latency_s(self) -> float:
        return sum(c.latency_s for c in self.calls)

    @property
    def agents(self) -> list[str]:
        seen = []
        for c in self.calls:
            if c.agent not in seen:
                seen.append(c.agent)
        return seen

    def summary(self) -> dict:
        return {
            "task_id": self.task_id,
            "n_calls": self.n_calls,
            "n_agents": len(self.agents),
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "latency_s": round(self.latency_s, 3),
            "consumed_by": {k: sorted(v) for k, v in self.consumed_by.items()},
        }
