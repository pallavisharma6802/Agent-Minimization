"""Ablation strategies — the "different ways" to shrink a multi-agent system.

Each returns a new MultiAgentSystem. By default these primitives now REFUSE to
touch an agent that a structural boundary protects (distinct tools / permissions /
data scope / owner / model, or a non-capability role_type) — see WHY_MULTIAGENT.md
for why "collapse to 1" is the wrong objective. Pass force=True only for
research/what-if runs that deliberately ignore boundaries.
"""
from __future__ import annotations

from typing import Callable

from .system import TASK, Agent, MultiAgentSystem


class BoundaryViolation(RuntimeError):
    """Raised when an ablation would cross a structural boundary."""


def _rewire_consumers(sys: MultiAgentSystem, dropped: str, replacement_inputs: list[str]) -> None:
    for a in sys.agents:
        if dropped in a.inputs:
            i = a.inputs.index(dropped)
            a.inputs = a.inputs[:i] + [s for s in replacement_inputs if s not in a.inputs] + a.inputs[i + 1:]


def remove(sys: MultiAgentSystem, name: str, force: bool = False) -> MultiAgentSystem:
    """Delete an agent; its consumers now read its inputs directly."""
    if not force:
        block = sys.structural_block(name)
        if block:
            raise BoundaryViolation(f"cannot remove {name!r}: {block}")
    s = sys.clone()
    victim = s.by_name(name)
    if s.sink == name:
        # promote its sole upstream (or first) to sink
        ups = [x for x in victim.inputs if x != TASK]
        s.sink = ups[0] if ups else TASK
    _rewire_consumers(s, name, victim.inputs)
    s.agents = [a for a in s.agents if a.name != name]
    return s


def identity(sys: MultiAgentSystem, name: str, force: bool = False) -> MultiAgentSystem:
    """Replace an agent with a no-LLM pass-through of its upstream text."""
    if not force:
        block = sys.structural_block(name)
        if block:
            raise BoundaryViolation(f"cannot neutralize {name!r}: {block}")
    s = sys.clone()
    a = s.by_name(name)

    def _passthrough(task_text: str, upstream: dict) -> str:
        vals = [v for k, v in upstream.items()]
        return vals[-1] if vals else task_text

    a.fn = _passthrough
    return s


def heuristic(sys: MultiAgentSystem, name: str, fn: Callable[[str, dict], str]) -> MultiAgentSystem:
    """Replace an agent with a cheap deterministic rule."""
    s = sys.clone()
    s.by_name(name).fn = fn
    return s


def downgrade(sys: MultiAgentSystem, name: str, model: str) -> MultiAgentSystem:
    s = sys.clone()
    s.by_name(name).model = model
    return s


def merge(sys: MultiAgentSystem, a_name: str, b_name: str, force: bool = False) -> MultiAgentSystem:
    """Fold b into a: one agent, concatenated role prompts, unioned inputs.
    Consumers of either now consume the merged agent. Refuses across any
    structural boundary unless force=True."""
    if not force:
        why = sys.merge_allowed(a_name, b_name)
        if why:
            raise BoundaryViolation(f"cannot merge {a_name!r}+{b_name!r}: {why}")
    s = sys.clone()
    a, b = s.by_name(a_name), s.by_name(b_name)
    merged = Agent(
        name=a_name,
        system_prompt=a.system_prompt.rstrip() + "\n\nAlso: " + b.system_prompt.strip(),
        kind="oversight" if "oversight" in (a.kind, b.kind) else "capability",
        model=a.model if _price_rank(a.model) >= _price_rank(b.model) else b.model,
        role_type=a.role_type, tools=a.tools | b.tools, permissions=a.permissions | b.permissions,
        data_scope=a.data_scope or b.data_scope, owner=a.owner or b.owner,
        inputs=[x for x in (a.inputs + b.inputs) if x != a_name and x != b_name] or [TASK],
    )
    # dedupe inputs, keep order
    seen, ins = set(), []
    for x in merged.inputs:
        if x not in seen:
            seen.add(x); ins.append(x)
    merged.inputs = ins

    s.agents = [merged if x.name == a_name else x for x in s.agents if x.name != b_name]
    _rewire_consumers(s, b_name, [a_name])
    if s.sink == b_name:
        s.sink = a_name
    return s


def _price_rank(model: str) -> int:
    return 1 if "large" in model or "sonnet" in model or "4o" in model and "mini" not in model else 0
