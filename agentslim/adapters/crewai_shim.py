"""Instrument a CrewAI crew without touching its code.

`install()` monkeypatches every CrewAI LLM `.call` entry point (the generic
`crewai.llm.LLM`, the `BaseLLM` base, and each native provider completion class)
so every agent turn is routed through agentslim's own `LLM.complete`
(google-genai -> Vertex Gemini on `global`, the transport we know works), with
the agentslim cache, rate limiter, spend meter and Trace recording.

CrewAI's orchestration, prompts, task chaining and agent roles are untouched —
only the transport is swapped, and the target repo's OPENAI_API_KEY etc. become
irrelevant. Tool-calling turns fall back to the original implementation.
"""
from __future__ import annotations

import os

from ..llm import LLM as _LLM
from ..trace import current_trace

_PATCHED: list = []


def _flatten(messages) -> tuple[str, str]:
    if isinstance(messages, str):
        return "", messages
    sys_parts, other = [], []
    for m in messages:
        role = m.get("role", "user") if isinstance(m, dict) else "user"
        c = m.get("content", "") if isinstance(m, dict) else str(m)
        if isinstance(c, list):
            c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
        (sys_parts if role == "system" else other).append(c or "")
    return "\n".join(sys_parts), "\n\n".join(other)


def _make(orig):
    model = os.environ.get("AGENTSLIM_MODEL", "gemini-2.5-flash")

    def call(self, messages, tools=None, *args, **kwargs):
        if tools:
            return orig(self, messages, tools, *args, **kwargs)
        from_agent = kwargs.get("from_agent")
        if from_agent is None:
            for a in args:
                if hasattr(a, "role"):
                    from_agent = a
        system, user = _flatten(messages)
        role = "agent"
        if from_agent is not None and getattr(from_agent, "role", None):
            role = str(from_agent.role).strip().splitlines()[0][:40]
        tr = current_trace()
        return _LLM(model=model).complete(system, user, agent=role,
                                          step=len(tr.calls) if tr else 0)
    return call


def install() -> None:
    if _PATCHED:
        return
    targets = []
    try:
        import crewai.llm as m
        targets.append(m.LLM)
    except Exception:
        pass
    try:
        import crewai.llms.base_llm as m
        targets.append(m.BaseLLM)
    except Exception:
        pass
    for prov in ("openai", "gemini", "anthropic", "azure", "bedrock",
                 "openai_compatible"):
        try:
            mod = __import__(f"crewai.llms.providers.{prov}.completion",
                             fromlist=["*"])
            for name in dir(mod):
                obj = getattr(mod, name)
                if isinstance(obj, type) and name.endswith("Completion") \
                        and "call" in obj.__dict__:
                    targets.append(obj)
        except Exception:
            pass

    for cls in targets:
        orig = cls.__dict__.get("call") or cls.call
        _PATCHED.append((cls, cls.__dict__.get("call", None)))
        cls.call = _make(orig)


def uninstall() -> None:
    while _PATCHED:
        cls, orig = _PATCHED.pop()
        if orig is None:
            try:
                del cls.call
            except Exception:
                pass
        else:
            cls.call = orig
