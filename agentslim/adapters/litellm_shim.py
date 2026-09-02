"""Global instrumentation for any framework that routes LLM calls through
litellm (CrewAI, many others).

`install()` monkeypatches `litellm.completion` to:
  - force every call onto our Vertex Gemini model (so the target repo's default
    OpenAI/Anthropic config is irrelevant and we spend only GCP credits),
  - apply the agentslim rate limiter, disk cache, and spend meter,
  - record a CallRecord on the active Trace, tagged with the caller's agent role
    when we can recover it from the messages.

This keeps the target repo's code path intact — same orchestration, same
prompts — while giving us the trace and the cost controls.
"""
from __future__ import annotations

import os
import re
import time

from ..llm import CACHE, METER, RATE, _estimate_tokens, _PRICING
from ..trace import CallRecord, current_trace

_ORIG = None
_MODEL = None


def _target_model() -> str:
    m = os.environ.get("AGENTSLIM_MODEL", "gemini-2.5-flash")
    # litellm wants a provider prefix for Vertex
    if os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true"):
        return f"vertex_ai/{m}"
    return f"gemini/{m}"


def _role_of(messages: list[dict]) -> str:
    for msg in messages:
        if msg.get("role") == "system":
            c = msg.get("content") or ""
            m = re.search(r"role[:\s]+([A-Za-z0-9 _-]{3,40})", c, re.I)
            if m:
                return m.group(1).strip()[:40]
            return (c.strip().split("\n", 1)[0])[:40] or "system"
    return "agent"


def _flatten(messages: list[dict]) -> tuple[str, str]:
    sys_parts, user_parts = [], []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
        c = c or ""
        (sys_parts if m.get("role") == "system" else user_parts).append(c)
    return "\n".join(sys_parts), "\n".join(user_parts)


def install() -> None:
    global _ORIG, _MODEL
    import litellm

    if _ORIG is not None:
        return
    _ORIG = litellm.completion
    _MODEL = _target_model()

    if os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true"):
        os.environ.setdefault("VERTEXAI_PROJECT", os.environ.get("GOOGLE_CLOUD_PROJECT", ""))
        os.environ.setdefault("VERTEXAI_LOCATION", os.environ.get("GOOGLE_CLOUD_LOCATION", "global"))

    def patched(*args, **kwargs):
        messages = kwargs.get("messages") or (args[1] if len(args) > 1 else args[0])
        kwargs["messages"] = messages
        if args:
            args = ()
        kwargs["model"] = _MODEL
        kwargs.pop("api_key", None)
        kwargs.setdefault("temperature", 0.0)

        system, user = _flatten(messages)
        role = _role_of(messages)
        parts = ("litellm", _MODEL, system, user)

        cached = CACHE.get(*parts)
        t0 = time.time()
        if cached is not None:
            text = cached
        else:
            RATE.wait()
            delay = 2.0
            for i in range(5):
                try:
                    resp = _ORIG(**kwargs)
                    break
                except Exception as e:  # noqa: BLE001
                    if i == 4 or not any(s in str(e).lower() for s in
                                         ("quota", "exhaust", "429", "503", "500",
                                          "unavailable", "timeout", "deadline")):
                        raise
                    time.sleep(delay); delay = min(delay * 2, 60)
            text = resp.choices[0].message.content or ""
            CACHE.put(text, *parts)

        in_tok = _estimate_tokens(system) + _estimate_tokens(user)
        out_tok = _estimate_tokens(text)
        pin, pout = _PRICING.get(_MODEL.split("/")[-1], (0.5, 1.5))
        cost = (in_tok * pin + out_tok * pout) / 1_000_000
        METER.charge(cost)

        tr = current_trace()
        if tr is not None:
            tr.record(CallRecord(agent=role, step=len(tr.calls), model=_MODEL,
                                 system=system, user=user, output=text,
                                 input_tokens=in_tok, output_tokens=out_tok,
                                 cost_usd=cost, latency_s=time.time() - t0))

        if cached is not None:
            # fabricate a minimal litellm-shaped response
            from litellm import ModelResponse
            mr = ModelResponse()
            mr.choices[0].message.content = text
            return mr
        return resp

    litellm.completion = patched


def uninstall() -> None:
    global _ORIG
    if _ORIG is not None:
        import litellm
        litellm.completion = _ORIG
        _ORIG = None
