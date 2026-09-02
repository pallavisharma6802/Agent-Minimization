"""Instrument OpenBMB/ChatDev (classic, v1.1.x) without editing its code.

`install()` replaces `camel.model_backend.OpenAIModel.run` so every agent turn in
ChatDev's 7-role software company is routed through agentslim's Vertex Gemini
transport, with the agentslim cache / rate limiter / spend meter / Trace. The
returned object is a real `openai` `ChatCompletion` so ChatDev's downstream code
(`response.choices[0].message.content`, `response.usage.*`) is unaffected.

ChatDev's phase pipeline (which agents participate) is controlled from the
runner by editing the loaded ChatChainConfig — see experiments/repos/chatdev_company.py.
"""
from __future__ import annotations

import os
import time

from ..llm import CACHE, METER, RATE, _estimate_tokens, _PRICING
from ..trace import CallRecord, current_trace

_INSTALLED = False


def _role_hint(messages) -> str:
    for m in messages:
        if m.get("role") == "system":
            c = m.get("content") or ""
            # ChatDev system prompts start with "... you are <Role> ..."
            for kw in ("Chief Executive Officer", "Chief Product Officer",
                       "Chief Technology Officer", "Chief Human Resource Officer",
                       "Programmer", "Code Reviewer", "Software Test Engineer",
                       "Chief Creative Officer", "Counselor"):
                if kw in c:
                    return kw
            return c.strip().split(".")[0][:40]
    return "agent"


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    import camel.model_backend as mb
    from openai.types.chat import ChatCompletion
    from openai.types.chat.chat_completion import Choice
    from openai.types.chat.chat_completion_message import ChatCompletionMessage
    from openai.types.completion_usage import CompletionUsage

    # newer openai ChatCompletionMessage carries refusal/annotations/audio/...
    # ChatDev does `ChatMessage(**dict(choice.message))` and only knows role/content.
    import inspect
    import camel.messages.chat_messages as _cm
    for _clsname in ("ChatMessage", "AssistantChatMessage", "UserChatMessage"):
        _cls = getattr(_cm, _clsname, None)
        if _cls is None or getattr(_cls, "_agentslim_patched", False):
            continue
        _orig_init = _cls.__init__
        _ok = set(inspect.signature(_orig_init).parameters)

        def _mk(orig, ok):
            def __init__(self, *a, **kw):
                orig(self, *a, **{k: v for k, v in kw.items() if k in ok})
            return __init__
        _cls.__init__ = _mk(_orig_init, _ok)
        _cls._agentslim_patched = True

    model = os.environ.get("AGENTSLIM_MODEL", "gemini-2.5-flash")

    def run(self, *args, **kwargs):
        messages = kwargs["messages"]
        system = "\n".join(m["content"] for m in messages if m.get("role") == "system")
        convo = "\n\n".join(f'{m["role"]}: {m["content"]}'
                            for m in messages if m.get("role") != "system")
        role = _role_hint(messages)
        parts = ("chatdev", model, system, convo)

        cached = CACHE.get(*parts)
        t0 = time.time()
        if cached is not None:
            text = cached
        else:
            RATE.wait()
            from ..llm import LLM
            text = LLM(model=model).complete(system, convo, agent=role)
            CACHE.put(text, *parts)

        in_tok = _estimate_tokens(system) + _estimate_tokens(convo)
        out_tok = _estimate_tokens(text)
        pin, pout = _PRICING.get(model, (0.5, 1.5))
        cost = (in_tok * pin + out_tok * pout) / 1_000_000
        if cached is None:
            # LLM.complete already metered it; nothing to do
            pass
        else:
            METER.charge(cost, live=False)

        tr = current_trace()
        if tr is not None and cached is not None:
            tr.record(CallRecord(agent=role, step=len(tr.calls), model=model,
                                 system=system, user=convo, output=text,
                                 input_tokens=in_tok, output_tokens=out_tok,
                                 cost_usd=cost, latency_s=time.time() - t0))

        # model_construct: skip validation AND keep __dict__ minimal so ChatDev's
        # `dict(choice.message)` yields only role/content (newer openai adds
        # refusal/annotations/audio fields that ChatMessage() would choke on).
        msg = ChatCompletionMessage.model_construct(role="assistant", content=text)
        choice = Choice.model_construct(index=0, finish_reason="stop", message=msg)
        usage = CompletionUsage.model_construct(
            prompt_tokens=in_tok, completion_tokens=out_tok, total_tokens=in_tok + out_tok)
        return ChatCompletion.model_construct(
            id="agentslim", created=int(time.time()), model=model,
            object="chat.completion", choices=[choice], usage=usage)

    mb.OpenAIModel.run = run
    _INSTALLED = True
