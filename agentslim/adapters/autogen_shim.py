"""A minimal AutoGen (autogen-agentchat v0.4+) `ChatCompletionClient` backed by
agentslim's `LLM` (Vertex Gemini / Groq / mock) with the agentslim cache, rate
limiter, spend meter and Trace. Pass `GeminiChatClient()` as any agent's
`model_client=`.

No tool calling — returns plain text. Fine for GroupChat routing / debate.
"""
from __future__ import annotations

import os

from autogen_core import CancellationToken
from autogen_core.models import (
    ChatCompletionClient, CreateResult, LLMMessage, ModelInfo, RequestUsage,
)

from ..llm import LLM as _LLM
from ..trace import current_trace


def _flatten(messages):
    sys_parts, other = [], []
    for m in messages:
        c = getattr(m, "content", "")
        if not isinstance(c, str):
            c = str(c)
        src = getattr(m, "source", "") or getattr(m, "type", "")
        if src == "system" or m.__class__.__name__ == "SystemMessage":
            sys_parts.append(c)
        else:
            other.append(f"{src}: {c}" if src else c)
    return "\n".join(sys_parts), "\n\n".join(other)


class GeminiChatClient(ChatCompletionClient):
    def __init__(self, model: str | None = None, agent_tag: str = "agent"):
        self._model = model or os.environ.get("AGENTSLIM_MODEL", "gemini-2.5-flash")
        self._tag = agent_tag
        self._usage = RequestUsage(prompt_tokens=0, completion_tokens=0)

    async def create(self, messages, *, tools=[], tool_choice="auto", json_output=None,
                     extra_create_args={}, cancellation_token=None) -> CreateResult:
        system, user = _flatten(messages)
        tr = current_trace()
        text = _LLM(model=self._model).complete(
            system, user, agent=self._tag, step=len(tr.calls) if tr else 0)
        pt = len(system + user) // 4
        ct = len(text) // 4
        self._usage = RequestUsage(prompt_tokens=self._usage.prompt_tokens + pt,
                                   completion_tokens=self._usage.completion_tokens + ct)
        return CreateResult(content=text, finish_reason="stop",
                            usage=RequestUsage(prompt_tokens=pt, completion_tokens=ct),
                            cached=False)

    async def create_stream(self, *a, **k):
        res = await self.create(*a, **k)
        yield res

    async def close(self): ...
    def actual_usage(self): return self._usage
    def total_usage(self): return self._usage
    def count_tokens(self, messages, **k): return sum(len(str(getattr(m, "content", ""))) // 4 for m in messages)
    def remaining_tokens(self, messages, **k): return 100_000

    @property
    def capabilities(self):
        return self.model_info

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(vision=False, function_calling=False, json_output=False,
                         family="unknown", structured_output=False)
