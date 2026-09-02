"""A LangChain `BaseChatModel` backed by agentslim's LLM (google-genai -> Vertex
Gemini on `global`), with the agentslim cache / rate limiter / spend meter /
Trace. Pass an instance anywhere a LangGraph builder wants a `model=`.

Tool calling is supported minimally: `bind_tools` stores the schemas and we ask
the model to emit a JSON tool call, parsed back into an AIMessage tool_calls.
Good enough for supervisor-style routing; not a full function-calling client.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from ..llm import LLM as _LLM
from ..trace import current_trace


def _text(m) -> str:
    c = getattr(m, "content", "")
    if isinstance(c, list):
        return " ".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c or ""


class GeminiChat(BaseChatModel):
    model_name: str = os.environ.get("AGENTSLIM_MODEL", "gemini-2.5-flash")
    agent_tag: str = "agent"
    _tools: list = []

    @property
    def _llm_type(self) -> str:
        return "agentslim-gemini"

    def bind_tools(self, tools, **kwargs):
        clone = self.__class__(model_name=self.model_name, agent_tag=self.agent_tag)
        specs = []
        for t in tools:
            name = getattr(t, "name", None) or getattr(t, "__name__", str(t))
            desc = getattr(t, "description", "") or ""
            specs.append({"name": name, "description": desc})
        clone._tools = specs
        return clone

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        system = "\n".join(_text(m) for m in messages if m.type == "system")
        convo = "\n\n".join(f"{m.type}: {_text(m)}" for m in messages if m.type != "system")
        if self._tools:
            names = ", ".join(t["name"] for t in self._tools)
            system += (f"\n\nYou may hand off by replying with ONLY a JSON object "
                       f'{{"tool":"<name>","args":{{}}}} where <name> is one of: {names}. '
                       f"Otherwise reply normally.")

        tr = current_trace()
        out = _LLM(model=self.model_name).complete(
            system, convo, agent=self.agent_tag,
            step=len(tr.calls) if tr else 0)

        tool_calls = []
        m = re.search(r'\{[^{}]*"tool"\s*:\s*"([^"]+)"[^{}]*\}', out)
        if m and self._tools:
            try:
                obj = json.loads(m.group(0))
                tool_calls = [{"name": obj["tool"], "args": obj.get("args", {}),
                               "id": f"call_{len(tr.calls) if tr else 0}"}]
                out = ""
            except Exception:
                pass

        msg = AIMessage(content=out, tool_calls=tool_calls)
        return ChatResult(generations=[ChatGeneration(message=msg)])
