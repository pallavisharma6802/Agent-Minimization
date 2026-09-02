"""Pluggable LLM backend with per-call logging.

Every agent LLM call in the framework goes through `LLM.complete`. The active
`Trace` (see agentslim.trace) records the call so ablations can reason about
which calls happened, how big their context was, and what they cost.

Backends:
  - "mock":     deterministic, no network. For plumbing / CI.
  - "openai":   needs `pip install openai` and OPENAI_API_KEY.
  - "anthropic":needs `pip install anthropic` and ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .trace import CallRecord, current_trace

# rough $/1M tokens (input, output); only used for relative comparison
_PRICING = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "claude-3-5-haiku-20241022": (0.80, 4.00),
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "mock-small": (0.10, 0.40),
    "mock-large": (2.00, 8.00),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-3.5-flash": (0.30, 2.50),
    "gemini-flash-latest": (0.30, 2.50),
}


def _estimate_tokens(text: str) -> int:
    # ~4 chars/token, good enough for relative accounting
    return max(1, len(text) // 4)


class SpendExceeded(RuntimeError):
    pass


class _Meter:
    """Process-wide running cost. Hard-aborts a run before it can eat credits.
    Cap via AGENTSLIM_MAX_USD (default $5). Set to 0 to disable."""
    def __init__(self) -> None:
        self.spent = 0.0
        self.calls = 0

    @property
    def cap(self) -> float:
        return float(os.environ.get("AGENTSLIM_MAX_USD", "5"))

    def charge(self, usd: float) -> None:
        self.spent += usd
        self.calls += 1
        if self.cap > 0 and self.spent > self.cap:
            raise SpendExceeded(
                f"cost ceiling hit: ${self.spent:.4f} > ${self.cap:.2f} after {self.calls} calls "
                f"(raise AGENTSLIM_MAX_USD to continue)")


METER = _Meter()


class _RateLimiter:
    """Simple token-bucket-ish throttle. AGENTSLIM_RPM requests/min (default 12,
    just under AI Studio free-tier limits). Thread-safe."""
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._times: list[float] = []

    def wait(self) -> None:
        rpm = float(os.environ.get("AGENTSLIM_RPM", "12"))
        if rpm <= 0:
            return
        with self._lock:
            now = time.time()
            self._times = [t for t in self._times if now - t < 60]
            if len(self._times) >= rpm:
                sleep_for = 60 - (now - self._times[0]) + 0.05
                if sleep_for > 0:
                    time.sleep(sleep_for)
                    now = time.time()
                    self._times = [t for t in self._times if now - t < 60]
            self._times.append(time.time())


RATE = _RateLimiter()


class _Cache:
    """On-disk response cache keyed by (backend, model, system, user). Makes
    re-running a sweep after a crash cheap and keeps ablation trials that reissue
    identical prompts from paying twice. Dir: AGENTSLIM_CACHE or .llm_cache/"""
    def __init__(self) -> None:
        self.dir = Path(os.environ.get("AGENTSLIM_CACHE", ".llm_cache"))

    def _key(self, *parts: str) -> Path:
        h = hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:32]
        return self.dir / f"{h}.json"

    def get(self, *parts: str) -> Optional[str]:
        if os.environ.get("AGENTSLIM_NOCACHE"):
            return None
        p = self._key(*parts)
        if p.exists():
            try:
                return json.loads(p.read_text())["output"]
            except Exception:
                return None
        return None

    def put(self, output: str, *parts: str) -> None:
        if os.environ.get("AGENTSLIM_NOCACHE"):
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        self._key(*parts).write_text(json.dumps({"output": output}))


CACHE = _Cache()


@dataclass
class LLM:
    model: str = "mock-small"
    backend: str = field(default_factory=lambda: os.environ.get("AGENTSLIM_BACKEND", "mock"))
    temperature: float = 0.0
    _client: object = field(default=None, repr=False)

    def complete(self, system: str, user: str, *, agent: str = "?", step: int = 0) -> str:
        cache_parts = (self.backend, self.model, system, user)
        cached = CACHE.get(*cache_parts)
        t0 = time.time()
        if cached is not None:
            out, dt = cached, 0.0
        else:
            if self.backend != "mock":
                RATE.wait()
            out = self._retrying_dispatch(system, user)
            dt = time.time() - t0
            CACHE.put(out, *cache_parts)

        in_tok = _estimate_tokens(system) + _estimate_tokens(user)
        out_tok = _estimate_tokens(out)
        pin, pout = _PRICING.get(self.model, (0.5, 1.5))
        cost = (in_tok * pin + out_tok * pout) / 1_000_000

        if self.backend != "mock":
            METER.charge(cost)

        tr = current_trace()
        if tr is not None:
            tr.record(CallRecord(
                agent=agent, step=step, model=self.model,
                system=system, user=user, output=out,
                input_tokens=in_tok, output_tokens=out_tok,
                cost_usd=cost, latency_s=dt,
            ))
        return out

    def _retrying_dispatch(self, system: str, user: str, tries: int = 5) -> str:
        delay = 2.0
        for i in range(tries):
            try:
                return self._dispatch(system, user)
            except Exception as e:  # noqa: BLE001
                msg = str(e).lower()
                transient = any(s in msg for s in
                                ("quota", "exhaust", "429", "resource_exhausted",
                                 "503", "unavailable", "500", "deadline", "timeout"))
                if not transient or i == tries - 1:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 60)
        raise RuntimeError("unreachable")

    # -- backends ---------------------------------------------------------
    def _dispatch(self, system: str, user: str) -> str:
        if self.backend == "mock":
            return _mock_complete(self.model, system, user)
        if self.backend == "openai":
            return self._openai(system, user)
        if self.backend == "anthropic":
            return self._anthropic(system, user)
        if self.backend == "gemini":
            return self._gemini(system, user)
        raise ValueError(f"unknown backend {self.backend!r}")

    def _gemini(self, system: str, user: str) -> str:
        if self._client is None:
            from google import genai
            if os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true"):
                # Vertex AI: auth via Application Default Credentials, billed to
                # GOOGLE_CLOUD_PROJECT (consumes the GCP free-trial credits).
                self._client = genai.Client(
                    vertexai=True,
                    project=os.environ["GOOGLE_CLOUD_PROJECT"],
                    location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
                )
            else:
                # Google AI Studio: simple API key, has a no-billing free tier.
                self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        from google.genai import types
        cfg = dict(system_instruction=system, temperature=self.temperature,
                   max_output_tokens=int(os.environ.get("AGENTSLIM_MAX_TOKENS", "512")))
        # kill thinking-token spend where the model/SDK supports it
        try:
            budget = int(os.environ.get("AGENTSLIM_THINKING_BUDGET", "0"))
            cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=budget)
        except Exception:
            pass
        try:
            r = self._client.models.generate_content(
                model=self.model, contents=user,
                config=types.GenerateContentConfig(**cfg))
        except Exception:
            cfg.pop("thinking_config", None)
            r = self._client.models.generate_content(
                model=self.model, contents=user,
                config=types.GenerateContentConfig(**cfg))
        return r.text or ""

    def _openai(self, system: str, user: str) -> str:
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()
        r = self._client.chat.completions.create(
            model=self.model, temperature=self.temperature,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        return r.choices[0].message.content or ""

    def _anthropic(self, system: str, user: str) -> str:
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        r = self._client.messages.create(
            model=self.model, max_tokens=1024, temperature=self.temperature,
            system=system, messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in r.content if getattr(b, "type", None) == "text")


# ----------------------------------------------------------------------
# Mock backend: a tiny "reasoner" that can actually do the pilot math tasks
# so the ablation harness produces meaningful (non-random) metric deltas
# without a network. It is deliberately imperfect: it drops ~1 in 6 hard
# problems unless a checker agent is present in the pipeline.
# ----------------------------------------------------------------------
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _mock_complete(model: str, system: str, user: str) -> str:
    role = system.lower()
    # The pilot passes the raw problem plus any upstream agent notes in `user`.
    problem = user
    nums = [float(x) for x in _NUM.findall(problem)]

    if "decompose" in role or "planner" in role:
        return f"Steps: identify quantities {nums[:4]}, combine per the question, report the number."

    if "checker" in role or "verifier" in role or "oversight" in role:
        # recompute independently; if it disagrees with upstream, correct it
        val = _mock_solve(problem, nums, careful=True)
        return f"VERIFIED: {val}"

    if "format" in role or "extractor" in role:
        found = _NUM.findall(problem)
        return found[-1] if found else "0"

    # default: the solver
    careful = "mock-large" in model
    return str(_mock_solve(problem, nums, careful=careful))


def _mock_solve(problem: str, nums, careful: bool) -> float:
    p = problem.lower()
    try:
        if "each" in p and "how many" in p and len(nums) >= 2:
            v = nums[0] * nums[1]
            if len(nums) >= 3 and ("more" in p or "additional" in p or "then" in p):
                v += nums[2]
            return _round(v)
        if ("total" in p or "altogether" in p or "sum" in p) and nums:
            return _round(sum(nums[:3]))
        if ("left" in p or "remain" in p or "spends" in p or "gives away" in p) and len(nums) >= 2:
            return _round(nums[0] - sum(nums[1:3]))
        if "times" in p and len(nums) >= 2:
            return _round(nums[0] * nums[1])
        if len(nums) >= 2:
            v = nums[0] + nums[1]
            # simulate an unforced error the checker would catch
            if not careful and len(nums) >= 3:
                return _round(v)  # ignores nums[2] -> wrong on 3-quantity problems
            return _round(sum(nums[:3]))
    except Exception:
        pass
    return _round(nums[-1] if nums else 0.0)


def _round(x: float) -> float:
    return round(x, 2) if x != int(x) else float(int(x))
