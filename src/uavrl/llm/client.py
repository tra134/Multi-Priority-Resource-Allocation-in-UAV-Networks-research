"""LLM client interface (Phase 7-8).

``LLMClient.call(prompt, features) -> (response_dict_or_None, latency_ms, ok, reason)``.

* ``MockLLMClient`` wraps MockLLM (Phases 1-7, tests, CI).
* ``RealLLMClient`` talks to any OpenAI-compatible endpoint
  (``LLM_BASE_URL`` / ``LLM_API_KEY`` / ``LLM_MODEL`` env vars, loaded from
  the local ``.env`` automatically) -- Phase 8.

The real client never raises out of ``call``: network errors, malformed JSON
and schema violations are reported as ``ok=False`` so the controller can
activate the safety fallback.  Wall-clock latency and token usage are recorded
for cost reporting.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Optional

from .mock import MockLLM
from .schema import LLMInvalid, LLMResponse


def _strip_json_fences(text: str) -> str:
    """Remove common Markdown fences around a JSON object."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


class LLMClient:
    def call(self, prompt: str, features: Optional[dict] = None):
        raise NotImplementedError


class MockLLMClient(LLMClient):
    def __init__(self, model="mock", latency_median_ms=350.0,
                 latency_sigma=0.6, failure_injection=None, rng=None,
                 cache=None):
        self.llm = MockLLM(
            model=model,
            latency_median_ms=latency_median_ms,
            latency_sigma=latency_sigma,
            failure_injection=failure_injection,
            rng=rng,
        )
        self.cache = cache

    def call(self, prompt: str, features: Optional[dict] = None):
        if self.cache is not None:
            cached = self.cache.get(self.llm.model, prompt)
            if cached is not None:
                return cached, 0.0, True, "cache"
        response, latency_ms, ok, reason = self.llm.call(prompt, features)
        if ok and self.cache is not None:
            self.cache.put(self.llm.model, prompt, response)
        return response, latency_ms, ok, reason


class RealLLMClient(LLMClient):
    """OpenAI-compatible chat client (Phase 8), configured via env/config.

    Works with Groq, OpenRouter, OpenAI, Ollama and vLLM.  Requests JSON output
    (``response_format=json_object``) and validates the result against
    :class:`~uavrl.llm.schema.LLMResponse`.

    Parameters
    ----------
    client : optional
        An already-constructed OpenAI client (used by tests / dependency
        injection).  When omitted, the official ``openai`` SDK is used.
    json_mode : bool
        Send ``response_format={"type": "json_object"}``.  If the endpoint
        rejects it, the client automatically retries without it.
    """

    def __init__(self, model=None, base_url=None, api_key=None,
                 temperature=0.0, max_tokens=200, retry_max=1, cache=None,
                 timeout_s=30.0, json_mode=True, client=None,
                 load_dotenv=True):
        if load_dotenv:
            from ..config import load_env
            load_env()
        self.model = model or os.environ.get("LLM_MODEL")
        self.base_url = base_url or os.environ.get("LLM_BASE_URL")
        self.api_key = api_key or os.environ.get("LLM_API_KEY")
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.retry_max = int(retry_max)
        self.cache = cache
        self.timeout_s = float(timeout_s)
        self.json_mode = bool(json_mode)
        self._client = client

        self.calls = 0
        self.successes = 0
        self.failures = 0
        self.cache_hits = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.last_usage: dict = {}
        self.last_error: Optional[str] = None

    # ------------------------------------------------------------------ setup
    def _ensure_client(self):
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise RuntimeError(
                "LLM_API_KEY is not set. Put it in .env (gitignored) or export it."
            )
        if not self.base_url:
            raise RuntimeError(
                "LLM_BASE_URL is not set. Example Groq: "
                "https://api.groq.com/openai/v1"
            )
        if not self.model:
            raise RuntimeError("LLM_MODEL is not set (e.g. openai/gpt-oss-120b).")
        from openai import OpenAI
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout_s,
            max_retries=0,  # retries are handled here, so they are counted
        )
        return self._client

    # ------------------------------------------------------------------ call
    def call(self, prompt: str, features: Optional[dict] = None):
        self.calls += 1
        if self.cache is not None:
            cached = self.cache.get(self.model, prompt)
            if cached is not None:
                self.cache_hits += 1
                return cached, 0.0, True, "cache"

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a safety arbiter for a wireless resource allocator. "
                    "Reply with one JSON object only; no prose, no code fences."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        last_reason = "unknown"
        last_latency = 0.0
        attempts = max(1, self.retry_max + 1)
        for attempt in range(attempts):
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            if self.json_mode:
                payload["response_format"] = {"type": "json_object"}

            t0 = time.perf_counter()
            try:
                client = self._ensure_client()
                completion = client.chat.completions.create(**payload)
            except Exception as exc:  # noqa: BLE001 - never crash the fast loop
                last_latency = (time.perf_counter() - t0) * 1000.0
                last_reason = f"request_error:{type(exc).__name__}:{exc}"
                self.last_error = last_reason
                # Some OpenAI-compatible servers do not implement json_object.
                if self.json_mode and _is_response_format_error(exc):
                    self.json_mode = False
                    continue
                if attempt + 1 < attempts:
                    continue
                self.failures += 1
                return None, last_latency, False, last_reason

            latency_ms = (time.perf_counter() - t0) * 1000.0
            last_latency = latency_ms
            self._record_usage(completion)

            try:
                text = completion.choices[0].message.content
            except (AttributeError, IndexError, TypeError) as exc:
                last_reason = f"bad_response:{type(exc).__name__}"
                self.last_error = last_reason
                if attempt + 1 < attempts:
                    continue
                self.failures += 1
                return None, latency_ms, False, last_reason

            try:
                obj = json.loads(_strip_json_fences(text or ""))
                LLMResponse(**obj)  # schema validation
            except (json.JSONDecodeError, TypeError, ValueError, KeyError,
                    LLMInvalid) as exc:
                last_reason = f"invalid_json_or_schema:{type(exc).__name__}"
                self.last_error = last_reason
                if attempt + 1 < attempts:
                    # Add the parser error to the retry prompt.
                    messages = messages + [
                        {"role": "assistant", "content": text or ""},
                        {
                            "role": "user",
                            "content": (
                                "That response was invalid (%s). Reply again with "
                                "only a valid JSON object matching the requested schema."
                                % last_reason
                            ),
                        },
                    ]
                    continue
                self.failures += 1
                return None, latency_ms, False, last_reason

            self.successes += 1
            if self.cache is not None:
                self.cache.put(self.model, prompt, obj)
            return obj, latency_ms, True, "ok"

        self.failures += 1
        return None, last_latency, False, last_reason

    def _record_usage(self, completion):
        usage = getattr(completion, "usage", None)
        if usage is None:
            self.last_usage = {}
            return
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.last_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        }

    def stats(self) -> dict:
        return {
            "model": self.model,
            "base_url": self.base_url,
            "calls": self.calls,
            "successes": self.successes,
            "failures": self.failures,
            "cache_hits": self.cache_hits,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "last_error": self.last_error,
        }


def _is_response_format_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "response_format" in text or "json_object" in text or "json mode" in text


def build_llm_client(config_path=None, use_mock=False, cache=None, mock_kwargs=None):
    """Build a real (default) or mock client from ``configs/llm.yaml``.

    Environment variables override the YAML values.  Useful from scripts so
    the same code path works for MockLLM sweeps and real-LLM runs.
    """
    from ..config import llm_settings
    from .cache import LLMCache

    cfg = llm_settings(config_path)
    cache_cfg = cfg.get("cache", {})
    if cache is None and cache_cfg.get("enabled", True):
        cache = LLMCache(enabled=True, dir=cache_cfg.get("dir", "results/llm_cache"))

    if use_mock:
        kwargs = dict(mock_kwargs or {})
        return MockLLMClient(model=kwargs.pop("model", "mock"), cache=cache, **kwargs)

    return RealLLMClient(
        model=cfg.get("model"),
        base_url=cfg.get("base_url"),
        api_key=cfg.get("api_key"),
        temperature=cfg.get("temperature", 0.0),
        max_tokens=cfg.get("max_tokens", 200),
        retry_max=cfg.get("retry_max", 1),
        timeout_s=cfg.get("timeout_s", 30.0),
        json_mode=cfg.get("response_format_json", True),
        cache=cache,
    )
