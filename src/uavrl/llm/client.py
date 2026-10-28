"""LLM client interface (Phase 7-8).

``LLMClient.call(prompt, features) -> (response_dict_or_None, latency_ms, ok, reason)``.

* ``MockLLMClient`` wraps MockLLM (Phases 1-7, tests, CI).
* ``RealLLMClient`` talks to any OpenAI-compatible endpoint
  (LLM_BASE_URL / LLM_API_KEY / LLM_MODEL env vars) -- Phase 8.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from .mock import MockLLM
from .schema import LLMResponse, LLMInvalid


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
    """OpenAI-compatible chat client (Phase 8).  Configured via env vars."""

    def __init__(self, model=None, base_url=None, api_key=None,
                 temperature=0.0, max_tokens=200, retry_max=1, cache=None):
        self.model = model or os.environ.get("LLM_MODEL")
        self.base_url = base_url or os.environ.get("LLM_BASE_URL")
        self.api_key = api_key or os.environ.get("LLM_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retry_max = retry_max
        self.cache = cache
        self._http = None

    def _ensure_http(self):
        if self._http is None:
            import httpx
            self._http = httpx.Client(
                base_url=self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30.0,
            )

    def call(self, prompt: str, features: Optional[dict] = None):
        if self.cache is not None:
            cached = self.cache.get(self.model, prompt)
            if cached is not None:
                return cached, 0.0, True, "cache"
        self._ensure_http()
        messages = [
            {"role": "system", "content": "Reply with a JSON object only."},
            {"role": "user", "content": prompt},
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        payload["response_format"] = {"type": "json_object"}
        resp = self._http.post("/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        try:
            response = json.loads(text)
            LLMResponse(**response)  # validate schema
            ok, reason = True, "ok"
        except (json.JSONDecodeError, ValueError, KeyError):
            response = None
            ok, reason = False, "invalid_json_or_schema"
        if ok and self.cache is not None:
            self.cache.put(self.model, prompt, response)
        return response, 0.0, ok, reason  # latency measured by caller
