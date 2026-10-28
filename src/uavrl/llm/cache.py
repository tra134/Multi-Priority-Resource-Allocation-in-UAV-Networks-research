"""Disk cache for LLM responses (Phase 7-8).

Keyed by hash(model + prompt).  Numeric prompt features are rounded to 2
significant figures by the prompt builder so identical situations hit the
cache.  Cache files live in results/llm_cache/ (also copied to Drive on Colab).
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Optional


class LLMCache:
    def __init__(self, enabled=True, dir="results/llm_cache"):
        self.enabled = enabled
        self.dir = dir
        if self.enabled:
            os.makedirs(self.dir, exist_ok=True)

    def _key(self, model: str, prompt: str) -> str:
        h = hashlib.sha256((model + "\n" + prompt).encode("utf-8")).hexdigest()
        return os.path.join(self.dir, f"{h}.json")

    def get(self, model: str, prompt: str) -> Optional[dict]:
        if not self.enabled:
            return None
        path = self._key(model, prompt)
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def put(self, model: str, prompt: str, response: dict):
        if not self.enabled:
            return
        path = self._key(model, prompt)
        try:
            with open(path, "w") as f:
                json.dump(response, f)
        except OSError:
            pass
