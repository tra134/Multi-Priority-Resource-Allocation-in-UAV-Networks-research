"""Tests for the real OpenAI-compatible LLM client (Phase 8).

No network is used: a fake OpenAI client is injected so the parsing, retry and
failure paths are deterministic.
"""

from __future__ import annotations

from types import SimpleNamespace

from uavrl.llm.client import RealLLMClient, _strip_json_fences
from uavrl.llm.schema import LLMResponse

VALID = {
    "operating_mode": "SOS_SURGE",
    "min_bandwidth_ratio": {"emergency": 0.6, "important": 0.2, "normal": 0.15},
    "power_boost_emergency": True,
    "ttl_duration_slots": 300,
}


def _completion(content, prompt_tokens=10, completion_tokens=20):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _client(responses, retry_max=1):
    fake = FakeCompletions(responses)
    openai_client = SimpleNamespace(
        chat=SimpleNamespace(completions=fake)
    )
    client = RealLLMClient(
        model="fake-model",
        base_url="http://localhost/v1",
        api_key="test-key",
        retry_max=retry_max,
        load_dotenv=False,
        client=openai_client,
    )
    return client, fake


def test_strip_json_fences():
    assert _strip_json_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert _strip_json_fences('{"a": 1}') == '{"a": 1}'


def test_real_client_parses_valid_json_and_records_usage():
    import json

    client, fake = _client([_completion(json.dumps(VALID))])
    response, latency_ms, ok, reason = client.call("prompt")
    assert ok and reason == "ok"
    assert response == VALID
    assert latency_ms >= 0.0
    assert fake.calls == 1
    # Schema validation is part of the client contract.
    assert LLMResponse(**response).operating_mode == "SOS_SURGE"
    assert client.last_usage["prompt_tokens"] == 10
    assert client.last_usage["completion_tokens"] == 20
    assert client.stats()["successes"] == 1


def test_real_client_retries_invalid_json_then_succeeds():
    import json

    client, fake = _client([
        _completion("not json"),
        _completion(json.dumps(VALID)),
    ])
    response, _, ok, reason = client.call("prompt")
    assert ok and reason == "ok"
    assert response == VALID
    assert fake.calls == 2


def test_real_client_returns_failure_without_raising():
    client, fake = _client([RuntimeError("network down")])
    response, _, ok, reason = client.call("prompt")
    assert response is None
    assert not ok
    assert "request_error" in reason
    assert fake.calls == 2  # one retry
    assert client.stats()["failures"] == 1


def test_real_client_validates_schema():
    import json

    bad = dict(VALID)
    bad["min_bandwidth_ratio"] = {"emergency": 0.9, "important": 0.9, "normal": 0.9}
    client, fake = _client([_completion(json.dumps(bad))], retry_max=0)
    response, _, ok, reason = client.call("prompt")
    assert response is None
    assert not ok
    assert "invalid_json_or_schema" in reason


def test_real_client_cache_hit():
    import json
    import tempfile

    from uavrl.llm.cache import LLMCache

    with tempfile.TemporaryDirectory() as d:
        cache = LLMCache(enabled=True, dir=d)
        fake = FakeCompletions([_completion(json.dumps(VALID))])
        openai_client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
        client = RealLLMClient(
            model="fake-model", base_url="http://localhost/v1",
            api_key="test-key", load_dotenv=False, client=openai_client,
            cache=cache,
        )
        r1, _, ok1, _ = client.call("same prompt")
        r2, lat2, ok2, reason2 = client.call("same prompt")
        assert ok1 and ok2
        assert r1 == r2 == VALID
        assert reason2 == "cache"
        assert lat2 == 0.0
        assert fake.calls == 1
        assert client.stats()["cache_hits"] == 1


def test_real_client_missing_key_returns_failure(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    client = RealLLMClient(
        model="fake", base_url="http://localhost/v1", api_key="",
        load_dotenv=False,
    )
    response, _, ok, reason = client.call("prompt")
    assert response is None
    assert not ok
    assert "LLM_API_KEY" in reason or "RuntimeError" in reason
