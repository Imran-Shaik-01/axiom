import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import providers


class FakeStream:
    def __init__(self, tokens=None, error=None):
        self.tokens = list(tokens or [])
        self.error = error

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.tokens:
            return self.tokens.pop(0)
        if self.error:
            err, self.error = self.error, None
            raise err
        raise StopAsyncIteration

    async def aclose(self):
        return


@pytest.mark.asyncio
async def test_failover_to_second_provider(monkeypatch):
    providers.reset_state()
    providers.PROVIDERS["groq"]["keys"] = ["g1"]
    providers.PROVIDERS["gemini"]["keys"] = ["e1"]
    providers.PROVIDERS["openrouter"]["keys"] = []
    providers.ORDER[:] = ["groq", "gemini", "openrouter"]

    calls = []

    def fake_stream_one(client, name, key, model, messages, temperature=None):
        calls.append(name)
        if name == "groq":
            return FakeStream(error=providers.ProviderError(429, "rate", 1))
        return FakeStream(tokens=["Hello", " world"])

    monkeypatch.setattr(providers, "stream_one", fake_stream_one)
    monkeypatch.setattr(providers, "get_client", lambda: object())

    events = []
    async for ev, data in providers.stream_with_failover(
        [{"role": "user", "content": "hi"}], internal=True
    ):
        events.append((ev, data))

    kinds = [e[0] for e in events]
    assert "groq" in calls and "gemini" in calls
    assert "token" in kinds
    assert "done" in kinds
    text = "".join(d.get("text", "") for k, d in events if k == "token")
    assert "Hello" in text


@pytest.mark.asyncio
async def test_continuation_strips_think(monkeypatch):
    providers.reset_state()
    providers.PROVIDERS["groq"]["keys"] = ["g1"]
    providers.PROVIDERS["gemini"]["keys"] = ["e1"]
    providers.PROVIDERS["openrouter"]["keys"] = []
    providers.ORDER[:] = ["groq", "gemini"]

    seen = []

    def fake_stream_one(client, name, key, model, messages, temperature=None):
        seen.append((name, messages))
        if name == "groq":
            return FakeStream(tokens=["<think>nope</think>partial"], error=providers.ProviderError(503, "break", 1))
        return FakeStream(tokens=[" more"])

    monkeypatch.setattr(providers, "stream_one", fake_stream_one)
    monkeypatch.setattr(providers, "get_client", lambda: object())

    async for _ev, _data in providers.stream_with_failover(
        [{"role": "user", "content": "hi"}], internal=True
    ):
        pass

    gemini_msgs = next(messages for name, messages in seen if name == "gemini")
    # reasoning must not be replayed to the next model
    dumped = str(gemini_msgs)
    assert "<think>" not in dumped
    assert "partial" in dumped


def test_auto_route_prefers_observed_low_latency(monkeypatch):
    providers.circuit_open_until.clear()
    monkeypatch.setattr(providers, "ORDER", ["groq", "gemini", "openrouter"])
    monkeypatch.setattr(providers, "ewma_ttft", {"gemini": 80.0, "groq": 110.0})

    assert providers._provider_order("auto") == ["gemini", "groq", "openrouter"]


@pytest.mark.parametrize(
    ("error", "counts"),
    [
        (providers.ProviderError(400, "bad request"), False),
        (providers.ProviderError(429, "rate limited"), True),
        (providers.ProviderError(503, "unavailable"), True),
        (providers.httpx.ReadTimeout("timeout"), True),
        (providers.httpx.ConnectError("connection failed"), False),
    ],
)
def test_circuit_only_counts_rate_server_errors_and_timeouts(error, counts):
    assert providers._should_count_circuit_failure(error) is counts


@pytest.mark.asyncio
async def test_byok_failure_does_not_cool_down_server_key_or_trip_circuit(monkeypatch):
    providers.reset_state()
    monkeypatch.setattr(providers, "ORDER", ["groq"])
    monkeypatch.setitem(providers.PROVIDERS["groq"], "keys", ["server-key"])
    monkeypatch.setitem(providers.PROVIDERS["groq"], "models", ["test-model"])
    monkeypatch.setitem(providers.PROVIDERS["gemini"], "keys", [])
    monkeypatch.setitem(providers.PROVIDERS["openrouter"], "keys", [])
    providers._rr["groq"] = 0

    def fake_stream_one(client, name, key, model, messages, temperature=None):
        return FakeStream(error=providers.ProviderError(401, "invalid key"))

    monkeypatch.setattr(providers, "stream_one", fake_stream_one)
    monkeypatch.setattr(providers, "get_client", lambda: object())

    async for _event, _data in providers.stream_with_failover(
        [{"role": "user", "content": "hi"}], extra_keys={"groq": "invalid-byok-key"}, internal=True
    ):
        pass

    assert providers.cooldown_until == {}
    assert providers.model_cooldown == {}
    assert providers.circuit_fails == {}
    assert providers.circuit_open_until == {}
    assert providers.pick_node(set())[1:3] == (0, "server-key")


@pytest.mark.asyncio
async def test_bad_requests_do_not_count_toward_provider_circuit(monkeypatch):
    providers.reset_state()
    monkeypatch.setattr(providers, "ORDER", ["groq"])
    monkeypatch.setitem(providers.PROVIDERS["groq"], "keys", ["server-key"])
    monkeypatch.setitem(providers.PROVIDERS["groq"], "models", ["test-model"])
    monkeypatch.setitem(providers.PROVIDERS["gemini"], "keys", [])
    monkeypatch.setitem(providers.PROVIDERS["openrouter"], "keys", [])

    async def bad_request(*args, **kwargs):
        raise providers.ProviderError(400, "invalid request")

    monkeypatch.setattr(providers, "complete_one", bad_request)
    monkeypatch.setattr(providers, "get_client", lambda: object())

    for _ in range(4):
        providers.cooldown_until.clear()
        await providers.complete_with_failover([{"role": "user", "content": "hi"}])

    assert providers.circuit_fails.get("groq", 0) == 0
    assert "groq" not in providers.circuit_open_until
