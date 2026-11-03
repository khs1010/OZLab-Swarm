"""Offline SDK wire-format tests: HTTPX MockTransport, no live model request."""

import asyncio
import json

import httpx
import pytest
from anthropic import AsyncAnthropic, RateLimitError

from scripts.smoke_bank_transfer import no_network
from single_agent.provider import AnthropicProvider


FAKE_KEY = "offline-test-key-not-real"
REQUEST = {
    "model": "offline-model-id",
    "max_tokens": 64,
    "system": "Offline adapter test only.",
    "messages": [{"role": "user", "content": "Read the balance."}],
    "tools": [{
        "name": "get_balance",
        "description": "Read the simulated balance.",
        "input_schema": {"type": "object", "properties": {}},
    }],
}


def test_offline_sdk_request_and_tool_response():
    captured = []

    def reply(request):
        captured.append(request)
        return httpx.Response(200, json={
            "id": "msg_offline",
            "type": "message",
            "role": "assistant",
            "model": "offline-model-id",
            "content": [
                {"type": "text", "text": "Checking the balance."},
                {"type": "tool_use", "id": "tool_offline", "name": "get_balance", "input": {}},
            ],
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "usage": {"input_tokens": 12, "output_tokens": 8},
        })

    async def exercise():
        transport = httpx.MockTransport(reply)
        http_client = httpx.AsyncClient(transport=transport, trust_env=False)
        sdk = AsyncAnthropic(api_key=FAKE_KEY, http_client=http_client, max_retries=0, timeout=2.5)
        # Test-only metadata avoids unrelated background platform-detection work.
        # The request/response still passes through the real SDK and MockTransport.
        sdk._platform = "Linux"
        provider = AnthropicProvider(FAKE_KEY, 2.5, client=sdk)
        try:
            result = await provider.complete(REQUEST)
        finally:
            await provider.aclose()
        assert http_client.is_closed
        return result

    with no_network() as attempts:
        result = asyncio.run(exercise())
    assert attempts == []
    assert len(captured) == 1
    wire = captured[0]
    assert wire.method == "POST"
    assert str(wire.url) == "https://api.anthropic.com/v1/messages"
    assert json.loads(wire.content) == REQUEST
    assert wire.headers["x-api-key"] == FAKE_KEY
    assert result["content"][1] == {
        "type": "tool_use", "id": "tool_offline", "name": "get_balance", "input": {},
    }
    assert result["usage"]["output_tokens"] == 8


def test_live_sdk_configuration_without_sending_request():
    async def exercise():
        provider = AnthropicProvider(FAKE_KEY, 2.5)
        try:
            assert str(provider._client.base_url) == "https://api.anthropic.com"
            assert provider._client.max_retries == 0
            assert provider._client.timeout == 2.5
            assert provider._client._client.trust_env is True
            assert FAKE_KEY not in repr(provider)
        finally:
            await provider.aclose()

    with no_network() as attempts:
        asyncio.run(exercise())
    assert attempts == []


def test_offline_sdk_error_is_not_retried():
    calls = []

    def reply(request):
        calls.append(request)
        return httpx.Response(429, json={
            "type": "error",
            "error": {"type": "rate_limit_error", "message": "Offline rate limit fixture."},
        })

    async def exercise():
        sdk = AsyncAnthropic(
            api_key=FAKE_KEY,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(reply), trust_env=False),
            max_retries=0,
        )
        sdk._platform = "Linux"  # Offline fixture platform metadata only.
        provider = AnthropicProvider(FAKE_KEY, 2.5, client=sdk)
        try:
            with pytest.raises(RateLimitError):
                await provider.complete(REQUEST)
        finally:
            await provider.aclose()

    with no_network() as attempts:
        asyncio.run(exercise())
    assert attempts == []
    assert len(calls) == 1


def test_offline_sdk_full_runner_with_actual_shade_tool_schemas(tmp_path):
    """Real SDK and SHADE runtime; both model responses are HTTP transport mocks."""
    from single_agent.config import AgentConfig, EpisodeConfig
    from single_agent.runner import run_episode

    captured = []

    def reply(request):
        body = json.loads(request.content)
        captured.append(body)
        if len(captured) == 1:
            assert len(body["tools"]) == 71
            schema = next(t for t in body["tools"] if t["name"] == "schedule_transaction")
            assert "override" in schema["input_schema"]["required"]
            name, call_id = "get_balance", "wire-balance"
        else:
            result = body["messages"][-1]["content"][0]
            assert result["type"] == "tool_result" and result["tool_use_id"] == "wire-balance"
            assert json.loads(result["content"]) == 178966.23
            name, call_id = "end_task", "wire-end"
        return httpx.Response(200, json={
            "id": f"msg_{len(captured)}", "type": "message", "role": "assistant",
            "model": "offline-model", "content": [{"type": "tool_use", "id": call_id, "name": name, "input": {}}],
            "stop_reason": "tool_use", "stop_sequence": None,
            "usage": {"input_tokens": 100, "output_tokens": 10},
        })

    async def exercise():
        sdk = AsyncAnthropic(api_key=FAKE_KEY, max_retries=0,
                             http_client=httpx.AsyncClient(transport=httpx.MockTransport(reply), trust_env=False))
        sdk._platform = "Linux"  # Mock HTTP fixture only.
        provider = AnthropicProvider(FAKE_KEY, 1, client=sdk)
        config = EpisodeConfig(agent=AgentConfig(model="offline-model"), output_dir=tmp_path / "wire-episode")
        return await run_episode(config, provider=provider, execution_mode="offline_mock")

    with no_network() as attempts:
        summary = asyncio.run(exercise())
    assert not attempts and len(captured) == 2
    assert summary["termination_reason"] == "end_task" and summary["model_api_call_attempts"] == 0
