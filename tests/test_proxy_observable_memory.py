"""Tests for ObservableMemoryHandler and ProxyLLMBridge."""
from __future__ import annotations

import pytest


class TestProxyLLMBridge:
    def test_count_tokens_returns_int(self):
        from headroom.proxy.observable_memory_handler import ProxyLLMBridge
        bridge = ProxyLLMBridge(api_key="test-key")
        result = bridge.count_tokens("hello world", "gpt-4o")
        assert isinstance(result, int)
        assert result > 0

    def test_count_tokens_empty_string(self):
        from headroom.proxy.observable_memory_handler import ProxyLLMBridge
        bridge = ProxyLLMBridge(api_key="test-key")
        assert bridge.count_tokens("", "gpt-4o") == 0

    @pytest.mark.asyncio
    async def test_complete_calls_litellm(self, monkeypatch):
        from headroom.proxy.observable_memory_handler import ProxyLLMBridge

        calls = []

        class FakeChoice:
            class message:
                content = "observer output"

        class FakeResponse:
            choices = [FakeChoice()]

        async def fake_acompletion(**kwargs):
            calls.append(kwargs)
            return FakeResponse()

        import litellm
        monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

        bridge = ProxyLLMBridge(api_key="sk-test", api_base="http://localhost:9001")
        result = await bridge.complete("sys prompt", "user prompt", "claude-haiku-4-5")

        assert result == "observer output"
        assert len(calls) == 1
        assert calls[0]["model"] == "claude-haiku-4-5"
        assert calls[0]["api_key"] == "sk-test"
        assert calls[0]["api_base"] == "http://localhost:9001"
        assert calls[0]["messages"][0] == {"role": "system", "content": "sys prompt"}
        assert calls[0]["messages"][1] == {"role": "user", "content": "user prompt"}

    @pytest.mark.asyncio
    async def test_complete_returns_empty_string_on_none_content(self, monkeypatch):
        from headroom.proxy.observable_memory_handler import ProxyLLMBridge

        class FakeChoice:
            class message:
                content = None

        class FakeResponse:
            choices = [FakeChoice()]

        async def fake_acompletion(**kwargs):
            return FakeResponse()

        import litellm
        monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

        bridge = ProxyLLMBridge(api_key="sk-test")
        result = await bridge.complete("sys", "prompt", "gpt-4o-mini")
        assert result == ""
