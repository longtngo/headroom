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

    @pytest.mark.asyncio
    async def test_complete_propagates_litellm_error(self, monkeypatch):
        from headroom.proxy.observable_memory_handler import ProxyLLMBridge

        async def exploding_acompletion(**kwargs):
            raise RuntimeError("network failure")

        import litellm
        monkeypatch.setattr(litellm, "acompletion", exploding_acompletion)

        bridge = ProxyLLMBridge(api_key="sk-test")
        with pytest.raises(RuntimeError, match="network failure"):
            await bridge.complete("sys", "prompt", "gpt-4o")


class TestResolveThreadId:
    """Tests for resolve_thread_id and _derive_thread_id."""

    def _call(self, headers=None, messages=None, body=None, user_id=None):
        from headroom.proxy.observable_memory_handler import resolve_thread_id
        return resolve_thread_id(
            headers=headers or {},
            messages=messages or [],
            body=body or {},
            user_id=user_id,
        )

    # --- Explicit header priority ---

    def test_headroom_native_header_wins(self):
        result = self._call(
            headers={"x-headroom-thread-id": "my-thread", "helicone-session-id": "other"},
            messages=[{"role": "user", "content": "hi"}],
            body={"system": "sys"},
        )
        assert result == "my-thread"

    def test_helicone_header_used_when_no_headroom(self):
        result = self._call(
            headers={"helicone-session-id": "helicone-abc"},
            messages=[{"role": "user", "content": "hi"}],
        )
        assert result == "helicone-abc"

    def test_portkey_header_used(self):
        result = self._call(
            headers={"x-portkey-trace-id": "portkey-xyz"},
            messages=[{"role": "user", "content": "hi"}],
        )
        assert result == "portkey-xyz"

    def test_mcp_session_id_used(self):
        result = self._call(
            headers={"mcp-session-id": "mcp-session-123"},
            messages=[{"role": "user", "content": "hi"}],
        )
        assert result == "mcp-session-123"

    def test_headers_are_case_insensitive(self):
        result = self._call(
            headers={"X-Headroom-Thread-Id": "case-test"},
        )
        assert result == "case-test"

    # --- Content hash fallback ---

    def test_derives_id_from_anthropic_body(self):
        """Anthropic: system is top-level body field, messages[0] is user."""
        result = self._call(
            headers={},
            messages=[{"role": "user", "content": "Tell me about Python"}],
            body={"system": "You are a helpful assistant."},
            user_id="user-alice",
        )
        assert result is not None
        assert len(result) == 16  # sha256 hex prefix

    def test_derives_id_from_openai_body(self):
        """OpenAI: system is first message with role==system."""
        result = self._call(
            headers={},
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Tell me about Python"},
            ],
            body={},
            user_id="user-alice",
        )
        assert result is not None
        assert len(result) == 16

    def test_hash_is_stable(self):
        """Same inputs always produce same thread ID."""
        args = dict(
            headers={},
            messages=[{"role": "user", "content": "Hello, I need help"}],
            body={"system": "You are an expert."},
            user_id="user-bob",
        )
        r1 = self._call(**args)
        r2 = self._call(**args)
        assert r1 == r2

    def test_different_first_messages_produce_different_ids(self):
        base = dict(body={"system": "sys"}, user_id="user-1", headers={})
        r1 = self._call(messages=[{"role": "user", "content": "Question A"}], **base)
        r2 = self._call(messages=[{"role": "user", "content": "Question B"}], **base)
        assert r1 != r2

    def test_returns_none_when_messages_empty(self):
        result = self._call(headers={}, messages=[], body={})
        assert result is None

    def test_multimodal_content_extracts_text(self):
        """Content as list of blocks — extract text parts."""
        result = self._call(
            headers={},
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "Describe this image"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ]}],
            body={},
        )
        assert result is not None

    def test_anonymous_user_id_still_derives_id(self):
        """No user_id falls back to 'anonymous' salt."""
        result = self._call(
            headers={},
            messages=[{"role": "user", "content": "hello"}],
            body={"system": "sys"},
            user_id=None,
        )
        assert result is not None

    def test_empty_header_value_falls_through_to_hash(self):
        """An explicitly empty header value should fall through to hash derivation."""
        result = self._call(
            headers={"x-headroom-thread-id": ""},
            messages=[{"role": "user", "content": "hello"}],
            body={"system": "sys"},
        )
        # Should derive a hash, not return empty string
        assert result is not None
        assert result != ""
        assert len(result) == 16

    def test_extract_text_content_null_text_field(self):
        """Content blocks with text=null should not crash."""
        result = self._call(
            headers={},
            messages=[{"role": "user", "content": [
                {"type": "text", "text": None},
                {"type": "text", "text": "real content"},
            ]}],
            body={},
        )
        assert result is not None
