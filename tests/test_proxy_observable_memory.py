"""Tests for ObservableMemoryHandler and ProxyLLMBridge."""
from __future__ import annotations

import asyncio

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
        args = {
            "headers": {},
            "messages": [{"role": "user", "content": "Hello, I need help"}],
            "body": {"system": "You are an expert."},
            "user_id": "user-bob",
        }
        r1 = self._call(**args)
        r2 = self._call(**args)
        assert r1 == r2

    def test_different_first_messages_produce_different_ids(self):
        base = {"body": {"system": "sys"}, "user_id": "user-1", "headers": {}}
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


class FakeProc:
    """Fake ObservableMemoryProcessor for handler tests."""

    def __init__(self, stored: str | None = None):
        self.stored = stored
        self.observe_calls: list[dict] = []
        self.saved: dict[str, str] = {}

    async def get_observations(self, thread_id: str) -> str | None:
        return self.stored

    async def observe(self, thread_id, messages, model, context_window, **kwargs):
        self.observe_calls.append({
            "thread_id": thread_id,
            "messages": messages,
            "model": model,
            "context_window": context_window,
        })


class TestObservableMemoryHandler:

    def _make_handler(self, stored: str | None = None):
        from headroom.observable_memory import ObservableMemoryConfig
        from headroom.proxy.observable_memory_handler import ObservableMemoryHandler, ProxyLLMBridge

        config = ObservableMemoryConfig(enabled=True, observer_model="gpt-4o-mini")
        bridge = ProxyLLMBridge(api_key="sk-test")
        handler = ObservableMemoryHandler(config=config, llm=bridge)
        # Replace internal processor with fake
        handler._proc = FakeProc(stored=stored)
        return handler

    # --- inject_observations ---

    @pytest.mark.asyncio
    async def test_inject_observations_anthropic_appends_to_system(self):
        handler = self._make_handler(stored="* 🔴 (14:30) user prefers Python")
        body = {"system": "You are a helpful assistant.", "messages": []}
        await handler.inject_observations("thread-1", body, provider="anthropic")
        assert "<memory>" in body["system"]
        assert "user prefers Python" in body["system"]
        assert body["system"].startswith("You are a helpful assistant.")

    @pytest.mark.asyncio
    async def test_inject_observations_anthropic_no_existing_system(self):
        handler = self._make_handler(stored="* 🔴 (14:30) test obs")
        body = {"messages": []}
        await handler.inject_observations("thread-1", body, provider="anthropic")
        assert body["system"] == "<memory>\n* 🔴 (14:30) test obs\n</memory>"

    @pytest.mark.asyncio
    async def test_inject_observations_openai_appends_to_system_message(self):
        handler = self._make_handler(stored="* 🟡 (14:30) working on feature X")
        body = {"messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]}
        await handler.inject_observations("thread-1", body, provider="openai")
        sys_msg = body["messages"][0]
        assert "<memory>" in sys_msg["content"]
        assert "working on feature X" in sys_msg["content"]

    @pytest.mark.asyncio
    async def test_inject_observations_openai_no_system_prepends(self):
        handler = self._make_handler(stored="* 🟢 (14:30) background fact")
        body = {"messages": [{"role": "user", "content": "Hello"}]}
        await handler.inject_observations("thread-1", body, provider="openai")
        assert body["messages"][0]["role"] == "system"
        assert "<memory>" in body["messages"][0]["content"]

    @pytest.mark.asyncio
    async def test_inject_observations_noop_when_no_observations(self):
        handler = self._make_handler(stored=None)
        body = {"system": "You are helpful.", "messages": []}
        original_system = body["system"]
        await handler.inject_observations("thread-1", body, provider="anthropic")
        assert body["system"] == original_system  # unchanged

    @pytest.mark.asyncio
    async def test_inject_observations_openai_list_content_appends_block(self):
        """OpenAI system message with list content gets a new text block appended."""
        handler = self._make_handler(stored="* 🔴 (14:30) user prefers TypeScript")
        body = {"messages": [
            {"role": "system", "content": [{"type": "text", "text": "You are helpful."}]},
            {"role": "user", "content": "Hello"},
        ]}
        await handler.inject_observations("thread-1", body, provider="openai")
        sys_content = body["messages"][0]["content"]
        assert isinstance(sys_content, list)
        assert any(
            b.get("type") == "text" and "<memory>" in b.get("text", "")
            for b in sys_content
        )

    @pytest.mark.asyncio
    async def test_inject_observations_anthropic_list_system_appends_block(self):
        """Anthropic system as list of blocks gets a new text block appended."""
        handler = self._make_handler(stored="* 🔴 (14:30) user prefers Python")
        body = {"system": [{"type": "text", "text": "You are helpful."}], "messages": []}
        await handler.inject_observations("thread-1", body, provider="anthropic")
        sys_content = body["system"]
        assert isinstance(sys_content, list)
        assert any(
            b.get("type") == "text" and "<memory>" in b.get("text", "")
            for b in sys_content
        )

    # --- schedule_observe ---

    @pytest.mark.asyncio
    async def test_schedule_observe_creates_task(self):
        import asyncio
        handler = self._make_handler()
        messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]

        handler.schedule_observe(
            thread_id="thread-1",
            messages=messages,
            model="claude-opus-4-6",
            context_window=128_000,
        )

        # Give the event loop a tick to run the created task
        await asyncio.sleep(0)
        proc: FakeProc = handler._proc  # type: ignore
        assert len(proc.observe_calls) == 1
        assert proc.observe_calls[0]["thread_id"] == "thread-1"
        assert proc.observe_calls[0]["model"] == "claude-opus-4-6"


class TestProxyConfig:

    def test_observable_memory_disabled_by_default(self):
        from headroom.proxy.server import HeadroomProxyConfig
        config = HeadroomProxyConfig()
        assert config.observable_memory_enabled is False

    def test_observable_memory_config_fields_exist(self):
        from headroom.proxy.server import HeadroomProxyConfig
        config = HeadroomProxyConfig(
            observable_memory_enabled=True,
            observable_memory_observer_model="claude-haiku-4-5-20251001",
            observable_memory_reflector_model="claude-haiku-4-5-20251001",
            observable_memory_db_path="/tmp/om.db",
            observable_memory_message_threshold_ratio=0.3,
            observable_memory_observation_threshold_ratio=0.4,
            observable_memory_instruction="Focus on errors.",
            observable_memory_observer_api_key="sk-other",
        )
        assert config.observable_memory_enabled is True
        assert config.observable_memory_observer_model == "claude-haiku-4-5-20251001"
        assert config.observable_memory_reflector_model == "claude-haiku-4-5-20251001"
        assert config.observable_memory_db_path == "/tmp/om.db"
        assert config.observable_memory_message_threshold_ratio == 0.3
        assert config.observable_memory_observation_threshold_ratio == 0.4
        assert config.observable_memory_instruction == "Focus on errors."
        assert config.observable_memory_observer_api_key == "sk-other"


class TestAnthropicHandlerObservableMemory:
    """Integration test: OM observations injected into Anthropic requests."""

    @pytest.mark.asyncio
    async def test_observations_injected_into_system_prompt(self, monkeypatch):
        """When OM has stored observations, they appear in the forwarded request body."""
        import json
        from unittest.mock import MagicMock

        from starlette.requests import Request

        from headroom.proxy.server import HeadroomProxy, HeadroomProxyConfig

        # Build proxy with OM enabled
        config = HeadroomProxyConfig(
            optimize=False,
            observable_memory_enabled=True,
            observable_memory_observer_model="claude-haiku-4-5-20251001",
        )
        proxy = HeadroomProxy(config)

        # Inject a fake handler whose processor has stored observations
        from headroom.observable_memory import ObservableMemoryConfig, ObservableMemoryProcessor
        from headroom.observable_memory.store import InMemoryObservationStore
        from headroom.proxy.observable_memory_handler import ObservableMemoryHandler, ProxyLLMBridge

        store = InMemoryObservationStore()
        await store.save("test-thread", "* 🔴 (14:30) user prefers Python")

        om_config = ObservableMemoryConfig(enabled=True)
        bridge = ProxyLLMBridge(api_key="sk-test")
        handler = ObservableMemoryHandler(config=om_config, llm=bridge)
        handler._proc = ObservableMemoryProcessor(config=om_config, llm=bridge, store=store)
        proxy.observable_memory_handler = handler

        # Capture the body forwarded to the upstream API
        forwarded_bodies = []

        async def fake_post(url, json=None, headers=None):
            forwarded_bodies.append(json)
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello!"}],
                "model": "claude-opus-4-6",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
            return mock_resp

        proxy.http_client = MagicMock()
        proxy.http_client.post = fake_post

        body = {
            "model": "claude-opus-4-6",
            "system": "You are a helpful assistant.",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
        }
        body_bytes = json.dumps(body).encode()

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/messages",
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body_bytes)).encode()),
                (b"x-api-key", b"sk-test"),
                (b"x-headroom-thread-id", b"test-thread"),
            ],
        }

        async def receive():
            return {"type": "http.request", "body": body_bytes}

        request = Request(scope, receive)
        await proxy.handle_anthropic_messages(request)

        assert len(forwarded_bodies) == 1
        forwarded_system = forwarded_bodies[0].get("system", "")
        assert "<memory>" in forwarded_system
        assert "user prefers Python" in forwarded_system
        assert "You are a helpful assistant." in forwarded_system

    @pytest.mark.asyncio
    async def test_schedule_observe_called_after_response(self, monkeypatch):
        """After the proxy handles a response, schedule_observe is called with the thread_id."""
        import asyncio
        import json
        from unittest.mock import MagicMock

        from starlette.requests import Request

        from headroom.proxy.server import HeadroomProxy, HeadroomProxyConfig

        # Build proxy with OM enabled
        config = HeadroomProxyConfig(
            optimize=False,
            observable_memory_enabled=True,
            observable_memory_observer_model="claude-haiku-4-5-20251001",
        )
        proxy = HeadroomProxy(config)

        # Inject a fake handler using FakeProc so we can capture observe calls
        from headroom.observable_memory import ObservableMemoryConfig
        from headroom.proxy.observable_memory_handler import ObservableMemoryHandler, ProxyLLMBridge

        om_config = ObservableMemoryConfig(enabled=True)
        bridge = ProxyLLMBridge(api_key="sk-test")
        handler = ObservableMemoryHandler(config=om_config, llm=bridge)
        proc = FakeProc(stored=None)
        handler._proc = proc
        proxy.observable_memory_handler = handler

        # Stub the upstream HTTP call
        async def fake_post(url, json=None, headers=None):
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello!"}],
                "model": "claude-opus-4-6",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
            return mock_resp

        proxy.http_client = MagicMock()
        proxy.http_client.post = fake_post

        body = {
            "model": "claude-opus-4-6",
            "system": "You are a helpful assistant.",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
        }
        body_bytes = json.dumps(body).encode()

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/messages",
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body_bytes)).encode()),
                (b"x-api-key", b"sk-test"),
                (b"x-headroom-thread-id", b"test-thread"),
            ],
        }

        async def receive():
            return {"type": "http.request", "body": body_bytes}

        request = Request(scope, receive)
        await proxy.handle_anthropic_messages(request)

        # Give the event loop a tick so the background observe task can run
        await asyncio.sleep(0)

        assert len(proc.observe_calls) == 1
        assert proc.observe_calls[0]["thread_id"] == "test-thread"


class TestOpenAIHandlerObservableMemory:

    @pytest.mark.asyncio
    async def test_observations_injected_into_openai_system_message(self, monkeypatch):
        import json
        from unittest.mock import MagicMock

        from starlette.requests import Request

        from headroom.observable_memory import ObservableMemoryConfig, ObservableMemoryProcessor
        from headroom.observable_memory.store import InMemoryObservationStore
        from headroom.proxy.observable_memory_handler import ObservableMemoryHandler, ProxyLLMBridge
        from headroom.proxy.server import HeadroomProxy, HeadroomProxyConfig

        config = HeadroomProxyConfig(optimize=False, observable_memory_enabled=True)
        proxy = HeadroomProxy(config)

        store = InMemoryObservationStore()
        await store.save("oai-thread", "* 🔴 (14:30) user prefers TypeScript")

        om_config = ObservableMemoryConfig(enabled=True)
        bridge = ProxyLLMBridge(api_key="sk-test")
        handler = ObservableMemoryHandler(config=om_config, llm=bridge)
        handler._proc = ObservableMemoryProcessor(config=om_config, llm=bridge, store=store)
        proxy.observable_memory_handler = handler

        forwarded_bodies = []

        async def fake_post(url, json=None, headers=None):
            forwarded_bodies.append(json)
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "choices": [{"message": {"role": "assistant", "content": "Sure!"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
            return mock_resp

        proxy.http_client = MagicMock()
        proxy.http_client.post = fake_post

        body = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are an assistant."},
                {"role": "user", "content": "Help me write TypeScript"},
            ],
        }
        body_bytes = json.dumps(body).encode()

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body_bytes)).encode()),
                (b"authorization", b"Bearer sk-test"),
                (b"x-headroom-thread-id", b"oai-thread"),
            ],
        }

        async def receive():
            return {"type": "http.request", "body": body_bytes}

        request = Request(scope, receive)
        await proxy.handle_openai_chat(request)

        assert len(forwarded_bodies) == 1
        system_msg = next(
            (m for m in forwarded_bodies[0]["messages"] if m["role"] == "system"), None
        )
        assert system_msg is not None
        assert "<memory>" in system_msg["content"]
        assert "user prefers TypeScript" in system_msg["content"]

    @pytest.mark.asyncio
    async def test_schedule_observe_called_after_openai_response(self):
        import json
        from unittest.mock import MagicMock

        from starlette.requests import Request

        from headroom.observable_memory import ObservableMemoryConfig
        from headroom.proxy.observable_memory_handler import ObservableMemoryHandler, ProxyLLMBridge
        from headroom.proxy.server import HeadroomProxy, HeadroomProxyConfig

        config = HeadroomProxyConfig(optimize=False, observable_memory_enabled=True)
        proxy = HeadroomProxy(config)

        om_config = ObservableMemoryConfig(enabled=True)
        bridge = ProxyLLMBridge(api_key="sk-test")
        handler = ObservableMemoryHandler(config=om_config, llm=bridge)
        proc = FakeProc()
        handler._proc = proc
        proxy.observable_memory_handler = handler

        async def fake_post(url, json=None, headers=None):
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "choices": [{"message": {"role": "assistant", "content": "Sure!"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
            return mock_resp

        proxy.http_client = MagicMock()
        proxy.http_client.post = fake_post

        body = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "Help me write TypeScript"},
            ],
        }
        body_bytes = json.dumps(body).encode()

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body_bytes)).encode()),
                (b"authorization", b"Bearer sk-test"),
                (b"x-headroom-thread-id", b"oai-thread-2"),
            ],
        }

        async def receive():
            return {"type": "http.request", "body": body_bytes}

        request = Request(scope, receive)
        await proxy.handle_openai_chat(request)

        await asyncio.sleep(0)  # let background task run
        assert len(proc.observe_calls) == 1
        assert proc.observe_calls[0]["thread_id"] == "oai-thread-2"
