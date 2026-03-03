# Observable Memory — Proxy Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wire the existing `ObservableMemoryProcessor` into the Headroom proxy so that conversations passing through the proxy automatically receive proactive background compression via Observer/Reflector LLM agents, without requiring any client-side changes.

**Architecture:** A new `ObservableMemoryHandler` class (in `headroom/proxy/observable_memory_handler.py`) mirrors the existing `MemoryHandler` pattern. It is wired into `handle_anthropic_messages` and `handle_openai_chat` as two calls: `inject_observations` (pre-request, injects `<memory>...</memory>` into system prompt) and `schedule_observe` (post-response, fire-and-forget `asyncio.create_task`). Thread ID is resolved from explicit headers (`x-headroom-thread-id`, `Helicone-Session-Id`, `x-portkey-trace-id`, `Mcp-Session-Id`) or derived from a content hash of `user_id + system[:200] + first_user_msg[:300]`.

**Tech Stack:** Python 3.11, `asyncio`, `litellm` (already a core dep), `headroom.observable_memory` (Phase 1), `hashlib`, `aiosqlite` (optional, for SQLite persistence).

**Reference files:**
- Design doc: `docs/plans/2026-03-03-observable-memory-proxy-integration-design.md`
- Phase 1 implementation: `headroom/observable_memory/` (fully self-contained)
- Existing pattern to mirror: `headroom/proxy/memory_handler.py`
- Server entry points: `headroom/proxy/server.py` lines ~302 (config), ~1233 (init), ~1805 (Anthropic pre-request), ~4043 (`handle_openai_chat`)
- Existing proxy tests: `tests/test_proxy_memory_integration.py` (style reference)

---

### Task 1: `ProxyLLMBridge` — LLM provider adapter for proxy context

**Files:**
- Create: `headroom/proxy/observable_memory_handler.py`
- Create: `tests/test_proxy_observable_memory.py`

The `ObservableMemoryProcessor` requires an `LLMProvider` implementor. `ProxyLLMBridge` uses `litellm.acompletion` with the proxy's credentials.

**Step 1: Write the failing test**

```python
# tests/test_proxy_observable_memory.py
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
```

**Step 2: Run to verify it fails**

```bash
.venv/bin/pytest tests/test_proxy_observable_memory.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'headroom.proxy.observable_memory_handler'`

**Step 3: Implement `ProxyLLMBridge`**

Create `headroom/proxy/observable_memory_handler.py`:

```python
"""Observable Memory integration handler for the proxy server.

Provides proactive background compression of conversation history via
Observer/Reflector LLM agents. Mirrors the MemoryHandler pattern.

Usage:
    handler = ObservableMemoryHandler(config, bridge)

    # Pre-request: inject stored observations into system prompt
    await handler.inject_observations(thread_id, body, provider="anthropic")

    # Post-response: fire-and-forget observation of new turn
    handler.schedule_observe(thread_id, messages, model, context_window)
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Headers checked in priority order for thread ID
_THREAD_ID_HEADERS = [
    "x-headroom-thread-id",   # headroom native
    "helicone-session-id",    # Helicone
    "x-portkey-trace-id",     # Portkey
    "mcp-session-id",         # MCP protocol
]


class ProxyLLMBridge:
    """LLMProvider implementation for use inside the proxy.

    Uses litellm so observer/reflector can call any provider
    (Anthropic, OpenAI, etc.) with the proxy's own credentials.
    """

    def __init__(self, api_key: str, api_base: str | None = None) -> None:
        self._api_key = api_key
        self._api_base = api_base

    async def complete(self, system: str, prompt: str, model: str) -> str:
        import litellm

        response = await litellm.acompletion(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            api_key=self._api_key,
            api_base=self._api_base,
        )
        return response.choices[0].message.content or ""

    def count_tokens(self, text: str, model: str) -> int:
        from headroom.observable_memory import count_string

        return count_string(text, model)
```

**Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_proxy_observable_memory.py::TestProxyLLMBridge -v
```

Expected: 4 PASSED

**Step 5: Commit**

```bash
git add headroom/proxy/observable_memory_handler.py tests/test_proxy_observable_memory.py
git commit -m "feat(proxy): add ProxyLLMBridge for observable memory"
```

---

### Task 2: Thread ID resolution

**Files:**
- Modify: `headroom/proxy/observable_memory_handler.py` (append)
- Modify: `tests/test_proxy_observable_memory.py` (append)

**Step 1: Write the failing tests**

Append to `tests/test_proxy_observable_memory.py`:

```python
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
```

**Step 2: Run to verify failure**

```bash
.venv/bin/pytest tests/test_proxy_observable_memory.py::TestResolveThreadId -v 2>&1 | head -20
```

Expected: `ImportError: cannot import name 'resolve_thread_id'`

**Step 3: Implement**

Append to `headroom/proxy/observable_memory_handler.py`:

```python
def _extract_text_content(content: Any) -> str:
    """Extract plain text from message content (str or list of blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return " ".join(parts)
    return ""


def _derive_thread_id(
    user_id: str | None,
    system_text: str,
    first_user_msg: str,
) -> str | None:
    """Derive a stable 16-char hex thread ID from conversation fingerprint."""
    if not first_user_msg:
        return None
    salt = user_id or "anonymous"
    fingerprint = f"{salt}|{system_text[:200]}|{first_user_msg[:300]}"
    return hashlib.sha256(fingerprint.encode()).hexdigest()[:16]


def resolve_thread_id(
    headers: dict[str, str],
    messages: list[dict[str, Any]],
    body: dict[str, Any],
    user_id: str | None = None,
) -> str | None:
    """Resolve a thread ID for Observable Memory scoping.

    Priority:
      1. x-headroom-thread-id header (headroom native)
      2. helicone-session-id header
      3. x-portkey-trace-id header
      4. mcp-session-id header
      5. Content hash: sha256(user_id + system[:200] + first_user_msg[:300])[:16]
      6. None — skip OM for this request

    Args:
        headers: Request headers (any case).
        messages: Full messages array from request body.
        body: Full request body (for Anthropic top-level 'system' field).
        user_id: Value of x-headroom-user-id header, or None.

    Returns:
        A thread ID string, or None if none could be derived.
    """
    # 1–4: explicit headers (case-insensitive lookup)
    lower_headers = {k.lower(): v for k, v in headers.items()}
    for header in _THREAD_ID_HEADERS:
        if val := lower_headers.get(header):
            return val

    # 5: content hash fallback
    # Anthropic: system is body["system"] (top-level); messages[0] is always user
    # OpenAI: system is first message with role=="system"; first user message follows
    system_text = ""
    if "system" in body:
        # Anthropic format
        system_text = _extract_text_content(body["system"])
    else:
        # OpenAI format — find system message in array
        for msg in messages:
            if msg.get("role") == "system":
                system_text = _extract_text_content(msg.get("content", ""))
                break

    first_user_msg = ""
    for msg in messages:
        if msg.get("role") == "user":
            first_user_msg = _extract_text_content(msg.get("content", ""))
            break

    return _derive_thread_id(user_id, system_text, first_user_msg)
```

**Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_proxy_observable_memory.py::TestResolveThreadId -v
```

Expected: 12 PASSED

**Step 5: Commit**

```bash
git add headroom/proxy/observable_memory_handler.py tests/test_proxy_observable_memory.py
git commit -m "feat(proxy): add thread ID resolution for observable memory"
```

---

### Task 3: `ObservableMemoryHandler` — core handler class

**Files:**
- Modify: `headroom/proxy/observable_memory_handler.py` (append)
- Modify: `tests/test_proxy_observable_memory.py` (append)

**Step 1: Write the failing tests**

Append to `tests/test_proxy_observable_memory.py`:

```python
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

    # --- schedule_observe ---

    @pytest.mark.asyncio
    async def test_schedule_observe_creates_task(self):
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
```

**Step 2: Run to verify failure**

```bash
.venv/bin/pytest tests/test_proxy_observable_memory.py::TestObservableMemoryHandler -v 2>&1 | head -20
```

Expected: `ImportError: cannot import name 'ObservableMemoryHandler'`

**Step 3: Implement**

Append to `headroom/proxy/observable_memory_handler.py`:

```python
class ObservableMemoryHandler:
    """Handles Observable Memory injection and observation for the proxy.

    Pre-request: inject stored observations into the system prompt.
    Post-response: fire-and-forget observation of the new turn.

    Mirrors the MemoryHandler pattern in memory_handler.py.
    """

    def __init__(
        self,
        config: Any,  # ObservableMemoryConfig
        llm: ProxyLLMBridge,
    ) -> None:
        from headroom.observable_memory import ObservableMemoryProcessor

        self._proc = ObservableMemoryProcessor(config=config, llm=llm)

    async def inject_observations(
        self,
        thread_id: str,
        body: dict[str, Any],
        provider: str,
    ) -> None:
        """Inject stored observations as <memory>...</memory> into system prompt.

        Mutates body in place. No-op if no observations are stored.

        Args:
            thread_id: The resolved conversation thread ID.
            body: The full request body (mutated in place).
            provider: "anthropic" or "openai".
        """
        obs = await self._proc.get_observations(thread_id)
        if not obs:
            return

        block = f"<memory>\n{obs}\n</memory>"

        if provider == "anthropic":
            existing = body.get("system", "")
            if existing:
                body["system"] = f"{existing}\n\n{block}"
            else:
                body["system"] = block
        else:  # openai
            messages = body.get("messages", [])
            for i, msg in enumerate(messages):
                if msg.get("role") == "system":
                    messages[i] = {**msg, "content": msg["content"] + f"\n\n{block}"}
                    return
            # No system message found — prepend one
            messages.insert(0, {"role": "system", "content": block})

    def schedule_observe(
        self,
        thread_id: str,
        messages: list[dict[str, Any]],
        model: str,
        context_window: int,
    ) -> None:
        """Schedule a fire-and-forget observation of the new turn.

        Returns immediately. If the observer LLM fails, the circuit
        breaker in ObservableMemoryProcessor catches it silently.

        Args:
            thread_id: The resolved conversation thread ID.
            messages: Full message history including the assistant reply.
            model: Model name (used for token counting).
            context_window: Context window size in tokens.
        """
        asyncio.create_task(
            self._proc.observe(
                thread_id=thread_id,
                messages=messages,
                model=model,
                context_window=context_window,
            )
        )
```

**Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_proxy_observable_memory.py -v
```

Expected: all tests PASSED (Tasks 1–3 combined)

**Step 5: Commit**

```bash
git add headroom/proxy/observable_memory_handler.py tests/test_proxy_observable_memory.py
git commit -m "feat(proxy): add ObservableMemoryHandler with inject and schedule_observe"
```

---

### Task 4: Config fields + proxy `__init__` wiring

**Files:**
- Modify: `headroom/proxy/server.py:302` (config fields, after `memory_bridge_export_path`)
- Modify: `headroom/proxy/server.py:1256` (init, after `self.memory_handler = MemoryHandler(...)`)

**Step 1: Write the failing test**

Append to `tests/test_proxy_observable_memory.py`:

```python
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
            observable_memory_db_path="/tmp/om.db",
            observable_memory_message_threshold_ratio=0.3,
            observable_memory_observation_threshold_ratio=0.4,
            observable_memory_instruction="Focus on errors.",
            observable_memory_observer_api_key="sk-other",
        )
        assert config.observable_memory_enabled is True
        assert config.observable_memory_observer_model == "claude-haiku-4-5-20251001"
        assert config.observable_memory_db_path == "/tmp/om.db"
```

**Step 2: Run to verify failure**

```bash
.venv/bin/pytest tests/test_proxy_observable_memory.py::TestProxyConfig -v 2>&1 | head -10
```

Expected: `TypeError: HeadroomProxyConfig.__init__() got an unexpected keyword argument 'observable_memory_enabled'`

**Step 3: Add config fields to `HeadroomProxyConfig`**

In `headroom/proxy/server.py`, find the line (around line 322):
```python
    memory_bridge_export_path: str = ""
```

Insert after it (before `# Compression Hooks`):

```python
    # Observable Memory (proactive background compression via Observer/Reflector agents)
    observable_memory_enabled: bool = False
    observable_memory_observer_model: str | None = None   # defaults to upstream model
    observable_memory_reflector_model: str | None = None  # defaults to observer model
    observable_memory_db_path: str = ":memory:"
    observable_memory_message_threshold_ratio: float = 0.25
    observable_memory_observation_threshold_ratio: float = 0.35
    observable_memory_instruction: str | None = None
    observable_memory_observer_api_key: str | None = None  # for cross-provider observer
```

**Step 4: Run config test**

```bash
.venv/bin/pytest tests/test_proxy_observable_memory.py::TestProxyConfig -v
```

Expected: 2 PASSED

**Step 5: Wire handler into `HeadroomProxy.__init__`**

In `headroom/proxy/server.py`, add the import near the top of the file alongside the other proxy handler imports (around line 87 where `MemoryHandler` is imported):

```python
from headroom.proxy.observable_memory_handler import ObservableMemoryHandler, ProxyLLMBridge
```

Then find (around line 1256):
```python
            self.memory_handler = MemoryHandler(memory_config)
```

Insert after it:

```python
        # Observable Memory Handler (proactive background compression)
        self.observable_memory_handler: ObservableMemoryHandler | None = None
        if config.observable_memory_enabled:
            from headroom.observable_memory import ObservableMemoryConfig

            om_config = ObservableMemoryConfig(
                enabled=True,
                observer_model=config.observable_memory_observer_model,
                reflector_model=config.observable_memory_reflector_model,
                db_path=config.observable_memory_db_path,
                message_threshold_ratio=config.observable_memory_message_threshold_ratio,
                observation_threshold_ratio=config.observable_memory_observation_threshold_ratio,
                instruction=config.observable_memory_instruction,
            )
            om_api_key = config.observable_memory_observer_api_key or getattr(config, "api_key", "")
            om_bridge = ProxyLLMBridge(api_key=om_api_key)
            self.observable_memory_handler = ObservableMemoryHandler(
                config=om_config, llm=om_bridge
            )
```

**Step 6: Run all OM tests**

```bash
.venv/bin/pytest tests/test_proxy_observable_memory.py -v
```

Expected: all PASSED

**Step 7: Run broader test suite to check no regressions**

```bash
.venv/bin/pytest tests/ -q --ignore=tests/test_proxy_observable_memory.py -x 2>&1 | tail -10
```

Expected: existing suite passes (same counts as baseline)

**Step 8: Commit**

```bash
git add headroom/proxy/server.py headroom/proxy/observable_memory_handler.py tests/test_proxy_observable_memory.py
git commit -m "feat(proxy): wire ObservableMemoryHandler into proxy config and init"
```

---

### Task 5: Wire into `handle_anthropic_messages`

**Files:**
- Modify: `headroom/proxy/server.py` (two insertion points in `handle_anthropic_messages`)
- Modify: `tests/test_proxy_observable_memory.py` (append integration test)

The Anthropic handler has two integration points:

1. **Pre-request** (after memory injection, ~line 1847): resolve thread ID + inject observations
2. **Post-response** (after response is fully received, both streaming and non-streaming): schedule_observe

**Step 1: Write the failing integration test**

Append to `tests/test_proxy_observable_memory.py`:

```python
class TestAnthropicHandlerObservableMemory:
    """Integration test: OM observations injected into Anthropic requests."""

    @pytest.mark.asyncio
    async def test_observations_injected_into_system_prompt(self, monkeypatch):
        """When OM has stored observations, they appear in the forwarded request body."""
        import json
        from unittest.mock import AsyncMock, MagicMock, patch

        import httpx
        from starlette.testclient import TestClient

        from headroom.proxy.server import HeadroomProxy, HeadroomProxyConfig

        # Build proxy with OM enabled
        config = HeadroomProxyConfig(
            optimize=False,
            observable_memory_enabled=True,
            observable_memory_observer_model="claude-haiku-4-5-20251001",
        )
        proxy = HeadroomProxy(config)

        # Inject a fake handler whose processor has stored observations
        from headroom.proxy.observable_memory_handler import ObservableMemoryHandler, ProxyLLMBridge
        from headroom.observable_memory import ObservableMemoryConfig, ObservableMemoryProcessor
        from headroom.observable_memory.store import InMemoryObservationStore

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

        from starlette.requests import Request
        import io

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
```

**Step 2: Run to verify failure**

```bash
.venv/bin/pytest tests/test_proxy_observable_memory.py::TestAnthropicHandlerObservableMemory -v 2>&1 | head -20
```

Expected: FAILED — `<memory>` not found in forwarded system

**Step 3: Add pre-request OM injection in `handle_anthropic_messages`**

In `headroom/proxy/server.py`, find the block ending with memory tool injection (around line 1847):
```python
                    if beta_headers:
                        ...
                        headers[key] = value
```

After the entire memory handler block (after line ~1847), insert:

```python
        # Observable Memory: inject stored observations into system prompt
        om_thread_id: str | None = None
        if self.observable_memory_handler:
            from headroom.proxy.observable_memory_handler import resolve_thread_id

            om_thread_id = resolve_thread_id(
                headers=headers,
                messages=messages,
                body=body,
                user_id=headers.get("x-headroom-user-id"),
            )
            if om_thread_id:
                try:
                    await self.observable_memory_handler.inject_observations(
                        om_thread_id, body, provider="anthropic"
                    )
                    logger.debug(f"[{request_id}] ObservableMemory: injected observations for thread={om_thread_id}")
                except Exception as e:
                    logger.warning(f"[{request_id}] ObservableMemory: inject failed: {e}")
```

**Step 4: Run the integration test**

```bash
.venv/bin/pytest tests/test_proxy_observable_memory.py::TestAnthropicHandlerObservableMemory -v
```

Expected: PASSED

**Step 5: Add post-response `schedule_observe` for non-streaming Anthropic**

In `handle_anthropic_messages`, find where the non-streaming response JSON is parsed (around line 2010–2060 where `resp_json` is assembled and returned). After extracting the assistant reply text, add:

```python
                    # Observable Memory: schedule background observation
                    if self.observable_memory_handler and om_thread_id:
                        try:
                            # Extract assistant reply text from response
                            _om_reply = ""
                            for _block in resp_json.get("content", []):
                                if isinstance(_block, dict) and _block.get("type") == "text":
                                    _om_reply += _block.get("text", "")
                            if _om_reply:
                                context_limit = self.anthropic_provider.get_context_limit(model)
                                self.observable_memory_handler.schedule_observe(
                                    thread_id=om_thread_id,
                                    messages=list(messages) + [{"role": "assistant", "content": _om_reply}],
                                    model=model,
                                    context_window=context_limit,
                                )
                        except Exception as e:
                            logger.debug(f"[{request_id}] ObservableMemory: schedule_observe failed: {e}")
```

**Step 6: Run full OM test file**

```bash
.venv/bin/pytest tests/test_proxy_observable_memory.py -v
```

Expected: all PASSED

**Step 7: Regression check**

```bash
.venv/bin/pytest tests/ -q -x --ignore=tests/test_proxy_observable_memory.py 2>&1 | tail -10
```

Expected: same pass/skip counts as before

**Step 8: Commit**

```bash
git add headroom/proxy/server.py tests/test_proxy_observable_memory.py
git commit -m "feat(proxy): wire observable memory into handle_anthropic_messages"
```

---

### Task 6: Wire into `handle_openai_chat`

**Files:**
- Modify: `headroom/proxy/server.py` (two insertion points in `handle_openai_chat`)
- Modify: `tests/test_proxy_observable_memory.py` (append)

`handle_openai_chat` starts at line ~4043. The pattern is identical to Task 5 but `provider="openai"` and the reply is extracted from `choices[0].message.content`.

**Step 1: Write the failing test**

Append to `tests/test_proxy_observable_memory.py`:

```python
class TestOpenAIHandlerObservableMemory:

    @pytest.mark.asyncio
    async def test_observations_injected_into_openai_system_message(self, monkeypatch):
        import json
        from unittest.mock import MagicMock

        from starlette.requests import Request

        from headroom.proxy.server import HeadroomProxy, HeadroomProxyConfig
        from headroom.proxy.observable_memory_handler import ObservableMemoryHandler, ProxyLLMBridge
        from headroom.observable_memory import ObservableMemoryConfig, ObservableMemoryProcessor
        from headroom.observable_memory.store import InMemoryObservationStore

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
```

**Step 2: Run to verify failure**

```bash
.venv/bin/pytest tests/test_proxy_observable_memory.py::TestOpenAIHandlerObservableMemory -v 2>&1 | head -10
```

Expected: FAILED — `<memory>` not found in system message

**Step 3: Add OM wiring to `handle_openai_chat`**

In `headroom/proxy/server.py`, in `handle_openai_chat` (starts ~line 4043), find the block after the CCR/memory tool injection (the section that ends with `body["messages"] = optimized_messages`, around line 4225).

Insert **pre-request OM injection** after that line:

```python
        # Observable Memory: inject stored observations into system prompt
        om_thread_id: str | None = None
        if self.observable_memory_handler:
            from headroom.proxy.observable_memory_handler import resolve_thread_id

            om_thread_id = resolve_thread_id(
                headers=headers,
                messages=messages,
                body=body,
                user_id=headers.get("x-headroom-user-id"),
            )
            if om_thread_id:
                try:
                    await self.observable_memory_handler.inject_observations(
                        om_thread_id, body, provider="openai"
                    )
                    logger.debug(f"[{request_id}] ObservableMemory: injected observations for thread={om_thread_id}")
                except Exception as e:
                    logger.warning(f"[{request_id}] ObservableMemory: inject failed: {e}")
```

Insert **post-response `schedule_observe`** after the non-streaming response JSON is parsed (after `resp_json = response.json()` and usage extraction, around line ~4350):

```python
                    # Observable Memory: schedule background observation
                    if self.observable_memory_handler and om_thread_id:
                        try:
                            _om_reply = ""
                            for _choice in resp_json.get("choices", []):
                                _om_reply += _choice.get("message", {}).get("content", "") or ""
                            if _om_reply:
                                context_limit = self.openai_provider.get_context_limit(model)
                                self.observable_memory_handler.schedule_observe(
                                    thread_id=om_thread_id,
                                    messages=list(messages) + [{"role": "assistant", "content": _om_reply}],
                                    model=model,
                                    context_window=context_limit,
                                )
                        except Exception as e:
                            logger.debug(f"[{request_id}] ObservableMemory: schedule_observe failed: {e}")
```

**Step 4: Run all OM tests**

```bash
.venv/bin/pytest tests/test_proxy_observable_memory.py -v
```

Expected: all PASSED

**Step 5: Full regression check**

```bash
.venv/bin/pytest tests/ -q 2>&1 | tail -15
```

Expected: same baseline pass/skip counts. No new failures.

**Step 6: Commit**

```bash
git add headroom/proxy/server.py tests/test_proxy_observable_memory.py
git commit -m "feat(proxy): wire observable memory into handle_openai_chat"
```

---

## Summary

| Task | Files Changed | Tests Added |
|---|---|---|
| 1 | `observable_memory_handler.py` (new) | `TestProxyLLMBridge` (4 tests) |
| 2 | `observable_memory_handler.py` | `TestResolveThreadId` (12 tests) |
| 3 | `observable_memory_handler.py` | `TestObservableMemoryHandler` (7 tests) |
| 4 | `server.py` (config + init) | `TestProxyConfig` (2 tests) |
| 5 | `server.py` (Anthropic handler) | `TestAnthropicHandlerObservableMemory` (1 test) |
| 6 | `server.py` (OpenAI handler) | `TestOpenAIHandlerObservableMemory` (1 test) |

**Total new tests: ~27**

After all tasks, verify the full suite:

```bash
.venv/bin/pytest tests/ -q 2>&1 | tail -5
```
