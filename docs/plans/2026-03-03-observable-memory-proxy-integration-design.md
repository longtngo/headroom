# Observable Memory — Proxy Integration Design

**Date:** 2026-03-03
**Status:** Approved

---

## Goal

Integrate Observable Memory into the Headroom proxy so that long-running conversations passing through the proxy automatically receive proactive background compression — without requiring any changes to the upstream client (Claude Code, Codex CLI, LangChain, etc.).

## Context

Observable Memory (Phase 1) ships a fully self-contained `headroom.observable_memory` package: Observer/Reflector LLM agents, circuit breaker, SQLite/in-memory stores, and an `ObservableMemoryProcessor` orchestrator. Phase 2 wires it into the proxy's request/response pipeline.

The core challenge: the Anthropic Messages API and OpenAI Chat Completions API are stateless — no `thread_id` is transmitted natively. Uninstrumented clients (Claude Code, Codex CLI) do not send conversation identifiers. Headroom must derive or honor a thread ID from available signal before it can scope observations per conversation.

---

## Architecture

New `ObservableMemoryHandler` class (mirrors existing `MemoryHandler` pattern). The proxy stores it as `self.observable_memory_handler` and calls it at two points per request:

```
Request arrives
      │
      ▼
ObservableMemoryHandler.resolve_thread_id(headers, messages, body, user_id)
      │
      ├── explicit header found? → use it
      ├── messages present? → derive content hash
      └── none → skip OM for this request
      │
      ▼  (if thread_id resolved)
ObservableMemoryHandler.inject_observations(thread_id, body, provider)
  → loads stored observations
  → appends <memory>...</memory> to system prompt
      │
      ▼
[upstream LLM call — unchanged]
      │
      ▼
ObservableMemoryHandler.schedule_observe(thread_id, messages + reply, model, ctx_window)
  → asyncio.create_task(proc.observe(...))   ← fire-and-forget, non-blocking
```

**Streaming:** chunks are accumulated (reusing the existing CCR buffer). `schedule_observe` fires after the last chunk.

---

## Thread ID Resolution

Priority order (first match wins):

| Priority | Source | Notes |
|---|---|---|
| 1 | `x-headroom-thread-id` header | Headroom native |
| 2 | `Helicone-Session-Id` header | Helicone interop |
| 3 | `x-portkey-trace-id` header | Portkey interop |
| 4 | `Mcp-Session-Id` header | MCP protocol |
| 5 | Content hash (derived) | Uninstrumented clients |
| 6 | None | Skip OM, passthrough |

**Content hash derivation:**

```python
_THIRD_PARTY_THREAD_HEADERS = [
    "x-headroom-thread-id",
    "helicone-session-id",
    "x-portkey-trace-id",
    "mcp-session-id",
]

def resolve_thread_id(headers, messages, body, user_id) -> str | None:
    lower_headers = {k.lower(): v for k, v in headers.items()}
    for header in _THIRD_PARTY_THREAD_HEADERS:
        if val := lower_headers.get(header):
            return val
    return _derive_thread_id(user_id, system_text, first_user_msg_text)

def _derive_thread_id(user_id, system_text, first_user_msg_text) -> str | None:
    if not first_user_msg_text:
        return None
    fingerprint = f"{user_id or 'anonymous'}|{system_text[:200]}|{first_user_msg_text[:300]}"
    return hashlib.sha256(fingerprint.encode()).hexdigest()[:16]
```

**Per-provider input extraction:**

| Field | Anthropic | OpenAI |
|---|---|---|
| `system_text` | `body.get("system", "")` | first `role=="system"` message content |
| `first_user_msg` | `messages[0]["content"]` | first `role=="user"` message content |

Content may be a list of blocks (multimodal) — extract text parts only.

---

## LLM Bridge (`ProxyLLMBridge`)

Implements the `LLMProvider` protocol using `litellm` (already a core dependency):

```python
class ProxyLLMBridge:
    def __init__(self, api_key: str, api_base: str | None = None) -> None: ...

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

Uses the proxy's own API key by default. An optional `observable_memory_observer_api_key` allows cross-provider observer calls.

---

## Config Changes

Flat fields on `HeadroomProxyConfig` (consistent with existing `memory_*`, `cache_*` convention):

```python
observable_memory_enabled: bool = False
observable_memory_observer_model: str | None = None   # defaults to upstream model
observable_memory_reflector_model: str | None = None  # defaults to observer model
observable_memory_db_path: str = ":memory:"
observable_memory_message_threshold_ratio: float = 0.25
observable_memory_observation_threshold_ratio: float = 0.35
observable_memory_instruction: str | None = None
observable_memory_observer_api_key: str | None = None
```

**Server startup:**

```python
self.observable_memory_handler: ObservableMemoryHandler | None = None
if config.observable_memory_enabled:
    om_config = ObservableMemoryConfig(
        enabled=True,
        observer_model=config.observable_memory_observer_model,
        reflector_model=config.observable_memory_reflector_model,
        db_path=config.observable_memory_db_path,
        message_threshold_ratio=config.observable_memory_message_threshold_ratio,
        observation_threshold_ratio=config.observable_memory_observation_threshold_ratio,
        instruction=config.observable_memory_instruction,
    )
    api_key = config.observable_memory_observer_api_key or config.api_key
    bridge = ProxyLLMBridge(api_key=api_key)
    self.observable_memory_handler = ObservableMemoryHandler(om_config, bridge)
```

**Example YAML:**

```yaml
observable_memory_enabled: true
observable_memory_observer_model: "claude-haiku-4-5-20251001"
observable_memory_db_path: "/var/data/om_sessions.db"
observable_memory_instruction: "Focus on technical decisions and error resolutions."
```

---

## Integration Points in Handlers

Both `handle_anthropic_messages` and `handle_openai_chat` get identical wiring:

**Pre-request (after tag/header extraction):**

```python
om_thread_id: str | None = None
if self.observable_memory_handler:
    om_thread_id = self.observable_memory_handler.resolve_thread_id(
        headers, messages, body,
        user_id=headers.get("x-headroom-user-id"),
    )
    if om_thread_id:
        await self.observable_memory_handler.inject_observations(
            om_thread_id, body, provider="anthropic"  # or "openai"
        )
```

**Post-response (after full response received or stream consumed):**

```python
if self.observable_memory_handler and om_thread_id:
    self.observable_memory_handler.schedule_observe(
        thread_id=om_thread_id,
        messages=messages + [{"role": "assistant", "content": reply_text}],
        model=model,
        context_window=self._get_context_window(model),
    )
```

**`inject_observations` per provider:**

```python
async def inject_observations(self, thread_id, body, provider) -> None:
    obs = await self.proc.get_observations(thread_id)
    if not obs:
        return
    block = f"<memory>\n{obs}\n</memory>"

    if provider == "anthropic":
        existing = body.get("system", "")
        body["system"] = f"{existing}\n\n{block}".strip()
    else:  # openai
        for i, msg in enumerate(body["messages"]):
            if msg["role"] == "system":
                body["messages"][i] = {**msg, "content": msg["content"] + f"\n\n{block}"}
                return
        body["messages"].insert(0, {"role": "system", "content": block})
```

---

## Files Changed

| File | Change |
|---|---|
| `headroom/proxy/observable_memory_handler.py` | **New** — `ObservableMemoryHandler`, `ProxyLLMBridge`, `resolve_thread_id`, `_derive_thread_id` |
| `headroom/proxy/server.py` | Wire handler in `__init__`; inject in `handle_anthropic_messages` + `handle_openai_chat` |
| `tests/test_proxy/test_observable_memory_handler.py` | **New** — unit tests for handler and thread ID resolution |

---

## Out of Scope

- Proxy pipeline (`TransformPipeline`) integration — OM is not a transform
- `HeadroomClient` integration — separate effort
- Google Gemini and OpenAI Responses API handlers — follow-up after core Anthropic/OpenAI handlers
