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
import hashlib  # noqa: F401 — used by thread ID resolution (Task 2)
import logging
from typing import Any  # noqa: F401 — used by thread ID resolution (Task 2)

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
        """Count tokens using tiktoken. For non-OpenAI models, this is an approximation."""
        from headroom.observable_memory import count_string

        return count_string(text, model)


def _extract_text_content(content: Any) -> str:
    """Extract plain text from message content (str or list of blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text") or "")
        return " ".join(parts)
    return ""


def _derive_thread_id(
    user_id: str | None,
    system_text: str,
    first_user_msg: str,
) -> str | None:
    """Derive a stable 16-char hex thread ID from conversation fingerprint."""
    if not first_user_msg.strip():
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
        if val := lower_headers.get(header, "").strip():
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
