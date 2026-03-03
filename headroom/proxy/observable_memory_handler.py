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
