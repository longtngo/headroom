"""Tests for ObservableMemoryProcessor — observe/reflect orchestration."""
from __future__ import annotations

import pytest

from headroom.observable_memory.processor import ObservableMemoryProcessor
from headroom.observable_memory.store import InMemoryObservationStore
from headroom.observable_memory.types import ObservableMemoryConfig


class FakeLLM:
    """LLM that returns canned observer/reflector responses."""

    def __init__(
        self,
        observer_response: str = "<observations>\n* 🔴 (14:30) test obs\n</observations>",
        reflector_response: str = "<observations>\n* 🔴 (14:30) reflected obs\n</observations>",
    ):
        self.observer_response = observer_response
        self.reflector_response = reflector_response
        self.calls: list[str] = []

    async def complete(self, system: str, prompt: str, model: str) -> str:
        # Detect call type: reflector system prompt contains "reflector"
        if "reflector" in system.lower() or "reflection" in prompt.lower():
            self.calls.append("reflector")
            return self.reflector_response
        self.calls.append("observer")
        return self.observer_response

    def count_tokens(self, text: str, model: str) -> int:
        return len(text.split())


# ── Constructor & validation ──────────────────────────────────────────────────


def test_processor_raises_without_llm_when_enabled():
    config = ObservableMemoryConfig(enabled=True)
    with pytest.raises(ValueError, match="LLMProvider"):
        ObservableMemoryProcessor(config=config, llm=None)


def test_processor_accepts_disabled_without_llm():
    config = ObservableMemoryConfig(enabled=False)
    proc = ObservableMemoryProcessor(config=config, llm=None)
    assert proc is not None


def test_processor_warns_on_small_context(caplog):
    """A context window < 2x min_context_window triggers a warning."""
    import logging

    config = ObservableMemoryConfig(enabled=True, min_context_window=8_000)
    llm = FakeLLM()
    proc = ObservableMemoryProcessor(config=config, llm=llm)

    with caplog.at_level(logging.WARNING, logger="headroom.observable_memory"):
        proc.validate_context_window(context_window=10_000)  # < 2 * 8000

    assert any("small" in r.message.lower() or "context" in r.message.lower()
               for r in caplog.records)


def test_processor_raises_on_insufficient_context():
    """A context window < min_context_window raises ConfigurationError."""
    from headroom.observable_memory.processor import ConfigurationError

    config = ObservableMemoryConfig(enabled=True, min_context_window=8_000)
    llm = FakeLLM()
    proc = ObservableMemoryProcessor(config=config, llm=llm)

    with pytest.raises(ConfigurationError, match="context window"):
        proc.validate_context_window(context_window=5_000)  # < 8000


# ── Observe cycle ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_processor_observe_stores_observations():
    config = ObservableMemoryConfig(enabled=True, observer_model="gpt-4o")
    store = InMemoryObservationStore()
    llm = FakeLLM()
    proc = ObservableMemoryProcessor(config=config, llm=llm, store=store)

    messages = [{"role": "user", "content": "Hello"}]
    await proc.observe("thread-1", messages, model="gpt-4o", context_window=128_000)

    saved = await store.load("thread-1")
    assert saved is not None
    assert "test obs" in saved


@pytest.mark.asyncio
async def test_processor_disabled_is_noop():
    config = ObservableMemoryConfig(enabled=False)
    store = InMemoryObservationStore()
    proc = ObservableMemoryProcessor(config=config, llm=None, store=store)

    messages = [{"role": "user", "content": "Hello"}]
    await proc.observe("thread-1", messages, model="gpt-4o", context_window=128_000)

    assert await store.load("thread-1") is None


@pytest.mark.asyncio
async def test_processor_skips_below_threshold():
    """Processor does not observe when messages are below the threshold ratio."""
    config = ObservableMemoryConfig(
        enabled=True,
        observer_model="gpt-4o",
        message_threshold_ratio=0.50,  # only observe when messages use 50%+ of context
    )
    store = InMemoryObservationStore()
    llm = FakeLLM()
    proc = ObservableMemoryProcessor(config=config, llm=llm, store=store)

    # 5-token message in a 1000-token context = 0.5% << 50% threshold
    messages = [{"role": "user", "content": "hi"}]
    await proc.observe(
        "thread-1",
        messages,
        model="gpt-4o",
        context_window=1000,
        current_token_count=5,
    )

    assert await store.load("thread-1") is None


@pytest.mark.asyncio
async def test_processor_reflect_compresses_observations():
    """When observations exceed threshold, Reflector is called."""
    config = ObservableMemoryConfig(
        enabled=True,
        observer_model="gpt-4o",
        reflector_model="gpt-4o",
        observation_threshold_ratio=0.01,  # very low: always trigger reflection
    )
    store = InMemoryObservationStore()
    llm = FakeLLM()
    proc = ObservableMemoryProcessor(config=config, llm=llm, store=store)

    # Pre-populate with observations that exceed the tiny threshold
    existing = "* 🔴 (14:30) " + ("long observation " * 50)
    await store.save("thread-1", existing)

    messages = [{"role": "user", "content": "hi"}]
    await proc.observe(
        "thread-1",
        messages,
        model="gpt-4o",
        context_window=10_000,
        current_token_count=5_000,  # 50% of context → above message_threshold_ratio=0.25
    )

    saved = await store.load("thread-1")
    # Reflector was triggered: saved observations should be the reflector's output
    assert saved is not None
    assert "reflected obs" in saved
    assert "reflector" in llm.calls


@pytest.mark.asyncio
async def test_processor_get_observations():
    config = ObservableMemoryConfig(enabled=True)
    store = InMemoryObservationStore()
    await store.save("thread-1", "* 🔴 (14:30) User prefers Python")
    proc = ObservableMemoryProcessor(config=config, llm=FakeLLM(), store=store)

    obs = await proc.get_observations("thread-1")
    assert obs == "* 🔴 (14:30) User prefers Python"


@pytest.mark.asyncio
async def test_processor_get_observations_returns_none_when_empty():
    config = ObservableMemoryConfig(enabled=True)
    proc = ObservableMemoryProcessor(
        config=config, llm=FakeLLM(), store=InMemoryObservationStore()
    )
    assert await proc.get_observations("nonexistent") is None


@pytest.mark.asyncio
async def test_processor_clear_observations():
    store = InMemoryObservationStore()
    await store.save("thread-1", "* some obs")
    proc = ObservableMemoryProcessor(
        config=ObservableMemoryConfig(enabled=True), llm=FakeLLM(), store=store
    )
    await proc.clear_observations("thread-1")
    assert await proc.get_observations("thread-1") is None


@pytest.mark.asyncio
async def test_processor_observe_without_token_count_always_proceeds():
    """When current_token_count is omitted, the threshold check is skipped."""
    config = ObservableMemoryConfig(
        enabled=True,
        observer_model="gpt-4o",
        message_threshold_ratio=0.99,  # would normally skip
    )
    store = InMemoryObservationStore()
    llm = FakeLLM()
    proc = ObservableMemoryProcessor(config=config, llm=llm, store=store)

    messages = [{"role": "user", "content": "Hello"}]
    # Omit current_token_count — should always observe regardless of ratio
    await proc.observe("thread-1", messages, model="gpt-4o", context_window=128_000)

    saved = await store.load("thread-1")
    assert saved is not None
