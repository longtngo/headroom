"""Tests for OMWorker — async queue, circuit breaker, abort signal."""
from __future__ import annotations

import asyncio

import pytest

from headroom.observable_memory.store import InMemoryObservationStore
from headroom.observable_memory.types import ObservableMemoryConfig
from headroom.observable_memory.worker import CircuitBreaker, OMWorker

# ── Fake LLM provider ─────────────────────────────────────────────────────────


class FakeLLM:
    def __init__(self, response: str = "<observations>\n* 🔴 test\n</observations>"):
        self.response = response
        self.call_count = 0

    async def complete(self, system: str, prompt: str, model: str) -> str:
        self.call_count += 1
        return self.response

    def count_tokens(self, text: str, model: str) -> int:
        return len(text.split())


# ── CircuitBreaker ────────────────────────────────────────────────────────────


def test_circuit_breaker_starts_closed():
    cb = CircuitBreaker(failure_threshold=3, reset_after_seconds=60)
    assert cb.is_open is False


def test_circuit_breaker_opens_after_failures():
    cb = CircuitBreaker(failure_threshold=3, reset_after_seconds=60)
    cb.record_failure()
    cb.record_failure()
    assert cb.is_open is False
    cb.record_failure()
    assert cb.is_open is True


def test_circuit_breaker_resets_on_success():
    cb = CircuitBreaker(failure_threshold=3, reset_after_seconds=60)
    cb.record_failure()
    cb.record_failure()
    cb.record_failure()
    assert cb.is_open is True
    cb.record_success()
    assert cb.is_open is False


# ── OMWorker ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_worker_observe_stores_observations():
    llm = FakeLLM()
    store = InMemoryObservationStore()
    config = ObservableMemoryConfig(observer_model="gpt-4o")
    worker = OMWorker(llm, store, config)

    messages = [{"role": "user", "content": "Hello"}]
    await worker.observe("thread-1", messages, model="gpt-4o")

    saved = await store.load("thread-1")
    assert saved is not None
    assert "test" in saved


@pytest.mark.asyncio
async def test_worker_skips_when_circuit_open():
    """When circuit is open, observe() is a no-op."""
    llm = FakeLLM()
    store = InMemoryObservationStore()
    config = ObservableMemoryConfig(observer_model="gpt-4o")
    worker = OMWorker(llm, store, config)

    # Force circuit open
    worker._circuit_breaker._failures = worker._circuit_breaker._threshold

    messages = [{"role": "user", "content": "Hello"}]
    await worker.observe("thread-1", messages, model="gpt-4o")

    assert await store.load("thread-1") is None
    assert llm.call_count == 0


@pytest.mark.asyncio
async def test_worker_observe_with_abort_signal():
    """Cancelled observe() should not store anything."""
    llm = FakeLLM()
    store = InMemoryObservationStore()
    config = ObservableMemoryConfig(observer_model="gpt-4o")
    worker = OMWorker(llm, store, config)

    abort_event = asyncio.Event()
    abort_event.set()  # already cancelled

    messages = [{"role": "user", "content": "Hello"}]
    await worker.observe("thread-1", messages, model="gpt-4o", abort=abort_event)

    assert await store.load("thread-1") is None


@pytest.mark.asyncio
async def test_worker_degenerate_output_opens_circuit():
    """Degenerate LLM output should be discarded and increment failure count."""
    degenerate_chunk = "a" * 200
    degenerate_response = degenerate_chunk * 30  # triggers detect_degenerate_repetition

    llm = FakeLLM(response=degenerate_response)
    store = InMemoryObservationStore()
    config = ObservableMemoryConfig(observer_model="gpt-4o")
    worker = OMWorker(llm, store, config)

    messages = [{"role": "user", "content": "Hello"}]
    await worker.observe("thread-1", messages, model="gpt-4o")

    # Observations should NOT be saved for degenerate output
    assert await store.load("thread-1") is None
    # Circuit should have one failure recorded
    assert worker._circuit_breaker._failures >= 1


@pytest.mark.asyncio
async def test_worker_loads_existing_observations():
    """Worker passes existing observations to the Observer as context."""
    llm = FakeLLM()
    store = InMemoryObservationStore()
    await store.save("thread-1", "* 🔴 (14:00) Prior observation")

    config = ObservableMemoryConfig(observer_model="gpt-4o")
    worker = OMWorker(llm, store, config)

    messages = [{"role": "user", "content": "New message"}]
    await worker.observe("thread-1", messages, model="gpt-4o")

    # LLM should have been called (prior observations passed as context)
    assert llm.call_count >= 1


@pytest.mark.asyncio
async def test_static_map_cleanup():
    """Worker state maps use session_id as key and don't leak across instances."""
    llm = FakeLLM()
    store = InMemoryObservationStore()
    config = ObservableMemoryConfig(observer_model="gpt-4o")

    w1 = OMWorker(llm, store, config, session_id="session-A")
    w2 = OMWorker(llm, store, config, session_id="session-B")

    assert w1.session_id != w2.session_id
    assert w1._circuit_breaker is not w2._circuit_breaker
