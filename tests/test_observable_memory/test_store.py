"""Tests for ObservationStore implementations."""
from __future__ import annotations

import pytest

from headroom.observable_memory.store import (
    InMemoryObservationStore,
    ObservationStore,
    SQLiteObservationStore,
)


# ── InMemoryObservationStore ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_in_memory_load_returns_none_when_empty():
    store = InMemoryObservationStore()
    assert await store.load("thread-1") is None


@pytest.mark.asyncio
async def test_in_memory_save_and_load():
    store = InMemoryObservationStore()
    await store.save("thread-1", "* 🔴 (14:30) User prefers Python")
    result = await store.load("thread-1")
    assert result == "* 🔴 (14:30) User prefers Python"


@pytest.mark.asyncio
async def test_in_memory_save_overwrites():
    store = InMemoryObservationStore()
    await store.save("thread-1", "old observations")
    await store.save("thread-1", "new observations")
    result = await store.load("thread-1")
    assert result == "new observations"


@pytest.mark.asyncio
async def test_in_memory_delete():
    store = InMemoryObservationStore()
    await store.save("thread-1", "some observations")
    await store.delete("thread-1")
    assert await store.load("thread-1") is None


@pytest.mark.asyncio
async def test_in_memory_delete_nonexistent_is_noop():
    store = InMemoryObservationStore()
    await store.delete("nonexistent")  # should not raise


@pytest.mark.asyncio
async def test_in_memory_multiple_threads():
    store = InMemoryObservationStore()
    await store.save("thread-1", "obs for thread 1")
    await store.save("thread-2", "obs for thread 2")
    assert await store.load("thread-1") == "obs for thread 1"
    assert await store.load("thread-2") == "obs for thread 2"


# ── SQLiteObservationStore ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sqlite_load_returns_none_when_empty():
    store = SQLiteObservationStore(db_path=":memory:")
    assert await store.load("thread-1") is None
    await store.close()


@pytest.mark.asyncio
async def test_sqlite_save_and_load():
    store = SQLiteObservationStore(db_path=":memory:")
    await store.save("thread-1", "* 🔴 (14:30) User prefers Python")
    result = await store.load("thread-1")
    assert result == "* 🔴 (14:30) User prefers Python"
    await store.close()


@pytest.mark.asyncio
async def test_sqlite_save_upserts():
    store = SQLiteObservationStore(db_path=":memory:")
    await store.save("thread-1", "old")
    await store.save("thread-1", "new")
    assert await store.load("thread-1") == "new"
    await store.close()


@pytest.mark.asyncio
async def test_sqlite_delete():
    store = SQLiteObservationStore(db_path=":memory:")
    await store.save("thread-1", "some data")
    await store.delete("thread-1")
    assert await store.load("thread-1") is None
    await store.close()


@pytest.mark.asyncio
async def test_sqlite_multiple_threads():
    store = SQLiteObservationStore(db_path=":memory:")
    await store.save("thread-1", "obs A")
    await store.save("thread-2", "obs B")
    assert await store.load("thread-1") == "obs A"
    assert await store.load("thread-2") == "obs B"
    await store.close()


def test_in_memory_store_is_observation_store():
    """InMemoryObservationStore satisfies the ObservationStore ABC."""
    store = InMemoryObservationStore()
    assert isinstance(store, ObservationStore)
