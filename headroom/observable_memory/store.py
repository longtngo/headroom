"""Observation persistence layer for Observable Memory.

Two implementations:
- InMemoryObservationStore: ephemeral, for testing and single-process use
- SQLiteObservationStore: durable, uses aiosqlite for async I/O
"""
from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any


class ObservationStore(ABC):
    """Abstract base class for observation persistence.

    Stores per-thread observations as a single consolidated string
    (the reflector's output). Thread IDs are arbitrary strings supplied
    by the caller (e.g., conversation/session IDs).
    """

    @abstractmethod
    async def load(self, thread_id: str) -> str | None:
        """Load observations for a thread.

        Args:
            thread_id: Unique thread identifier.

        Returns:
            Observations string, or None if no observations exist.
        """
        ...

    @abstractmethod
    async def save(self, thread_id: str, observations: str) -> None:
        """Save (upsert) observations for a thread.

        Args:
            thread_id: Unique thread identifier.
            observations: Consolidated observation string from the Reflector.
        """
        ...

    @abstractmethod
    async def delete(self, thread_id: str) -> None:
        """Delete observations for a thread.

        Args:
            thread_id: Unique thread identifier.
        """
        ...

    async def close(self) -> None:  # noqa: B027 – intentional no-op default, not abstract
        """Release any resources held by the store. No-op by default."""


class InMemoryObservationStore(ObservationStore):
    """Ephemeral in-memory store. Useful for testing and short-lived processes."""

    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    async def load(self, thread_id: str) -> str | None:
        return self._data.get(thread_id)

    async def save(self, thread_id: str, observations: str) -> None:
        self._data[thread_id] = observations

    async def delete(self, thread_id: str) -> None:
        self._data.pop(thread_id, None)


class SQLiteObservationStore(ObservationStore):
    """Durable SQLite-backed store using aiosqlite for async I/O.

    The DB schema is minimal:
        observations(thread_id TEXT PRIMARY KEY, data TEXT NOT NULL, updated_at REAL)

    Args:
        db_path: Path to the SQLite database file, or ':memory:' for in-process.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self._db_path = db_path
        self._conn: Any = None  # aiosqlite.Connection, lazily initialised
        self._lock = asyncio.Lock()

    async def _get_conn(self) -> Any:
        async with self._lock:
            if self._conn is None:
                import aiosqlite  # deferred: only required when observable-memory extra is installed

                self._conn = await aiosqlite.connect(self._db_path)
                await self._conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS observations (
                        thread_id  TEXT PRIMARY KEY,
                        data       TEXT NOT NULL,
                        updated_at REAL NOT NULL
                    )
                    """
                )
                await self._conn.commit()
        return self._conn

    async def load(self, thread_id: str) -> str | None:
        conn = await self._get_conn()
        async with conn.execute(
            "SELECT data FROM observations WHERE thread_id = ?", (thread_id,)
        ) as cursor:
            row = await cursor.fetchone()
        return row[0] if row else None

    async def save(self, thread_id: str, observations: str) -> None:
        conn = await self._get_conn()
        await conn.execute(
            "INSERT OR REPLACE INTO observations (thread_id, data, updated_at) VALUES (?, ?, ?)",
            (thread_id, observations, time.time()),
        )
        await conn.commit()

    async def delete(self, thread_id: str) -> None:
        conn = await self._get_conn()
        await conn.execute(
            "DELETE FROM observations WHERE thread_id = ?", (thread_id,)
        )
        await conn.commit()

    async def close(self) -> None:
        """Close the database connection. Call when done (e.g., in teardown)."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
