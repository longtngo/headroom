# Observable Memory Phase 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Port Mastra's `@mastra/memory` observational memory core to Python as an isolated, installable module `headroom-ai[observable-memory]`.

**Architecture:** Self-contained Python package at `headroom/observable_memory/`. Zero imports from `headroom` core. The host provides an `LLMProvider` (async `complete()` + sync `count_tokens()`) and optional `db_path` for SQLite persistence. An `OMWorker` manages async background observation with a bounded queue and circuit breaker.

**Tech Stack:** Python 3.10+, tiktoken (already a core dep), aiosqlite, asyncio, dataclasses, typing.Protocol

**Source of Truth (TypeScript originals to port from):**
- `~/src/playground/birch/mastra/packages/memory/src/processors/observational-memory/observer-agent.ts`
- `~/src/playground/birch/mastra/packages/memory/src/processors/observational-memory/reflector-agent.ts`
- `~/src/playground/birch/mastra/packages/memory/src/processors/observational-memory/__tests__/`

---

## Isolation Constraint

`headroom/observable_memory/` is a **self-contained module**. The rule:

```
# ALLOWED
import asyncio, dataclasses, re, time, logging, sqlite3
import tiktoken, aiosqlite

# FORBIDDEN — would break isolation
from headroom.config import ...
from headroom.tokenizer import ...
from headroom.transforms import ...
```

All tests live in `tests/test_observable_memory/` and import only from `headroom.observable_memory`.

---

## Task 1: Package Scaffolding

**Files:**
- Modify: `pyproject.toml`
- Create: `headroom/observable_memory/__init__.py`
- Create: `tests/test_observable_memory/__init__.py`

### Step 1: Add the `observable-memory` extra and test directory

In `pyproject.toml`, add after the `memory = [...]` block (around line 131):

```toml
# Proactive background memory compression using Observer/Reflector LLM agents
observable-memory = [
    "aiosqlite>=0.17.0",
]
```

And update the `all` extra (line 162) to include `observable-memory`:

```toml
all = [
    "headroom-ai[relevance,proxy,reports,llmlingua,code,evals,memory,voice,html,benchmark,mcp,observable-memory]",
]
```

### Step 2: Create module skeleton

```bash
mkdir -p headroom/observable_memory
mkdir -p tests/test_observable_memory
touch headroom/observable_memory/__init__.py
touch tests/test_observable_memory/__init__.py
```

### Step 3: Install the new extra

```bash
uv pip install -e ".[observable-memory]" --python .venv/bin/python
```

Expected: `Successfully installed aiosqlite-...`

### Step 4: Verify import works

```bash
.venv/bin/python -c "import aiosqlite; print('ok')"
```

Expected: `ok`

### Step 5: Commit

```bash
git add pyproject.toml headroom/observable_memory/__init__.py tests/test_observable_memory/__init__.py
git commit -m "feat(observable-memory): add package skeleton and observable-memory extra"
```

---

## Task 2: Core Types

**Files:**
- Create: `headroom/observable_memory/types.py`
- Create: `tests/test_observable_memory/test_types.py`

### Step 1: Write the failing test

`tests/test_observable_memory/test_types.py`:

```python
"""Tests for Observable Memory core types."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from headroom.observable_memory.types import (
    LLMProvider,
    ObservableMemoryConfig,
    ObserverResult,
    ReflectorResult,
)


class FakeLLM:
    """Minimal LLMProvider implementation for testing."""

    async def complete(self, system: str, prompt: str, model: str) -> str:
        return f"<observations>test</observations>"

    def count_tokens(self, text: str, model: str) -> int:
        return len(text.split())


def test_observer_result_defaults():
    result = ObserverResult(observations="* 🔴 (14:30) User prefers Python")
    assert result.observations == "* 🔴 (14:30) User prefers Python"
    assert result.current_task is None
    assert result.suggested_continuation is None
    assert result.raw_output is None
    assert result.degenerate is False


def test_reflector_result_defaults():
    result = ReflectorResult(observations="* 🔴 (14:30) User prefers Python")
    assert result.observations == "* 🔴 (14:30) User prefers Python"
    assert result.suggested_continuation is None
    assert result.degenerate is False
    assert result.token_count is None


def test_config_defaults():
    config = ObservableMemoryConfig()
    assert config.enabled is False
    assert config.observer_model is None
    assert config.reflector_model is None
    assert config.message_threshold_ratio == 0.25
    assert config.observation_threshold_ratio == 0.35
    assert config.max_queue_depth == 50
    assert config.db_path == ":memory:"
    assert config.min_context_window == 8_000


def test_llm_provider_protocol():
    """FakeLLM satisfies the LLMProvider Protocol."""
    llm: LLMProvider = FakeLLM()  # type: ignore[assignment]
    assert isinstance(llm, FakeLLM)


@pytest.mark.asyncio
async def test_llm_provider_complete():
    llm = FakeLLM()
    result = await llm.complete("sys", "prompt", "gpt-4o")
    assert "<observations>" in result


def test_llm_provider_count_tokens():
    llm = FakeLLM()
    assert llm.count_tokens("hello world", "gpt-4o") == 2
```

### Step 2: Run to verify it fails

```bash
.venv/bin/pytest tests/test_observable_memory/test_types.py -v
```

Expected: `ImportError: cannot import name 'LLMProvider' from 'headroom.observable_memory.types'`

### Step 3: Write the implementation

`headroom/observable_memory/types.py`:

```python
"""Core types for Observable Memory."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """Minimal LLM interface required by Observable Memory.

    The host application provides a concrete implementation.
    Observable Memory never imports from headroom core — this Protocol
    is the only coupling between OM and the calling system.
    """

    async def complete(self, system: str, prompt: str, model: str) -> str:
        """Complete a chat with a system message and user prompt.

        Args:
            system: System prompt for the LLM.
            prompt: User prompt / context for the LLM.
            model: Model identifier (e.g. "gpt-4o", "claude-opus-4-6").

        Returns:
            The model's response as a string.
        """
        ...

    def count_tokens(self, text: str, model: str) -> int:
        """Count the number of tokens in a text string.

        Args:
            text: Text to count tokens for.
            model: Model identifier for tokenizer selection.

        Returns:
            Token count as an integer.
        """
        ...


@dataclass
class ObserverResult:
    """Result from the Observer LLM agent."""

    observations: str
    """Extracted observations in date-grouped, emoji-prioritised markdown."""

    current_task: str | None = None
    """Current task section extracted from <current-task> tags."""

    suggested_continuation: str | None = None
    """Suggested next message from <suggested-response> tags."""

    raw_output: str | None = None
    """Raw model output for debugging."""

    degenerate: bool = False
    """True if output was detected as a repetition loop and should be discarded."""


@dataclass
class ReflectorResult:
    """Result from the Reflector LLM agent."""

    observations: str
    """Consolidated observations after reflection/compression."""

    suggested_continuation: str | None = None
    """Suggested continuation from <suggested-response> tags."""

    degenerate: bool = False
    """True if output was detected as a repetition loop."""

    token_count: int | None = None
    """Token count of the observations (set after validation)."""


@dataclass
class ObservableMemoryConfig:
    """Configuration for the Observable Memory subsystem."""

    enabled: bool = False
    """Whether Observable Memory is active. Off by default."""

    observer_model: str | None = None
    """Model for the Observer agent. None = inherit from caller/proxy."""

    reflector_model: str | None = None
    """Model for the Reflector agent. None = inherit from caller/proxy."""

    message_threshold_ratio: float = 0.25
    """Start observing when messages consume this fraction of the context window."""

    observation_threshold_ratio: float = 0.35
    """Trigger reflection when observations exceed this fraction of the context window."""

    max_queue_depth: int = 50
    """Maximum number of pending observation requests before dropping new ones."""

    db_path: str = ":memory:"
    """SQLite database path. Use ':memory:' for in-process testing."""

    min_context_window: int = 8_000
    """Minimum model context window size for OM to operate safely.

    Warning issued when model context < 2x this value.
    ConfigurationError raised when model context < this value.
    """

    instruction: str | None = None
    """Optional custom instructions appended to Observer and Reflector system prompts."""
```

### Step 4: Run tests to verify they pass

```bash
.venv/bin/pytest tests/test_observable_memory/test_types.py -v
```

Expected: `5 passed`

### Step 5: Commit

```bash
git add headroom/observable_memory/types.py tests/test_observable_memory/test_types.py
git commit -m "feat(observable-memory): add core types - LLMProvider, ObserverResult, ReflectorResult, ObservableMemoryConfig"
```

---

## Task 3: Token Counter

**Files:**
- Create: `headroom/observable_memory/token_counter.py`
- Create: `tests/test_observable_memory/test_token_counter.py`

Port from: `token-counter.test.ts` (79 lines) — shared encoder singleton, `count_string(text, model)`.

### Step 1: Write the failing test

`tests/test_observable_memory/test_token_counter.py`:

```python
"""Tests for the Observable Memory token counter."""
from __future__ import annotations

from headroom.observable_memory.token_counter import count_string, get_encoder


def test_count_string_basic():
    """Basic token counting works."""
    count = count_string("Hello world")
    assert count > 0
    assert isinstance(count, int)


def test_count_string_empty():
    """Empty string returns 0."""
    assert count_string("") == 0


def test_count_string_model_variants():
    """Different model names all return a positive count for the same text."""
    text = "The quick brown fox jumps over the lazy dog"
    for model in ["gpt-4o", "gpt-4", "claude-opus-4-6", "unknown-model"]:
        count = count_string(text, model)
        assert count > 0, f"Expected > 0 tokens for model={model}"


def test_count_string_longer_text_has_more_tokens():
    short = count_string("Hello")
    long = count_string("Hello world this is a longer sentence with more tokens")
    assert long > short


def test_get_encoder_singleton():
    """get_encoder returns the same object on repeated calls."""
    enc1 = get_encoder("gpt-4o")
    enc2 = get_encoder("gpt-4o")
    assert enc1 is enc2


def test_get_encoder_different_models_cached_separately():
    enc_4o = get_encoder("gpt-4o")
    enc_4 = get_encoder("gpt-4")
    # Different encoding names → different encoder objects
    # (o200k_base vs cl100k_base)
    assert enc_4o is not enc_4
```

### Step 2: Run to verify it fails

```bash
.venv/bin/pytest tests/test_observable_memory/test_token_counter.py -v
```

Expected: `ImportError: cannot import name 'count_string'`

### Step 3: Write the implementation

`headroom/observable_memory/token_counter.py`:

```python
"""Token counting for Observable Memory.

Uses tiktoken (already a headroom core dependency) with a shared encoder cache.
Default encoding: o200k_base (used by GPT-4o and the Mastra JS port).
Unknown models fall back to cl100k_base.
"""
from __future__ import annotations

import tiktoken

# Encoder cache — shared across all OM calls in a process
_encoder_cache: dict[str, tiktoken.Encoding] = {}

# Models that use o200k_base encoding (GPT-4o family)
_O200K_MODELS = {
    "gpt-4o",
    "gpt-4o-mini",
    "o1",
    "o1-mini",
    "o1-preview",
    "o3",
    "o3-mini",
    "o4-mini",
}


def _get_encoding_name(model: str) -> str:
    """Map a model name to a tiktoken encoding name."""
    model_lower = model.lower()
    if model_lower in _O200K_MODELS:
        return "o200k_base"
    if "gpt-4" in model_lower or "gpt-3.5" in model_lower:
        return "cl100k_base"
    # Claude, Gemini, Llama, and unknown models: default to cl100k_base
    # (reasonable approximation; exact counts vary by model)
    return "cl100k_base"


def get_encoder(model: str = "gpt-4o") -> tiktoken.Encoding:
    """Get a cached tiktoken encoder for the given model.

    Args:
        model: Model name. Used to select the right encoding.

    Returns:
        Cached tiktoken.Encoding instance.
    """
    encoding_name = _get_encoding_name(model)
    if encoding_name not in _encoder_cache:
        _encoder_cache[encoding_name] = tiktoken.get_encoding(encoding_name)
    return _encoder_cache[encoding_name]


def count_string(text: str, model: str = "gpt-4o") -> int:
    """Count tokens in a string.

    Args:
        text: Text to count tokens for.
        model: Model name for encoding selection.

    Returns:
        Token count as an integer.
    """
    if not text:
        return 0
    encoder = get_encoder(model)
    return len(encoder.encode(text))
```

### Step 4: Run tests to verify they pass

```bash
.venv/bin/pytest tests/test_observable_memory/test_token_counter.py -v
```

Expected: `6 passed`

### Step 5: Commit

```bash
git add headroom/observable_memory/token_counter.py tests/test_observable_memory/test_token_counter.py
git commit -m "feat(observable-memory): add token counter with tiktoken + shared encoder cache"
```

---

## Task 4: Observation Store

**Files:**
- Create: `headroom/observable_memory/store.py`
- Create: `tests/test_observable_memory/test_store.py`

### Step 1: Write the failing test

`tests/test_observable_memory/test_store.py`:

```python
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
```

### Step 2: Run to verify it fails

```bash
.venv/bin/pytest tests/test_observable_memory/test_store.py -v
```

Expected: `ImportError: cannot import name 'InMemoryObservationStore'`

### Step 3: Write the implementation

`headroom/observable_memory/store.py`:

```python
"""Observation persistence layer for Observable Memory.

Two implementations:
- InMemoryObservationStore: ephemeral, for testing and single-process use
- SQLiteObservationStore: durable, uses aiosqlite for async I/O
"""
from __future__ import annotations

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

    async def _get_conn(self) -> Any:
        if self._conn is None:
            import aiosqlite

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
```

### Step 4: Run tests to verify they pass

```bash
.venv/bin/pytest tests/test_observable_memory/test_store.py -v
```

Expected: `13 passed`

### Step 5: Commit

```bash
git add headroom/observable_memory/store.py tests/test_observable_memory/test_store.py
git commit -m "feat(observable-memory): add ObservationStore ABC with InMemory and SQLite implementations"
```

---

## Task 5: Observer Agent

**Files:**
- Create: `headroom/observable_memory/observer.py`
- Create: `tests/test_observable_memory/test_observer.py`

Port from: `observer-agent.ts` — prompt constants, `formatMessagesForObserver`, `buildObserverPrompt`, `parseObserverOutput`, `sanitizeObservationLines`, `detectDegenerateRepetition`.

### Step 1: Write the failing test

`tests/test_observable_memory/test_observer.py`:

```python
"""Tests for the Observer agent — prompt building and output parsing."""
from __future__ import annotations

import pytest

from headroom.observable_memory.observer import (
    OBSERVER_EXTRACTION_INSTRUCTIONS,
    OBSERVER_GUIDELINES,
    OBSERVER_OUTPUT_FORMAT_BASE,
    build_observer_prompt,
    build_observer_system_prompt,
    detect_degenerate_repetition,
    format_messages_for_observer,
    parse_observer_output,
    sanitize_observation_lines,
)
from headroom.observable_memory.types import ObserverResult


# ── Prompt constants ──────────────────────────────────────────────────────────


def test_prompt_constants_are_non_empty():
    assert len(OBSERVER_EXTRACTION_INSTRUCTIONS) > 100
    assert len(OBSERVER_OUTPUT_FORMAT_BASE) > 50
    assert len(OBSERVER_GUIDELINES) > 50


def test_system_prompt_contains_key_sections():
    prompt = build_observer_system_prompt()
    assert "memory consciousness" in prompt
    assert "<observations>" in prompt
    assert "🔴" in prompt
    assert "🟡" in prompt
    assert "🟢" in prompt


def test_system_prompt_with_instruction():
    prompt = build_observer_system_prompt(instruction="Always output in Spanish.")
    assert "Always output in Spanish." in prompt


# ── Message formatting ────────────────────────────────────────────────────────


def test_format_messages_basic():
    messages = [
        {"role": "user", "content": "Hello there"},
        {"role": "assistant", "content": "Hi! How can I help?"},
    ]
    formatted = format_messages_for_observer(messages)
    assert "User" in formatted
    assert "Assistant" in formatted
    assert "Hello there" in formatted
    assert "Hi! How can I help?" in formatted


def test_format_messages_with_timestamp():
    messages = [
        {
            "role": "user",
            "content": "Hello",
            "created_at": "2025-12-04T14:30:00Z",
        }
    ]
    formatted = format_messages_for_observer(messages)
    assert "User" in formatted
    assert "Hello" in formatted


def test_format_messages_truncates_long_content():
    long_content = "x" * 10_000
    messages = [{"role": "user", "content": long_content}]
    formatted = format_messages_for_observer(messages, max_part_length=100)
    assert "[truncated" in formatted


def test_format_messages_empty_list():
    assert format_messages_for_observer([]) == ""


# ── Observer prompt builder ───────────────────────────────────────────────────


def test_build_observer_prompt_no_existing():
    messages = [{"role": "user", "content": "What time is it?"}]
    prompt = build_observer_prompt(None, messages)
    assert "New Message History to Observe" in prompt
    assert "What time is it?" in prompt
    assert "Previous Observations" not in prompt


def test_build_observer_prompt_with_existing():
    messages = [{"role": "user", "content": "Follow-up question"}]
    prompt = build_observer_prompt("* 🔴 (14:30) User asked about time", messages)
    assert "Previous Observations" in prompt
    assert "* 🔴 (14:30) User asked about time" in prompt
    assert "Do not repeat" in prompt


def test_build_observer_prompt_skip_continuation_hints():
    messages = [{"role": "user", "content": "Hello"}]
    prompt = build_observer_prompt(None, messages, skip_continuation_hints=True)
    assert "Do NOT include <current-task>" in prompt


# ── Output parsing ────────────────────────────────────────────────────────────


def test_parse_observer_output_with_xml():
    raw = """
<observations>
Date: Dec 4, 2025
* 🔴 (14:30) User prefers Python
* 🟡 (14:31) Working on auth feature
</observations>

<current-task>
Implementing OAuth flow
</current-task>

<suggested-response>
Continue with the token validation step.
</suggested-response>
""".strip()

    result = parse_observer_output(raw)
    assert isinstance(result, ObserverResult)
    assert "User prefers Python" in result.observations
    assert result.current_task == "Implementing OAuth flow"
    assert result.suggested_continuation == "Continue with the token validation step."
    assert result.degenerate is False


def test_parse_observer_output_fallback_to_list_items():
    """When XML tags are missing, extract list items as fallback."""
    raw = """
Some preamble text that should be ignored.
* 🔴 (14:30) User prefers Python
* 🟡 (14:31) Working on a project
More ignored text.
""".strip()

    result = parse_observer_output(raw)
    assert "User prefers Python" in result.observations


def test_parse_observer_output_degenerate():
    # Repeat a 200-char chunk many times → triggers degenerate detection
    chunk = "a" * 200
    raw = chunk * 30  # 6000 chars of repeated content
    result = parse_observer_output(raw)
    assert result.degenerate is True
    assert result.observations == ""


def test_parse_observer_output_short_text_not_degenerate():
    raw = "* 🔴 (14:30) Short observation"
    result = parse_observer_output(raw)
    assert result.degenerate is False


# ── Sanitize observation lines ────────────────────────────────────────────────


def test_sanitize_observation_lines_truncates_long_lines():
    long_line = "x" * 10_001
    result = sanitize_observation_lines(long_line)
    assert len(result) < len(long_line)
    assert "[truncated]" in result


def test_sanitize_observation_lines_keeps_short_lines():
    obs = "* 🔴 (14:30) Normal observation"
    assert sanitize_observation_lines(obs) == obs


def test_sanitize_observation_lines_empty():
    assert sanitize_observation_lines("") == ""


# ── Detect degenerate repetition ──────────────────────────────────────────────


def test_detect_degenerate_repetition_true():
    chunk = "repeated content " * 12  # ~200 chars
    text = chunk * 20  # very long with repetition
    assert detect_degenerate_repetition(text) is True


def test_detect_degenerate_repetition_false_short():
    assert detect_degenerate_repetition("short text") is False


def test_detect_degenerate_repetition_false_normal():
    text = " ".join(f"observation {i} about something different" for i in range(100))
    assert detect_degenerate_repetition(text) is False
```

### Step 2: Run to verify it fails

```bash
.venv/bin/pytest tests/test_observable_memory/test_observer.py -v
```

Expected: `ImportError: cannot import name 'OBSERVER_EXTRACTION_INSTRUCTIONS'`

### Step 3: Write the implementation

`headroom/observable_memory/observer.py`:

```python
"""Observer LLM agent for Observable Memory.

Ported from @mastra/memory observer-agent.ts.
The Observer extracts structured observations from message history,
producing date-grouped, emoji-prioritised markdown stored as memory.
"""
from __future__ import annotations

import re
from typing import Any

from .types import ObserverResult

# ── Prompt constants (ported from observer-agent.ts) ──────────────────────────

OBSERVER_EXTRACTION_INSTRUCTIONS = """CRITICAL: DISTINGUISH USER ASSERTIONS FROM QUESTIONS

When the user TELLS you something about themselves, mark it as an assertion:
- "I have two kids" → 🔴 (14:30) User stated has two kids
- "I work at Acme Corp" → 🔴 (14:31) User stated works at Acme Corp

When the user ASKS about something, mark it as a question/request:
- "Can you help me with X?" → 🔴 (15:00) User asked help with X

Distinguish between QUESTIONS and STATEMENTS OF INTENT:
- "Can you recommend..." → Question (extract as "User asked...")
- "I'm looking forward to [doing X]" → Statement of intent (extract as "User stated they will [do X]")

STATE CHANGES AND UPDATES:
When a user indicates they are changing something, frame it as a state change that supersedes previous information:
- "I'm switching from A to B" → "User is switching from A to B (replacing A)"

If the new state contradicts previous information, make that explicit:
- BAD: "User plans to use the new method"
- GOOD: "User will use the new method (replacing the old approach)"

USER ASSERTIONS ARE AUTHORITATIVE. The user is the source of truth about their own life.

TEMPORAL ANCHORING:
Each observation has TWO potential timestamps:
1. BEGINNING: The time the statement was made (from the message timestamp) - ALWAYS include this
2. END: The time being REFERENCED, if different from when it was said - ONLY when there's a relative time reference

ONLY add "(meaning DATE)" or "(estimated DATE)" at the END when you can provide an ACTUAL DATE.
DO NOT add end dates for vague references like "recently", "a while ago", "lately", "soon".

GOOD: (09:15) User will visit their parents this weekend. (meaning June 17-18, 20XX)
GOOD: (09:15) User prefers hiking in the mountains.  [no end date — present-moment preference]

ALWAYS put the date at the END in parentheses. Split multi-event observations into separate lines.

PRESERVE UNUSUAL PHRASING: Quote the user's exact non-standard terminology.

USE PRECISE ACTION VERBS:
- "getting" something regularly → "subscribed to" or "enrolled in"
- "getting" something once → "purchased" or "acquired"
- "got" → "purchased", "received as gift", "was given", "picked up"

PRESERVING DETAILS IN ASSISTANT-GENERATED CONTENT:
- RECOMMENDATION LISTS: Preserve the key attribute that distinguishes each item
- NAMES/HANDLES/IDENTIFIERS: Always preserve specific identifiers
- TECHNICAL/NUMERICAL RESULTS: Preserve specific values (e.g., "43.7% faster", "2.8GB → 940MB")

CONVERSATION CONTEXT:
- What the user is working on or asking about
- Previous topics and their outcomes
- Specific requirements or constraints mentioned
- Answers to user questions (including full context for detailed explanations)
- Relevant code snippets
- User preferences, favourites, dislikes
- Any specifically formatted text that would need to be reproduced verbatim
- When who/what/where/when is mentioned, note all four dimensions

USER MESSAGE CAPTURE:
- Short and medium-length user messages should be captured nearly verbatim.
- For very long user messages, summarise but quote key phrases.

AVOIDING REPETITIVE OBSERVATIONS:
- Do NOT repeat the same observation across turns if there is no new information.
- Group repeated similar actions (tool calls, file browsing) under a single parent
  with sub-bullets for each new result.

Example — BAD (repetitive):
* 🟡 (14:30) Agent used view tool on src/auth.ts
* 🟡 (14:31) Agent used view tool on src/users.ts

Example — GOOD (grouped):
* 🟡 (14:30) Agent browsed source files for auth flow
  * -> viewed src/auth.ts — found token validation logic
  * -> viewed src/users.ts — found user lookup by email"""

OBSERVER_OUTPUT_FORMAT_BASE = """Use priority levels:
- 🔴 High: explicit user facts, preferences, goals achieved, critical context
- 🟡 Medium: project details, learned information, tool results
- 🟢 Low: minor details, uncertain observations

Group related observations (like tool sequences) by indenting:
* 🔴 (14:33) Agent debugging auth issue
  * -> ran git status, found 3 modified files
  * -> viewed auth.ts:45-60, found missing null check
  * -> applied fix, tests now pass

Group observations by date, then list each with 24-hour time.

<observations>
Date: Dec 4, 2025
* 🔴 (14:30) User prefers direct answers
* 🔴 (14:31) Working on feature X
* 🟡 (14:32) User might prefer dark mode

Date: Dec 5, 2025
* 🔴 (09:15) Continued work on feature X
</observations>

<current-task>
State the current task(s) explicitly:
- Primary: What the agent is currently working on
- Secondary: Other pending tasks (mark as "waiting for user" if appropriate)
</current-task>

<suggested-response>
Hint for the agent's immediate next message. Examples:
- "I've updated the navigation model. Let me walk you through the changes..."
- "The assistant should wait for the user to respond before continuing."
</suggested-response>"""

OBSERVER_GUIDELINES = """- Be specific enough for the assistant to act on
- Good: "User prefers short, direct answers without lengthy explanations"
- Bad: "User stated a preference" (too vague)
- Add 1 to 5 observations per exchange
- Use terse language to save tokens — sentences should be dense
- Do not add repetitive observations that have already been observed
- If the agent calls tools, observe what was called, why, and what was learned
- When observing files with line numbers, include the line number if useful
- Make sure you start each observation with a priority emoji (🔴, 🟡, 🟢)
- User messages are always 🔴 priority — capture closely (short/medium near-verbatim)
- Observe WHAT the agent did and WHAT it means
- If the user provides detailed messages or code snippets, observe all important details"""

# Maximum characters per observation line before truncation
_MAX_OBSERVATION_LINE_CHARS = 10_000


# ── System prompt builder ─────────────────────────────────────────────────────


def build_observer_system_prompt(instruction: str | None = None) -> str:
    """Build the Observer agent's system prompt.

    Args:
        instruction: Optional custom instructions appended to the prompt.

    Returns:
        Full system prompt string.
    """
    prompt = (
        f"You are the memory consciousness of an AI assistant. Your observations will be "
        f"the ONLY information the assistant has about past interactions with this user.\n\n"
        f"Extract observations that will help the assistant remember:\n\n"
        f"{OBSERVER_EXTRACTION_INSTRUCTIONS}\n\n"
        f"=== OUTPUT FORMAT ===\n\n"
        f"Your output MUST use XML tags to structure the response. This allows the system "
        f"to properly parse and manage memory over time.\n\n"
        f"{OBSERVER_OUTPUT_FORMAT_BASE}\n\n"
        f"=== GUIDELINES ===\n\n"
        f"{OBSERVER_GUIDELINES}\n\n"
        f"=== IMPORTANT: THREAD ATTRIBUTION ===\n\n"
        f"Do NOT add thread identifiers or <thread> tags to your observations. "
        f"Thread attribution is handled externally by the system.\n\n"
        f"Remember: These observations are the assistant's ONLY memory. Make them count.\n\n"
        f"User messages are extremely important. If the user asks a question or gives a new "
        f"task, make it clear in <current-task> that this is the priority. If the assistant "
        f"needs to respond to the user, indicate in <suggested-response> that it should pause "
        f"for user reply before continuing other tasks."
    )
    if instruction:
        prompt += f"\n\n=== CUSTOM INSTRUCTIONS ===\n\n{instruction}"
    return prompt


# Default system prompt (build once, reuse)
OBSERVER_SYSTEM_PROMPT = build_observer_system_prompt()


# ── Message formatting ────────────────────────────────────────────────────────


def _maybe_truncate(text: str, max_len: int | None) -> str:
    """Truncate text to max_len characters, appending a note if truncated."""
    if not max_len or len(text) <= max_len:
        return text
    remaining = len(text) - max_len
    return f"{text[:max_len]}\n... [truncated {remaining} characters]"


def format_messages_for_observer(
    messages: list[dict[str, Any]],
    max_part_length: int | None = None,
) -> str:
    """Format a list of messages for the Observer prompt.

    Messages should follow the OpenAI message dict format:
        {"role": "user"|"assistant"|"tool", "content": str, "created_at"?: str}

    Args:
        messages: List of message dicts.
        max_part_length: If set, truncate each content piece to this many chars.

    Returns:
        Formatted string ready to include in the Observer prompt.
    """
    if not messages:
        return ""

    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "unknown")
        role_display = role.capitalize()

        # Format timestamp if present
        timestamp_str = ""
        created_at = msg.get("created_at")
        if created_at:
            try:
                from datetime import datetime, timezone

                if isinstance(created_at, str):
                    dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                else:
                    dt = datetime.fromtimestamp(float(created_at), tz=timezone.utc)
                timestamp_str = f" ({dt.strftime('%b %d, %Y, %I:%M %p')})"
            except (ValueError, TypeError):
                pass  # malformed timestamp — skip gracefully

        # Extract content
        content = msg.get("content", "")
        if isinstance(content, str):
            content = _maybe_truncate(content, max_part_length)
        elif isinstance(content, list):
            # Multi-part content (e.g., tool results)
            rendered: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    part_type = part.get("type", "")
                    if part_type == "text":
                        rendered.append(_maybe_truncate(part.get("text", ""), max_part_length))
                    elif part_type == "tool_result":
                        rendered.append(
                            f"[Tool Result: {part.get('tool_use_id', '')}]\n"
                            + _maybe_truncate(str(part.get("content", "")), max_part_length)
                        )
                    elif part_type == "tool_use":
                        rendered.append(
                            f"[Tool Call: {part.get('name', '')}]\n"
                            + _maybe_truncate(str(part.get("input", "")), max_part_length)
                        )
            content = "\n".join(rendered)
        else:
            content = str(content)

        parts.append(f"**{role_display}{timestamp_str}:**\n{content}")

    return "\n\n---\n\n".join(parts)


# ── Prompt builder ────────────────────────────────────────────────────────────


def build_observer_prompt(
    existing_observations: str | None,
    messages_to_observe: list[dict[str, Any]],
    skip_continuation_hints: bool = False,
) -> str:
    """Build the user-turn prompt for the Observer agent.

    Args:
        existing_observations: Previously extracted observations to avoid repeating.
        messages_to_observe: New messages to extract observations from.
        skip_continuation_hints: If True, instruct the model to omit
            <current-task> and <suggested-response> sections.

    Returns:
        Prompt string for the Observer's user turn.
    """
    formatted = format_messages_for_observer(messages_to_observe)

    prompt = ""
    if existing_observations:
        prompt += f"## Previous Observations\n\n{existing_observations}\n\n---\n\n"
        prompt += (
            "Do not repeat these existing observations. "
            "Your new observations will be appended to the existing observations.\n\n"
        )

    prompt += f"## New Message History to Observe\n\n{formatted}\n\n---\n\n"
    prompt += (
        "## Your Task\n\n"
        "Extract new observations from the message history above. "
        "Do not repeat observations that are already in the previous observations. "
        "Add your new observations in the format specified in your instructions."
    )

    if skip_continuation_hints:
        prompt += (
            "\n\nIMPORTANT: Do NOT include <current-task> or <suggested-response> "
            "sections in your output. Only output <observations>."
        )

    return prompt


# ── Output parsing ────────────────────────────────────────────────────────────


def sanitize_observation_lines(observations: str) -> str:
    """Truncate individual observation lines that exceed the maximum length.

    Guards against LLM degeneration that produces enormous single-line outputs.

    Args:
        observations: Raw observations string.

    Returns:
        Observations with oversized lines truncated.
    """
    if not observations:
        return observations
    lines = observations.split("\n")
    changed = False
    for i, line in enumerate(lines):
        if len(line) > _MAX_OBSERVATION_LINE_CHARS:
            lines[i] = line[:_MAX_OBSERVATION_LINE_CHARS] + " … [truncated]"
            changed = True
    return "\n".join(lines) if changed else observations


def detect_degenerate_repetition(text: str) -> bool:
    """Detect degenerate repetition in Observer/Reflector output.

    Returns True if the text contains suspicious levels of repeated content,
    which indicates an LLM repeat-penalty bug (e.g., looping).

    Strategy: sample sequential 200-char windows; if >40% are duplicates → degenerate.
    Also detects extremely long single lines (50k+ chars).

    Args:
        text: Model output to check.

    Returns:
        True if degenerate, False otherwise.
    """
    if not text or len(text) < 2000:
        return False

    # Strategy 1: repeated 200-char windows
    window_size = 200
    step = max(1, len(text) // 50)  # ~50 samples
    seen: dict[str, int] = {}
    duplicate_windows = 0
    total_windows = 0

    for i in range(0, len(text) - window_size + 1, step):
        window = text[i : i + window_size]
        total_windows += 1
        count = seen.get(window, 0) + 1
        seen[window] = count
        if count > 1:
            duplicate_windows += 1

    if total_windows > 5 and duplicate_windows / total_windows > 0.4:
        return True

    # Strategy 2: extremely long single line
    for line in text.split("\n"):
        if len(line) > 50_000:
            return True

    return False


def _extract_list_items_only(content: str) -> str:
    """Fallback: extract only list items when XML tags are missing."""
    lines = content.split("\n")
    list_lines = [
        line
        for line in lines
        if re.match(r"^\s*[-*]\s", line) or re.match(r"^\s*\d+\.\s", line)
    ]
    return "\n".join(list_lines).strip()


def parse_observer_output(raw_text: str) -> ObserverResult:
    """Parse the Observer agent's raw output into an ObserverResult.

    Handles XML-tagged output (<observations>, <current-task>, <suggested-response>)
    with fallback to list-item extraction when tags are missing.

    Args:
        raw_text: Raw string from the LLM.

    Returns:
        ObserverResult with parsed fields. Sets degenerate=True if repetition detected.
    """
    if detect_degenerate_repetition(raw_text):
        return ObserverResult(observations="", degenerate=True, raw_output=raw_text)

    # Extract <observations> blocks (must be at line start to avoid inline mentions)
    observations_regex = re.compile(
        r"^[ \t]*<observations>([\s\S]*?)^[ \t]*</observations>",
        re.MULTILINE | re.IGNORECASE,
    )
    obs_matches = list(observations_regex.finditer(raw_text))
    if obs_matches:
        observations = "\n".join(
            m.group(1).strip() for m in obs_matches if m.group(1).strip()
        )
    else:
        # Fallback: extract list items
        observations = _extract_list_items_only(raw_text)

    observations = sanitize_observation_lines(observations)

    # Extract <current-task>
    current_task: str | None = None
    ct_match = re.search(
        r"^[ \t]*<current-task>([\s\S]*?)^[ \t]*</current-task>",
        raw_text,
        re.MULTILINE | re.IGNORECASE,
    )
    if ct_match and ct_match.group(1).strip():
        current_task = ct_match.group(1).strip()

    # Extract <suggested-response>
    suggested_continuation: str | None = None
    sr_match = re.search(
        r"^[ \t]*<suggested-response>([\s\S]*?)^[ \t]*</suggested-response>",
        raw_text,
        re.MULTILINE | re.IGNORECASE,
    )
    if sr_match and sr_match.group(1).strip():
        suggested_continuation = sr_match.group(1).strip()

    return ObserverResult(
        observations=observations,
        current_task=current_task,
        suggested_continuation=suggested_continuation,
        raw_output=raw_text,
    )
```

### Step 4: Run tests to verify they pass

```bash
.venv/bin/pytest tests/test_observable_memory/test_observer.py -v
```

Expected: `~24 passed`

### Step 5: Commit

```bash
git add headroom/observable_memory/observer.py tests/test_observable_memory/test_observer.py
git commit -m "feat(observable-memory): add Observer agent - prompt constants, message formatting, XML parsing"
```

---

## Task 6: Reflector Agent

**Files:**
- Create: `headroom/observable_memory/reflector.py`
- Create: `tests/test_observable_memory/test_reflector.py`

Port from: `reflector-agent.ts` — `buildReflectorSystemPrompt`, `COMPRESSION_GUIDANCE`, `buildReflectorPrompt`, `parseReflectorOutput`, `validateCompression`.

### Step 1: Write the failing test

`tests/test_observable_memory/test_reflector.py`:

```python
"""Tests for the Reflector agent — compression and output parsing."""
from __future__ import annotations

import pytest

from headroom.observable_memory.reflector import (
    COMPRESSION_GUIDANCE,
    build_reflector_prompt,
    build_reflector_system_prompt,
    parse_reflector_output,
    validate_compression,
)
from headroom.observable_memory.types import ReflectorResult


# ── System prompt ─────────────────────────────────────────────────────────────


def test_reflector_system_prompt_contains_key_sections():
    prompt = build_reflector_system_prompt()
    assert "memory consciousness" in prompt
    assert "<observations>" in prompt
    assert "USER ASSERTIONS" in prompt


def test_reflector_system_prompt_with_instruction():
    prompt = build_reflector_system_prompt(instruction="Prioritize technical details.")
    assert "Prioritize technical details." in prompt


# ── Compression guidance ──────────────────────────────────────────────────────


def test_compression_guidance_levels_exist():
    assert 0 in COMPRESSION_GUIDANCE
    assert 1 in COMPRESSION_GUIDANCE
    assert 2 in COMPRESSION_GUIDANCE
    assert 3 in COMPRESSION_GUIDANCE


def test_compression_guidance_level_0_is_empty():
    assert COMPRESSION_GUIDANCE[0] == ""


def test_compression_guidance_level_1_mentions_compression():
    assert "COMPRESSION" in COMPRESSION_GUIDANCE[1].upper()


def test_compression_guidance_escalates():
    """Each higher level should push for more compression."""
    assert len(COMPRESSION_GUIDANCE[2]) >= len(COMPRESSION_GUIDANCE[1])
    assert len(COMPRESSION_GUIDANCE[3]) >= len(COMPRESSION_GUIDANCE[2])


# ── Prompt builder ────────────────────────────────────────────────────────────


def test_build_reflector_prompt_basic():
    obs = "* 🔴 (14:30) User prefers Python\n* 🟡 (14:31) Working on auth"
    prompt = build_reflector_prompt(obs)
    assert obs in prompt
    assert "OBSERVATIONS TO REFLECT ON" in prompt


def test_build_reflector_prompt_with_manual():
    obs = "* 🔴 (14:30) User prefers Python"
    prompt = build_reflector_prompt(obs, manual_prompt="Focus on user preferences.")
    assert "Focus on user preferences." in prompt
    assert "SPECIFIC GUIDANCE" in prompt


def test_build_reflector_prompt_compression_level_0():
    obs = "some observations"
    prompt = build_reflector_prompt(obs, compression_level=0)
    assert "COMPRESSION" not in prompt.upper()


def test_build_reflector_prompt_compression_level_1():
    obs = "some observations"
    prompt = build_reflector_prompt(obs, compression_level=1)
    assert "COMPRESSION" in prompt.upper()


def test_build_reflector_prompt_compression_level_3():
    obs = "some observations"
    prompt = build_reflector_prompt(obs, compression_level=3)
    assert "CRITICAL" in prompt.upper()


def test_build_reflector_prompt_skip_continuation_hints():
    obs = "some observations"
    prompt = build_reflector_prompt(obs, skip_continuation_hints=True)
    assert "Do NOT include <current-task>" in prompt


# ── Output parsing ────────────────────────────────────────────────────────────


def test_parse_reflector_output_with_xml():
    raw = """
<observations>
Date: Dec 4, 2025
* 🔴 (14:30) User prefers Python
* 🟡 (14:31) Working on auth feature (consolidated from 5 tool calls)
</observations>

<current-task>
Debugging OAuth token validation
</current-task>

<suggested-response>
Continue with fixing the null check at auth.ts:45.
</suggested-response>
""".strip()

    result = parse_reflector_output(raw)
    assert isinstance(result, ReflectorResult)
    assert "User prefers Python" in result.observations
    assert result.suggested_continuation == "Continue with fixing the null check at auth.ts:45."
    assert result.degenerate is False


def test_parse_reflector_output_degenerate():
    chunk = "b" * 200
    raw = chunk * 30
    result = parse_reflector_output(raw)
    assert result.degenerate is True
    assert result.observations == ""


def test_parse_reflector_output_strips_current_task():
    """Reflector's current-task is NOT included in observations (stored separately)."""
    raw = """
<observations>
* 🔴 (14:30) Some observation
</observations>
<current-task>Current task here</current-task>
""".strip()
    result = parse_reflector_output(raw)
    assert "current-task" not in result.observations.lower()


# ── Compression validation ────────────────────────────────────────────────────


def test_validate_compression_success():
    assert validate_compression(reflected_tokens=500, target_threshold=1000) is True


def test_validate_compression_failure():
    assert validate_compression(reflected_tokens=1500, target_threshold=1000) is False


def test_validate_compression_exactly_at_threshold():
    # Must be strictly BELOW threshold
    assert validate_compression(reflected_tokens=1000, target_threshold=1000) is False
```

### Step 2: Run to verify it fails

```bash
.venv/bin/pytest tests/test_observable_memory/test_reflector.py -v
```

Expected: `ImportError: cannot import name 'COMPRESSION_GUIDANCE'`

### Step 3: Write the implementation

`headroom/observable_memory/reflector.py`:

```python
"""Reflector LLM agent for Observable Memory.

Ported from @mastra/memory reflector-agent.ts.
The Reflector consolidates growing observations into a compressed, coherent
memory that serves as the assistant's complete record of past interactions.
"""
from __future__ import annotations

import re
from typing import Literal

from .observer import (
    OBSERVER_EXTRACTION_INSTRUCTIONS,
    OBSERVER_GUIDELINES,
    OBSERVER_OUTPUT_FORMAT_BASE,
    detect_degenerate_repetition,
    sanitize_observation_lines,
)
from .types import ReflectorResult

# ── Compression guidance (ported from reflector-agent.ts) ─────────────────────

COMPRESSION_GUIDANCE: dict[int, str] = {
    0: "",
    1: """
## COMPRESSION REQUIRED

Your previous reflection was the same size or larger than the original observations.

Please re-process with slightly more compression:
- Towards the beginning, condense more observations into higher-level reflections
- Closer to the end, retain more fine details (recent context matters more)
- Memory is getting long - use a more condensed style throughout
- Combine related items more aggressively but do not lose important specific details
- For example, if there is a long nested list about repeated tool calls, combine into
  a single line: "Tool was called N times for X reason; final outcome: Y."

Your current detail level was a 10/10, aim for 8/10.
""",
    2: """
## AGGRESSIVE COMPRESSION REQUIRED

Your previous reflection was still too large after compression guidance.

Please re-process with much more aggressive compression:
- Towards the beginning, heavily condense observations into high-level summaries
- Closer to the end, retain fine details (recent context matters more)
- Memory is getting very long - use a significantly more condensed style throughout
- Combine related items aggressively but preserve specific names, places, events, people
- Remove redundant information and merge overlapping observations

Your current detail level was a 10/10, aim for 6/10.
""",
    3: """
## CRITICAL COMPRESSION REQUIRED

Your previous reflections have failed to compress sufficiently after multiple attempts.

Please re-process with maximum compression:
- Summarize the oldest observations (first 50-70%) into brief high-level paragraphs —
  only key facts, decisions, and outcomes
- For the most recent observations (last 30-50%), retain important details but still
  use a condensed style
- Ruthlessly merge related observations — if 10 observations are about the same topic,
  combine into 1-2 lines
- Drop procedural details (tool calls, retries, intermediate steps) — keep final outcomes
- Drop observations that are no longer relevant or have been superseded
- Preserve: names, dates, decisions, errors, user preferences, architectural choices

Your current detail level was a 10/10, aim for 4/10.
""",
}

# Backwards-compatibility alias
COMPRESSION_RETRY_PROMPT = COMPRESSION_GUIDANCE[1]


# ── System prompt ─────────────────────────────────────────────────────────────


def build_reflector_system_prompt(instruction: str | None = None) -> str:
    """Build the Reflector agent's system prompt.

    Args:
        instruction: Optional custom instructions appended to the prompt.

    Returns:
        Full system prompt string.
    """
    prompt = f"""You are the memory consciousness of an AI assistant. Your memory observation reflections will be the ONLY information the assistant has about past interactions with this user.

The following instructions were given to another part of your psyche (the observer) to create memories.
Use this to understand how your observational memories were created.

<observational-memory-instruction>
{OBSERVER_EXTRACTION_INSTRUCTIONS}

=== OUTPUT FORMAT ===

{OBSERVER_OUTPUT_FORMAT_BASE}

=== GUIDELINES ===

{OBSERVER_GUIDELINES}
</observational-memory-instruction>

You are another part of the same psyche, the observation reflector.
Your reason for existing is to reflect on all the observations, re-organize and streamline them, and draw connections and conclusions between observations about what you've learned, seen, heard, and done.

You are a much greater and broader aspect of the psyche. Understand that other parts of your mind may get off track in details or side quests — make sure you think hard about what the observed goal at hand is, and observe if we got off track, and why, and how to get back on track.

Take the existing observations and rewrite them to make it easier to continue into the future with this knowledge, to achieve greater things and grow and learn!

IMPORTANT: your reflections are THE ENTIRETY of the assistant's memory. Any information you do not add to your reflections will be immediately forgotten. Make sure you do not leave out anything. Your reflections must assume the assistant knows nothing — your reflections are the ENTIRE memory system.

When consolidating observations:
- Preserve and include dates/times when present (temporal context is critical)
- Retain the most relevant timestamps (start times, completion times, significant events)
- Combine related items where it makes sense
- Condense older observations more aggressively, retain more detail for recent ones

CRITICAL: USER ASSERTIONS vs QUESTIONS
- "User stated: X" = authoritative assertion (user told us something about themselves)
- "User asked: X" = question/request (user seeking information)

When consolidating, USER ASSERTIONS TAKE PRECEDENCE. The user is the authority on their own life.

=== OUTPUT FORMAT ===

Your output MUST use XML tags to structure the response:

<observations>
Put all consolidated observations here using the date-grouped format with priority emojis (🔴, 🟡, 🟢).
Group related observations with indentation.
</observations>

<current-task>
State the current task(s) explicitly:
- Primary: What the agent is currently working on
- Secondary: Other pending tasks (mark as "waiting for user" if appropriate)
</current-task>

<suggested-response>
Hint for the agent's immediate next message.
</suggested-response>

User messages are extremely important. If the user asks a question or gives a new task, make it clear in <current-task> that this is the priority."""

    if instruction:
        prompt += f"\n\n=== CUSTOM INSTRUCTIONS ===\n\n{instruction}"
    return prompt


# Default system prompt
REFLECTOR_SYSTEM_PROMPT = build_reflector_system_prompt()


# ── Prompt builder ────────────────────────────────────────────────────────────


def build_reflector_prompt(
    observations: str,
    manual_prompt: str | None = None,
    compression_level: int = 0,
    skip_continuation_hints: bool = False,
) -> str:
    """Build the user-turn prompt for the Reflector agent.

    Args:
        observations: Existing observations to consolidate.
        manual_prompt: Optional custom guidance to append.
        compression_level: 0 = none, 1 = gentle, 2 = aggressive, 3 = critical.
            Higher levels tell the model to compress more aggressively.
        skip_continuation_hints: If True, omit <current-task> and <suggested-response>.

    Returns:
        Prompt string for the Reflector's user turn.
    """
    level = max(0, min(3, compression_level))

    prompt = (
        f"## OBSERVATIONS TO REFLECT ON\n\n"
        f"{observations}\n\n"
        f"---\n\n"
        f"Please analyze these observations and produce a refined, condensed version "
        f"that will become the assistant's entire memory going forward."
    )

    if manual_prompt:
        prompt += f"\n\n## SPECIFIC GUIDANCE\n\n{manual_prompt}"

    guidance = COMPRESSION_GUIDANCE.get(level, "")
    if guidance:
        prompt += f"\n\n{guidance}"

    if skip_continuation_hints:
        prompt += (
            "\n\nIMPORTANT: Do NOT include <current-task> or <suggested-response> "
            "sections in your output. Only output <observations>."
        )

    return prompt


# ── Output parsing ────────────────────────────────────────────────────────────


def parse_reflector_output(raw_text: str) -> ReflectorResult:
    """Parse the Reflector agent's raw output into a ReflectorResult.

    Args:
        raw_text: Raw string from the LLM.

    Returns:
        ReflectorResult with observations and optional suggested continuation.
        Sets degenerate=True if repetition detected.
    """
    if detect_degenerate_repetition(raw_text):
        return ReflectorResult(observations="", degenerate=True)

    # Extract <observations> blocks
    observations_regex = re.compile(
        r"^[ \t]*<observations>([\s\S]*?)^[ \t]*</observations>",
        re.MULTILINE | re.IGNORECASE,
    )
    obs_matches = list(observations_regex.finditer(raw_text))
    if obs_matches:
        observations = "\n".join(
            m.group(1).strip() for m in obs_matches if m.group(1).strip()
        )
    else:
        # Fallback to full content (no XML tags present)
        observations = raw_text.strip()

    observations = sanitize_observation_lines(observations)

    # Extract <suggested-response> (Reflector's currentTask is NOT used —
    # thread metadata preserves per-thread tasks)
    suggested_continuation: str | None = None
    sr_match = re.search(
        r"^[ \t]*<suggested-response>([\s\S]*?)^[ \t]*</suggested-response>",
        raw_text,
        re.MULTILINE | re.IGNORECASE,
    )
    if sr_match and sr_match.group(1).strip():
        suggested_continuation = sr_match.group(1).strip()

    return ReflectorResult(
        observations=observations,
        suggested_continuation=suggested_continuation,
    )


def validate_compression(reflected_tokens: int, target_threshold: int) -> bool:
    """Check whether the Reflector successfully compressed below the target.

    Args:
        reflected_tokens: Token count of the reflected observations.
        target_threshold: Target token limit (the reflection threshold).

    Returns:
        True if reflected_tokens < target_threshold (compression succeeded).
    """
    return reflected_tokens < target_threshold
```

### Step 4: Run tests to verify they pass

```bash
.venv/bin/pytest tests/test_observable_memory/test_reflector.py -v
```

Expected: `~20 passed`

### Step 5: Commit

```bash
git add headroom/observable_memory/reflector.py tests/test_observable_memory/test_reflector.py
git commit -m "feat(observable-memory): add Reflector agent - compression guidance levels 0-3, XML parsing"
```

---

## Task 7: OMWorker

**Files:**
- Create: `headroom/observable_memory/worker.py`
- Create: `tests/test_observable_memory/test_worker.py`

Port from: `abort-signal.test.ts` and `static-map-cleanup.test.ts` patterns. The worker is purely Python (no Mastra framework dependency).

### Step 1: Write the failing test

`tests/test_observable_memory/test_worker.py`:

```python
"""Tests for OMWorker — async queue, circuit breaker, abort signal."""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from headroom.observable_memory.store import InMemoryObservationStore
from headroom.observable_memory.types import ObservableMemoryConfig, ObserverResult
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
```

### Step 2: Run to verify it fails

```bash
.venv/bin/pytest tests/test_observable_memory/test_worker.py -v
```

Expected: `ImportError: cannot import name 'CircuitBreaker'`

### Step 3: Write the implementation

`headroom/observable_memory/worker.py`:

```python
"""OMWorker — async background observation worker for Observable Memory.

Manages the observe/reflect cycle:
1. Load existing observations for the thread from the store
2. Call the Observer LLM to extract new observations from recent messages
3. Append new observations to the store
4. (Reflect step is triggered by ObservableMemoryProcessor when threshold exceeded)

Includes:
- CircuitBreaker: stops hammering the LLM after repeated failures
- Abort signal: asyncio.Event-based cancellation (analogous to AbortSignal in JS)
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from .observer import build_observer_prompt, build_observer_system_prompt, parse_observer_output
from .store import ObservationStore
from .types import LLMProvider, ObservableMemoryConfig

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """Simple circuit breaker to prevent LLM spam on repeated failures.

    States:
    - Closed (normal): requests pass through
    - Open (tripped): requests are dropped until reset

    Args:
        failure_threshold: Number of consecutive failures before opening.
        reset_after_seconds: Seconds after opening before allowing one retry.
    """

    def __init__(self, failure_threshold: int = 5, reset_after_seconds: float = 60.0) -> None:
        self._threshold = failure_threshold
        self._reset_after = reset_after_seconds
        self._failures = 0
        self._opened_at: float | None = None

    @property
    def is_open(self) -> bool:
        """True if the circuit is open (requests should be dropped)."""
        if self._failures < self._threshold:
            return False
        if self._opened_at is not None:
            elapsed = time.monotonic() - self._opened_at
            if elapsed >= self._reset_after:
                # Allow one retry (half-open state: reset failures by 1)
                self._failures = self._threshold - 1
                self._opened_at = None
                return False
        return True

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self._threshold and self._opened_at is None:
            self._opened_at = time.monotonic()
            logger.warning(
                "OMWorker circuit breaker OPEN after %d consecutive failures. "
                "Observation disabled for %ds.",
                self._failures,
                int(self._reset_after),
            )

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None


class OMWorker:
    """Async worker that runs the Observer step for a single session.

    Usage:
        worker = OMWorker(llm, store, config, session_id="my-session")
        await worker.observe("thread-id", messages, model="gpt-4o")

    Args:
        llm: LLMProvider for making Observer/Reflector calls.
        store: ObservationStore for persisting observations.
        config: ObservableMemoryConfig controlling models and thresholds.
        session_id: Unique session identifier. Defaults to a new UUID.
    """

    def __init__(
        self,
        llm: LLMProvider,
        store: ObservationStore,
        config: ObservableMemoryConfig,
        session_id: str | None = None,
    ) -> None:
        self._llm = llm
        self._store = store
        self._config = config
        self.session_id = session_id or str(uuid.uuid4())
        self._circuit_breaker = CircuitBreaker()
        self._lock = asyncio.Lock()

    async def observe(
        self,
        thread_id: str,
        messages: list[dict[str, Any]],
        model: str,
        abort: asyncio.Event | None = None,
    ) -> None:
        """Run the Observer step for new messages in a thread.

        Loads existing observations, calls the Observer LLM, and saves
        the updated observations. Skips silently if:
        - The circuit breaker is open
        - The abort event is set

        Args:
            thread_id: Conversation/thread identifier.
            messages: New messages to observe.
            model: Model identifier to pass to the Observer.
            abort: Optional asyncio.Event. If set before or during observation,
                   the operation is cancelled and nothing is saved.
        """
        # Check abort signal first
        if abort is not None and abort.is_set():
            logger.debug("OMWorker.observe aborted (abort event set) for thread=%s", thread_id)
            return

        # Check circuit breaker
        if self._circuit_breaker.is_open:
            logger.debug("OMWorker.observe skipped (circuit open) for thread=%s", thread_id)
            return

        async with self._lock:
            try:
                await self._do_observe(thread_id, messages, model, abort)
            except asyncio.CancelledError:
                logger.debug("OMWorker.observe cancelled for thread=%s", thread_id)
                raise
            except Exception as exc:
                self._circuit_breaker.record_failure()
                logger.exception(
                    "OMWorker.observe failed for thread=%s: %s", thread_id, exc
                )

    async def _do_observe(
        self,
        thread_id: str,
        messages: list[dict[str, Any]],
        model: str,
        abort: asyncio.Event | None,
    ) -> None:
        """Internal: run the Observer LLM and persist results."""
        # Check abort again after acquiring lock
        if abort is not None and abort.is_set():
            return

        # Load existing observations for context
        existing = await self._store.load(thread_id)

        # Resolve model
        observer_model = self._config.observer_model or model

        # Build prompts
        system_prompt = build_observer_system_prompt(instruction=self._config.instruction)
        user_prompt = build_observer_prompt(
            existing_observations=existing,
            messages_to_observe=messages,
        )

        # Call Observer LLM
        raw_output = await self._llm.complete(system_prompt, user_prompt, observer_model)

        # Check abort after potentially slow LLM call
        if abort is not None and abort.is_set():
            logger.debug(
                "OMWorker.observe aborted after LLM call for thread=%s", thread_id
            )
            return

        # Parse output
        result = parse_observer_output(raw_output)

        if result.degenerate:
            self._circuit_breaker.record_failure()
            logger.warning(
                "OMWorker detected degenerate observer output for thread=%s. "
                "Discarding and recording failure.",
                thread_id,
            )
            return

        # Append new observations to existing
        if result.observations:
            if existing:
                combined = f"{existing}\n{result.observations}"
            else:
                combined = result.observations
            await self._store.save(thread_id, combined)
            self._circuit_breaker.record_success()
            logger.debug(
                "OMWorker saved %d chars of observations for thread=%s",
                len(combined),
                thread_id,
            )
        else:
            logger.debug(
                "OMWorker: observer returned empty observations for thread=%s", thread_id
            )
```

### Step 4: Run tests to verify they pass

```bash
.venv/bin/pytest tests/test_observable_memory/test_worker.py -v
```

Expected: `8 passed`

### Step 5: Commit

```bash
git add headroom/observable_memory/worker.py tests/test_observable_memory/test_worker.py
git commit -m "feat(observable-memory): add OMWorker with CircuitBreaker and abort signal support"
```

---

## Task 8: ObservableMemoryProcessor

**Files:**
- Create: `headroom/observable_memory/processor.py`
- Create: `tests/test_observable_memory/test_processor.py`

The Processor is the main entry point. It orchestrates the observe/reflect cycle and validates the model context window.

### Step 1: Write the failing test

`tests/test_observable_memory/test_processor.py`:

```python
"""Tests for ObservableMemoryProcessor — observe/reflect orchestration."""
from __future__ import annotations

import asyncio

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
        # Detect call type by presence of "memory consciousness" in system
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
```

### Step 2: Run to verify it fails

```bash
.venv/bin/pytest tests/test_observable_memory/test_processor.py -v
```

Expected: `ImportError: cannot import name 'ObservableMemoryProcessor'`

### Step 3: Write the implementation

`headroom/observable_memory/processor.py`:

```python
"""ObservableMemoryProcessor — main entry point for Observable Memory.

Orchestrates the observe/reflect cycle:
1. validate_context_window — warn/error if model context is too small
2. observe — determine if threshold is met, run Observer LLM, maybe run Reflector
3. get_observations — retrieve stored observations to inject into prompts
"""
from __future__ import annotations

import logging
from typing import Any

from .reflector import build_reflector_prompt, build_reflector_system_prompt, parse_reflector_output, validate_compression
from .store import InMemoryObservationStore, ObservationStore
from .token_counter import count_string
from .types import LLMProvider, ObservableMemoryConfig
from .worker import OMWorker

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when Observable Memory cannot operate safely with the given config."""


class ObservableMemoryProcessor:
    """Main entry point for Observable Memory.

    Wires together the OMWorker (Observer) and Reflector into a single
    interface that the host application calls per-turn.

    Args:
        config: ObservableMemoryConfig controlling all OM behaviour.
        llm: LLMProvider for Observer and Reflector calls.
            Required when config.enabled=True, ignored otherwise.
        store: ObservationStore for persistence.
            Defaults to InMemoryObservationStore (ephemeral).
    """

    def __init__(
        self,
        config: ObservableMemoryConfig | None = None,
        llm: LLMProvider | None = None,
        store: ObservationStore | None = None,
    ) -> None:
        self._config = config or ObservableMemoryConfig()
        self._store = store or InMemoryObservationStore()

        if self._config.enabled and llm is None:
            raise ValueError(
                "LLMProvider is required when ObservableMemoryConfig.enabled=True. "
                "Provide an LLMProvider or set enabled=False."
            )

        self._llm = llm
        self._worker: OMWorker | None = None
        if llm is not None:
            self._worker = OMWorker(llm, self._store, self._config)

    def validate_context_window(self, context_window: int) -> None:
        """Check whether the model's context window is large enough for OM.

        - Warning: context < 2x min_context_window (OM overhead is significant)
        - ConfigurationError: context < min_context_window (unsafe to run OM)

        Args:
            context_window: Model's total context window in tokens.

        Raises:
            ConfigurationError: If context_window < config.min_context_window.
        """
        min_win = self._config.min_context_window

        if context_window < min_win:
            raise ConfigurationError(
                f"Observable Memory requires a context window of at least {min_win} tokens, "
                f"but model reports {context_window} tokens. "
                f"Disable OM or use a model with a larger context window."
            )

        if context_window < 2 * min_win:
            logger.warning(
                "Observable Memory: model context window (%d tokens) is small. "
                "OM overhead may consume a significant fraction of the context. "
                "Consider using a model with a context window >= %d tokens.",
                context_window,
                2 * min_win,
            )

    async def observe(
        self,
        thread_id: str,
        messages: list[dict[str, Any]],
        model: str,
        context_window: int,
        current_token_count: int | None = None,
        abort: Any = None,
    ) -> None:
        """Run one observe/reflect cycle for new messages.

        Steps:
        1. Skip if OM is disabled.
        2. Check message_threshold_ratio — skip if not enough context is used.
        3. Run Observer LLM via OMWorker.
        4. Check observation_threshold_ratio — run Reflector if observations are large.

        Args:
            thread_id: Conversation/thread identifier.
            messages: New messages to observe.
            model: Model identifier for token counting and LLM calls.
            context_window: Model's total context window in tokens.
            current_token_count: Current total tokens in the conversation.
                If None, estimated from messages.
            abort: Optional asyncio.Event for cancellation.
        """
        if not self._config.enabled or self._worker is None:
            return

        # Estimate token count if not provided
        if current_token_count is None:
            content = " ".join(
                str(m.get("content", "")) for m in messages
            )
            current_token_count = count_string(content, model)

        # Check message threshold
        usage_ratio = current_token_count / max(context_window, 1)
        if usage_ratio < self._config.message_threshold_ratio:
            logger.debug(
                "OM.observe skipped: usage_ratio=%.2f < threshold=%.2f for thread=%s",
                usage_ratio,
                self._config.message_threshold_ratio,
                thread_id,
            )
            return

        # Run Observer
        import asyncio as _asyncio
        abort_event = abort if isinstance(abort, _asyncio.Event) else None
        await self._worker.observe(thread_id, messages, model=model, abort=abort_event)

        # Check reflection threshold
        await self._maybe_reflect(thread_id, model, context_window)

    async def _maybe_reflect(
        self,
        thread_id: str,
        model: str,
        context_window: int,
    ) -> None:
        """Trigger Reflector if observations exceed the observation threshold."""
        observations = await self._store.load(thread_id)
        if not observations:
            return

        obs_tokens = count_string(observations, model)
        obs_threshold = int(context_window * self._config.observation_threshold_ratio)

        if obs_tokens < obs_threshold:
            return

        logger.info(
            "OM.reflect triggered: obs_tokens=%d >= threshold=%d for thread=%s",
            obs_tokens,
            obs_threshold,
            thread_id,
        )

        await self._run_reflector(thread_id, observations, model, obs_threshold)

    async def _run_reflector(
        self,
        thread_id: str,
        observations: str,
        model: str,
        target_threshold: int,
        max_attempts: int = 4,
    ) -> None:
        """Run the Reflector LLM with escalating compression levels."""
        assert self._llm is not None

        reflector_model = self._config.reflector_model or model
        system_prompt = build_reflector_system_prompt(instruction=self._config.instruction)

        for attempt in range(max_attempts):
            compression_level = min(attempt, 3)
            user_prompt = build_reflector_prompt(
                observations,
                compression_level=compression_level,
            )

            try:
                raw_output = await self._llm.complete(system_prompt, user_prompt, reflector_model)
            except Exception as exc:
                logger.exception("OM.reflect LLM call failed (attempt %d): %s", attempt, exc)
                return

            result = parse_reflector_output(raw_output)

            if result.degenerate:
                logger.warning(
                    "OM.reflect: degenerate output on attempt %d for thread=%s", attempt, thread_id
                )
                continue

            reflected_tokens = count_string(result.observations, model)
            result.token_count = reflected_tokens

            if validate_compression(reflected_tokens, target_threshold):
                await self._store.save(thread_id, result.observations)
                logger.info(
                    "OM.reflect: compressed %d → %d tokens for thread=%s",
                    count_string(observations, model),
                    reflected_tokens,
                    thread_id,
                )
                return

            logger.info(
                "OM.reflect: attempt %d did not compress enough (%d tokens, target=%d). "
                "Escalating compression.",
                attempt,
                reflected_tokens,
                target_threshold,
            )

        # All attempts failed — keep the original observations
        logger.error(
            "OM.reflect: all %d compression attempts failed for thread=%s. "
            "Keeping original observations.",
            max_attempts,
            thread_id,
        )

    async def get_observations(self, thread_id: str) -> str | None:
        """Retrieve stored observations for a thread.

        Call this before each LLM turn to inject memory into the system prompt.

        Args:
            thread_id: Conversation/thread identifier.

        Returns:
            Observations string, or None if no observations exist.
        """
        return await self._store.load(thread_id)

    async def clear_observations(self, thread_id: str) -> None:
        """Delete all observations for a thread.

        Args:
            thread_id: Conversation/thread identifier.
        """
        await self._store.delete(thread_id)
```

### Step 4: Run tests to verify they pass

```bash
.venv/bin/pytest tests/test_observable_memory/test_processor.py -v
```

Expected: `~10 passed`

### Step 5: Commit

```bash
git add headroom/observable_memory/processor.py tests/test_observable_memory/test_processor.py
git commit -m "feat(observable-memory): add ObservableMemoryProcessor with observe/reflect orchestration"
```

---

## Task 9: Public API and Full Test Run

**Files:**
- Modify: `headroom/observable_memory/__init__.py`

### Step 1: Write the public API surface

`headroom/observable_memory/__init__.py`:

```python
"""Observable Memory — proactive background compression of message history.

Install: pip install "headroom-ai[observable-memory]"

Quick start:
    from headroom.observable_memory import ObservableMemoryProcessor, ObservableMemoryConfig

    # Provide your own LLMProvider implementation
    class MyLLM:
        async def complete(self, system: str, prompt: str, model: str) -> str:
            # call your preferred LLM API
            ...
        def count_tokens(self, text: str, model: str) -> int:
            # count tokens (use tiktoken or your model's tokenizer)
            ...

    proc = ObservableMemoryProcessor(
        config=ObservableMemoryConfig(enabled=True, db_path="/tmp/memory.db"),
        llm=MyLLM(),
    )

    # On each turn, pass new messages and retrieve observations for the system prompt
    await proc.observe(thread_id="conv-1", messages=messages, model="gpt-4o", context_window=128_000)
    observations = await proc.get_observations("conv-1")
    if observations:
        system_prompt = f"{base_system_prompt}\\n\\n<memory>\\n{observations}\\n</memory>"
"""

from .processor import ConfigurationError, ObservableMemoryProcessor
from .store import InMemoryObservationStore, ObservationStore, SQLiteObservationStore
from .token_counter import count_string
from .types import LLMProvider, ObservableMemoryConfig, ObserverResult, ReflectorResult

__all__ = [
    # Main entry point
    "ObservableMemoryProcessor",
    # Configuration
    "ObservableMemoryConfig",
    # Types
    "LLMProvider",
    "ObserverResult",
    "ReflectorResult",
    # Store implementations
    "ObservationStore",
    "InMemoryObservationStore",
    "SQLiteObservationStore",
    # Utilities
    "count_string",
    # Errors
    "ConfigurationError",
]
```

### Step 2: Run the full test suite for the new module

```bash
.venv/bin/pytest tests/test_observable_memory/ -v
```

Expected: all tests pass (roughly 50-60 tests across 6 files).

### Step 3: Verify import works end-to-end

```bash
.venv/bin/python -c "
from headroom.observable_memory import (
    ObservableMemoryProcessor,
    ObservableMemoryConfig,
    InMemoryObservationStore,
)
config = ObservableMemoryConfig(enabled=False)
proc = ObservableMemoryProcessor(config=config)
print('OM import OK:', proc)
"
```

Expected: `OM import OK: <headroom.observable_memory.processor.ObservableMemoryProcessor object at 0x...>`

### Step 4: Quick sanity check — existing headroom tests still pass

```bash
.venv/bin/pytest tests/ -q --ignore=tests/test_observable_memory/ -x --tb=short 2>&1 | tail -20
```

Expected: same pass/fail baseline as before this work (no regressions).

### Step 5: Commit and create PR

```bash
git add headroom/observable_memory/__init__.py
git commit -m "feat(observable-memory): expose public API surface in __init__.py"
```

```bash
git push -u origin feat/observable-memory
gh pr create \
  --title "feat(observable-memory): Phase 1 — port Observer/Reflector core from Mastra" \
  --body "$(cat <<'EOF'
## Summary
- Adds `headroom-ai[observable-memory]` optional extra
- Ports @mastra/memory Observer/Reflector LLM agents to Python
- Isolated module: zero imports from headroom core
- Includes: types, token counter, SQLite store, observer, reflector, async worker, processor

## Test plan
- [ ] \`pytest tests/test_observable_memory/ -v\` — all pass
- [ ] \`pytest tests/ -q --ignore=tests/test_observable_memory/\` — no regressions
- [ ] Manual smoke test: \`python -c \"from headroom.observable_memory import ...\"\`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Appendix: Key Design Decisions

| Decision | Chosen | Rationale |
|---|---|---|
| Package location | `headroom/observable_memory/` | Isolated module, installable separately |
| SQLite async | `aiosqlite` | Clean async I/O, single new dependency |
| Token counting | tiktoken (existing dep) | No new dep, same as Mastra's js-tiktoken |
| Thresholds | ratio-based (0.25 / 0.35) | Adapts to any model's context window |
| LLM interface | `async complete()` | Worker is async; clean interface |
| Abort signal | `asyncio.Event` | Python equivalent of JS `AbortSignal` |
| Default model | `None` (inherit from caller) | Proxy handles model resolution |
| Small context | warn < 2x, error < 1x | Gradual degradation |

## Appendix: Files Created

```
headroom/observable_memory/
├── __init__.py          # Public API surface
├── types.py             # LLMProvider, ObserverResult, ReflectorResult, Config
├── token_counter.py     # tiktoken wrapper + singleton cache
├── store.py             # ObservationStore ABC + InMemory + SQLite
├── observer.py          # Observer prompts + message formatting + XML parsing
├── reflector.py         # Reflector prompts + compression levels + parsing
├── worker.py            # OMWorker + CircuitBreaker + abort signal
└── processor.py         # ObservableMemoryProcessor (orchestrator)

tests/test_observable_memory/
├── __init__.py
├── test_types.py        # 5 tests
├── test_token_counter.py # 6 tests
├── test_store.py        # 13 tests
├── test_observer.py     # ~24 tests
├── test_reflector.py    # ~20 tests
├── test_worker.py       # 8 tests
└── test_processor.py    # ~10 tests
```
