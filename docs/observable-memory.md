# Observable Memory

**Proactive background compression of message history using Observer/Reflector LLM agents.**

Observable Memory watches your conversation as it grows. Before the context window fills up, it distills what happened into a compact set of observations — facts, preferences, tasks, and outcomes — that are injected into future turns so the model always has the right context.

```
pip install "headroom-ai[observable-memory]"
```

---

## Why Observable Memory?

Standard context management (rolling window, scoring) *drops* old messages when the context overflows. That's reactive: something is lost before you can do anything about it.

Observable Memory is *proactive*:

1. **Observer** — after each batch of new messages, an LLM reads them and writes structured observations (what happened, what matters, what's the current task).
2. **Reflector** — when observations themselves grow too large, another LLM pass consolidates them, weeding out redundancy and drawing connections.
3. **Injection** — before each LLM turn, the stored observations are prepended to the system prompt as `<memory>...</memory>`.

The model always sees a compact, current summary of the conversation rather than raw history.

---

## How It Differs from Headroom Memory

Both systems help LLMs remember. They solve different problems:

| | [Observable Memory](observable-memory.md) | [Hierarchical Memory](memory.md) |
|---|---|---|
| **What it stores** | Observations about the *current conversation* | Extracted facts across *all conversations* |
| **Persistence** | Per-session (SQLite optional) | Long-term, cross-session |
| **Extraction model** | Configurable Observer LLM | Inline (part of the main LLM response) |
| **Use case** | Keep a long agentic session coherent | Remember user preferences across days/weeks |
| **Install extra** | `observable-memory` | `memory` |

They can be used together: Observable Memory compresses *within* a session; Hierarchical Memory persists *across* sessions.

---

## Quick Start

```python
import asyncio
from headroom.observable_memory import ObservableMemoryProcessor, ObservableMemoryConfig

# 1. Implement LLMProvider — one method for LLM calls, one for token counting
class MyLLM:
    async def complete(self, system: str, prompt: str, model: str) -> str:
        # Call your preferred LLM API (OpenAI, Anthropic, Ollama, ...)
        ...

    def count_tokens(self, text: str, model: str) -> int:
        # Use tiktoken or your model's tokenizer
        from headroom.observable_memory import count_string
        return count_string(text, model)

# 2. Create the processor
proc = ObservableMemoryProcessor(
    config=ObservableMemoryConfig(
        enabled=True,
        observer_model="gpt-4o-mini",   # cheap model is fine for observation
        reflector_model="gpt-4o-mini",
        db_path="/tmp/my_session.db",   # omit for in-memory only
    ),
    llm=MyLLM(),
)

# 3. On each turn: observe new messages, then retrieve observations
async def chat_turn(thread_id: str, new_messages: list, context_window: int):
    # Background observation (call after the LLM responds)
    await proc.observe(
        thread_id=thread_id,
        messages=new_messages,
        model="gpt-4o",
        context_window=context_window,
    )

    # Retrieve observations to inject into next turn's system prompt
    observations = await proc.get_observations(thread_id)
    if observations:
        system_prompt = f"{base_system_prompt}\n\n<memory>\n{observations}\n</memory>"
```

---

## How It Works

```
Turn N arrives
      │
      ▼
┌─────────────────────────────────────────────────┐
│  ObservableMemoryProcessor.observe()            │
│                                                 │
│  1. Check message_threshold_ratio               │
│     current_tokens / context_window >= 0.25?   │
│     → skip if under threshold                  │
│                                                 │
│  2. OMWorker.observe()                          │
│     → format messages for Observer             │
│     → call Observer LLM                        │
│     → parse <observations> XML                 │
│     → append to stored observations            │
│                                                 │
│  3. Check observation_threshold_ratio           │
│     obs_tokens / context_window >= 0.35?       │
│     → trigger Reflector if over threshold      │
└─────────────────────────────────────────────────┘
      │
      ▼
Turn N+1 system prompt:
  base_system_prompt
  + <memory>
  +   * 🔴 (14:30) User is debugging a rate-limit bug in the proxy
  +   * 🟡 (14:35) Checked logs, found 429s from upstream at ~14:28
  +   * 🟢 (14:40) User prefers to fix the retry logic before changing limits
  + </memory>
```

### Observer

The Observer reads new messages and writes structured observations using priority emoji:

- 🔴 High priority — errors, blockers, decisions, user assertions
- 🟡 Medium priority — active tasks, important context
- 🟢 Low priority — background facts, preferences

Each observation includes a timestamp so the model can reason about recency.

### Reflector

When observations grow beyond the `observation_threshold_ratio`, the Reflector consolidates them with escalating compression:

| Level | Trigger | Detail target | Strategy |
|-------|---------|---------------|----------|
| 0 | First attempt | Full | Re-organize only |
| 1 | Output ≥ input | 8/10 | Condense older entries |
| 2 | Still too large | 6/10 | Merge aggressively |
| 3 | Still too large | 4/10 | Summarize oldest 50–70% |

The Reflector always keeps recent observations in higher detail than older ones.

### Circuit Breaker

The worker has a built-in circuit breaker that opens after 5 consecutive LLM failures and resets after 60 seconds. This prevents LLM errors from blocking turns.

---

## Configuration

```python
from headroom.observable_memory import ObservableMemoryConfig

config = ObservableMemoryConfig(
    enabled=True,

    # Model selection (falls back to the calling model if None)
    observer_model="gpt-4o-mini",
    reflector_model="gpt-4o-mini",

    # When to observe: only when this fraction of the context window is used
    # Default: 0.25 (observe once messages use 25%+ of the context window)
    message_threshold_ratio=0.25,

    # When to reflect: compress observations when they exceed this fraction
    # Default: 0.35 (run Reflector when observations use 35%+ of context)
    observation_threshold_ratio=0.35,

    # Maximum messages queued for observation (older ones are dropped)
    max_queue_depth=50,

    # SQLite persistence (default ":memory:" = in-process only)
    db_path=":memory:",

    # Minimum context window required to run OM safely
    # Warning logged if context < 2x this value
    # ConfigurationError raised if context < this value
    min_context_window=8_000,

    # Optional instructions appended to both Observer and Reflector system prompts
    instruction="Focus on technical decisions and error resolutions.",
)
```

### Configuration Reference

| Option | Default | Description |
|--------|---------|-------------|
| `enabled` | `False` | Enable/disable observable memory |
| `observer_model` | `None` | LLM for Observer (inherits caller's model if None) |
| `reflector_model` | `None` | LLM for Reflector (inherits observer model if None) |
| `message_threshold_ratio` | `0.25` | Min context usage before observing |
| `observation_threshold_ratio` | `0.35` | Observation size that triggers Reflector |
| `max_queue_depth` | `50` | Max queued messages per thread |
| `db_path` | `":memory:"` | SQLite path, or `":memory:"` for ephemeral |
| `min_context_window` | `8_000` | Minimum safe context window in tokens |
| `instruction` | `None` | Custom instructions for Observer and Reflector |

---

## Processor API

```python
from headroom.observable_memory import (
    ObservableMemoryProcessor,
    ObservableMemoryConfig,
    InMemoryObservationStore,
    SQLiteObservationStore,
    ConfigurationError,
)

proc = ObservableMemoryProcessor(
    config=ObservableMemoryConfig(enabled=True),
    llm=my_llm,
    store=InMemoryObservationStore(),  # or SQLiteObservationStore("/tmp/mem.db")
)

# Validate before use
try:
    proc.validate_context_window(context_window=128_000)
except ConfigurationError as e:
    print(f"Context too small: {e}")

# Observe new messages (call after each LLM response)
await proc.observe(
    thread_id="conv-1",
    messages=new_messages,
    model="gpt-4o",
    context_window=128_000,
    current_token_count=45_000,  # optional; estimated from messages if omitted
    abort=abort_event,           # optional asyncio.Event for cancellation
)

# Retrieve for injection (call before each LLM request)
observations = await proc.get_observations("conv-1")

# Clear when conversation ends
await proc.clear_observations("conv-1")
```

---

## Storage Backends

### InMemoryObservationStore (default)

Observations live in-process and are lost when the process exits. Good for:
- Short-lived sessions
- Testing
- Environments where disk writes are not desired

### SQLiteObservationStore

Persists observations to a SQLite file via `aiosqlite`. Good for:
- Multi-turn sessions that span process restarts
- Logging and debugging
- Sharing observations across multiple processor instances

```python
from headroom.observable_memory import (
    ObservableMemoryProcessor,
    ObservableMemoryConfig,
    SQLiteObservationStore,
)

store = SQLiteObservationStore("/var/data/observations.db")

proc = ObservableMemoryProcessor(
    config=ObservableMemoryConfig(enabled=True, observer_model="gpt-4o-mini"),
    llm=my_llm,
    store=store,
)

# Close the store when done (flushes async writes)
await store.close()
```

### Custom Store

Implement the `ObservationStore` ABC:

```python
from headroom.observable_memory import ObservationStore

class RedisObservationStore(ObservationStore):
    async def load(self, thread_id: str) -> str | None:
        return await redis.get(f"obs:{thread_id}")

    async def save(self, thread_id: str, observations: str) -> None:
        await redis.set(f"obs:{thread_id}", observations)

    async def delete(self, thread_id: str) -> None:
        await redis.delete(f"obs:{thread_id}")
```

---

## Observation Format

Observations follow the format used by Mastra's `@mastra/memory` package (compatible with the same system prompts):

```
Date: Dec 4, 2025
* 🔴 (14:30) User stated: prefers Python for backend services
* 🔴 (14:32) ConfigurationError raised in proxy — missing OPENAI_API_KEY
* 🟡 (14:35) Investigating rate-limit handling in the rolling window transform
  * 🟢 (14:36) Found threshold is set to 0.9, which is too high for this model
* 🟢 (14:40) User prefers verbose logging during debugging sessions
```

Thread attribution is preserved for multi-thread sessions:

```
Date: Dec 4, 2025
* 🔴 (14:30) User prefers TypeScript
<thread id="thread-1">
* 🟡 (14:35) Working on auth feature
</thread>
<thread id="thread-2">
* 🟡 (15:05) Debugging API endpoint
</thread>
```

---

## Isolation

The `headroom.observable_memory` package is **fully self-contained**. It has no imports from the rest of `headroom` core. This means:

- You can use it without installing the full `headroom-ai` package
- It works alongside any LLM client (OpenAI, Anthropic, LangChain, custom)
- Its only runtime dependency beyond stdlib is `aiosqlite` (for SQLite persistence) and `tiktoken` (already a core headroom dependency)

```bash
# Minimal install
pip install "headroom-ai[observable-memory]"

# Or just the extras
pip install tiktoken aiosqlite
```

---

## Integration Example: OpenAI

```python
import asyncio
from openai import AsyncOpenAI
from headroom.observable_memory import (
    ObservableMemoryProcessor,
    ObservableMemoryConfig,
    count_string,
)

openai = AsyncOpenAI()

class OpenAIProvider:
    async def complete(self, system: str, prompt: str, model: str) -> str:
        response = await openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    def count_tokens(self, text: str, model: str) -> int:
        return count_string(text, model)


proc = ObservableMemoryProcessor(
    config=ObservableMemoryConfig(
        enabled=True,
        observer_model="gpt-4o-mini",
        reflector_model="gpt-4o-mini",
    ),
    llm=OpenAIProvider(),
)


async def chat(thread_id: str, messages: list[dict]) -> str:
    # Inject observations into system prompt
    obs = await proc.get_observations(thread_id)
    system = "You are a helpful assistant."
    if obs:
        system += f"\n\n<memory>\n{obs}\n</memory>"

    # Call the LLM
    response = await openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": system}] + messages,
    )
    reply = response.choices[0].message.content or ""

    # Observe the new messages (including the assistant reply) in the background
    all_new = messages + [{"role": "assistant", "content": reply}]
    asyncio.create_task(
        proc.observe(
            thread_id=thread_id,
            messages=all_new,
            model="gpt-4o",
            context_window=128_000,
        )
    )

    return reply
```

---

## Troubleshooting

### Observations are never written

1. Check `config.enabled=True`
2. Check that `current_token_count / context_window >= message_threshold_ratio` — pass `current_token_count` explicitly to bypass the threshold check.
3. Enable debug logging: `logging.getLogger("headroom.observable_memory").setLevel(logging.DEBUG)`

### Observations are not compressing

The Reflector only runs when `obs_tokens / context_window >= observation_threshold_ratio`. Lower `observation_threshold_ratio` to trigger it earlier, or pass a smaller `context_window` value.

### LLM calls fail repeatedly

The circuit breaker opens after 5 consecutive failures. Check your `LLMProvider.complete()` implementation and LLM API credentials. The circuit resets after 60 seconds.

### `ConfigurationError: context window too small`

The default `min_context_window` is 8,000 tokens. For small models, set it lower:

```python
config = ObservableMemoryConfig(min_context_window=2_000)
```

---

## See Also

- [Memory](memory.md) — Hierarchical, persistent memory across sessions
- [Transforms](transforms.md) — Deterministic context management (rolling window, intelligent context)
- [Configuration](configuration.md) — Full `HeadroomConfig` reference
