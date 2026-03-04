# Observable Memory Integration — Design Tradeoff Document

**Date:** 2026-03-02
**Status:** Revised after round 2 review (R1–R5, N1–N3 addressed) — CONDITIONAL GO
**Author:** Engineering
**Target:** headroom v0.4.x

---

## 1. Background and Motivation

### 1.1 The Problem

Headroom's current context management strategies — `RollingWindow` and
`IntelligentContextManager` (ICM) — both operate reactively. They fire when the
token budget is already exhausted and respond by dropping messages. This has two
consequences:

1. **Lost semantics.** Dropped messages are gone from the LLM's working context.
   CCR (Compress-Cache-Retrieve) can recover the raw data on demand, but the LLM
   must know to ask for it — it has no awareness of what it's missing.

2. **No KV cache benefit.** The system prompt and message prefix change every turn
   as messages are dropped and markers are inserted, preventing prompt-cache hits
   across turns.

### 1.2 What Mastra's Observable Memory Adds

Mastra (`@mastra/memory`, v1.x) introduces an **Observational Memory** pattern:
two background agents that proactively compress raw message history into a
structured, dated, priority-tagged observation log. Key properties:

- **Observer Agent** compresses raw messages into structured observations tagged
  with date, priority (🔴 critical / 🟡 moderate / 🟢 informational), and
  relative temporal offset.
- **Reflector Agent** garbage-collects the observation log when it itself grows
  large — combining related observations and removing low-priority ones.
- **Append-only observations** produce a stable prefix that enables **4–10×
  cost reduction via KV prompt caching**.
- **Compression ratios**: 5–40× compression (verified in Mastra source —
  ratios are LLM-guided, not programmatically enforced; tool-call-heavy content
  compresses toward the higher end).

### 1.3 Source of Truth

All design claims are grounded in direct reading of the Mastra source:

| File                                                                                                                                                                                                                                           | Key detail                                                      |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| [`packages/memory/src/processors/observational-memory/observational-memory.ts`](https://github.com/mastra-ai/mastra/blob/2770921eec4d55a36b278d15c3a83f694e462ee5/packages/memory/src/processors/observational-memory/observational-memory.ts) | Threshold defaults, async/sync split, async buffering ops       |
| [`packages/core/src/memory/types.ts`](https://github.com/mastra-ai/mastra/blob/2770921eec4d55a36b278d15c3a83f694e462ee5/packages/core/src/memory/types.ts)                                                                                     | `ObservationConfig`, `ReflectionConfig` interfaces and defaults |
| [Commit `2770921eec`](https://github.com/mastra-ai/mastra/commit/2770921eec4d55a36b278d15c3a83f694e462ee5) (Tyler Barnes, Feb 3 2026)                                                                                                          | `feat: add observational memory (#12599)`                       |
| [Mastra OM docs](https://mastra.ai/docs/memory/observational-memory)                                                                                                                                                                           | Feature overview, quick start, configuration guide              |
| [Mastra OM API reference](https://mastra.ai/reference/memory/observational-memory)                                                                                                                                                             | Complete config tables and examples                             |

---

> **Design iteration history:** Five designs (rated 4–9/10) were explored and
> explicitly rejected before converging on the final approach. See
> [Appendix A](#appendix-a-design-iteration-history).

---

## 2. Final Design

### 2.1 Architecture Overview

```
Agent Request
      │
      ▼
┌─────────────────────────────────────────────────────┐
│  Transform Pipeline                                 │
│                                                     │
│  Step 1 — CacheAligner                              │
│  Step 2 — ContentRouter (SmartCrusher / etc.)       │
│  Step 3 — Context Strategy (pick ONE):              │
│           ├── RollingWindow      (position-based)   │
│           ├── IntelligentContext (semantic-scoring) │
│           └── ObservationalMemory (NEW)             │
└─────────────────────────────────────────────────────┘
      │
      ▼
  LLM Provider
```

`ObservationalMemoryTransform` is a peer strategy, not a wrapper around the
others. Selecting it via config disables ICM and RollingWindow for that session.

---

### 2.2 ObservationalMemoryTransform — Internal Flow

```
Turn start (SYNC)
├── resolve session_key from request (see §2.9)
├── read ObservationStore(session_key)
├── if buffered_obs ≥ activation ratio → swap buffered → active
└── inject active_obs as <observations>…</observations> system prefix
         tagged with metadata {"source": "observational-memory"}
         CacheAligner skips dynamic extraction for this tag (see §2.10)
         (stable across turns → KV cache hits)

After ContentRouter compression:
├── count unobserved message tokens (session token counter)
│
├── unobserved ≥ buffer ratio threshold   → enqueue ObserverAgent to proxy-level
│                                             OM worker (ASYNC, per-session mutex)
│                                             observe(slice) → append to buffered_obs
│
├── unobserved ≥ hard threshold (SYNC)    → ObserverAgent SYNC (blocks pipeline)
│                                             observe(all unobserved)
│                                             → 🔴🟡🟢 dated observations
│                                             → append to buffered_obs
│                                             → seal observed message IDs
│                                             GUARANTEED before LLM call
│
└── emergency terminal cap (see §2.12)
    if post-OM tokens still > model limit → hard drop oldest messages
    emit metric: om_emergency_cap_triggered

Reflector check (after Observer):
├── obs_tokens ≥ reflection buffer ratio  → enqueue ReflectorAgent ASYNC
│                                             (proxy-level OM worker, per-session mutex)
│
└── obs_tokens ≥ reflection hard threshold → ReflectorAgent SYNC (blocks pipeline)
                                              level 0–3 compression
                                              GUARANTEED before LLM call

→ LLM call with stable obs prefix + compressed messages
```

---

### 2.3 Configuration

```python
@dataclass
class ObservableMemoryConfig:
    enabled: bool = False

    # Observer thresholds
    # Ratio fires at (ratio × model_context_window) tokens of unobserved history.
    # Absolute overrides ratio when set.
    message_threshold_ratio: float = 0.25      # 25% of context window
    message_threshold_tokens: int | None = None

    # Reflector thresholds
    observation_threshold_ratio: float = 0.35  # 35% of context window
    observation_threshold_tokens: int | None = None

    # Buffer pre-warm (fraction of resolved hard threshold)
    # Async Observer fires at (buffer_ratio × message_threshold)
    buffer_ratio: float = 0.20

    # Agent models — None means inherit the proxy's configured provider/model.
    # Mastra hardcodes 'google/gemini-2.5-flash' for both; headroom intentionally
    # defers to the proxy's model so OM works out of the box regardless of provider.
    # Override to a cheaper model (e.g. "gpt-4o-mini", "claude-haiku-4-5-20251001")
    # to reduce background observation cost.
    observer_model: str | None = None
    reflector_model: str | None = None
```

**Threshold resolution at runtime:**

```python
def resolve_threshold(ratio, absolute_override, model_context_window):
    if absolute_override is not None:
        return absolute_override
    return int(model_context_window * ratio)
```

**Precedence (low → high):**

```
ratio default (0.25 / 0.35)
  → config file ratio override
    → config file absolute token override
      → CLI flag absolute token override
```

---

### 2.4 CLI Arguments

Pattern follows existing `--memory-*` flags in `headroom proxy`:

```bash
headroom proxy \
  --observable-memory \
  --om-message-threshold-ratio 0.25 \
  --om-message-threshold-tokens 32000 \
  --om-observation-threshold-ratio 0.35 \
  --om-observation-threshold-tokens 45000 \
  --om-buffer-ratio 0.20 \
  --om-observer-model gpt-4o-mini \
  --om-reflector-model gpt-4o-mini
```

When both ratio and absolute token flags are provided, the absolute token value
wins with a logged warning.

---

### 2.5 Storage — ObservationStore

Six fields per session, stored in a dedicated `om_observations` table alongside
the existing `SQLiteMemoryStore` (same SQLite file by default; see Open Question
#5). The partition key is `session_key` — see §2.9 for how it is derived.

| Field                   | Type             | Description                                                |
| ----------------------- | ---------------- | ---------------------------------------------------------- |
| `session_key`           | str              | Partition key — derived per §2.9                           |
| `active_observations`   | text             | Current stable observation block (XML-tagged)              |
| `buffered_observations` | text             | Pending observations not yet activated                     |
| `buffered_reflection`   | text             | Reflector's compressed output, pending swap                |
| `obs_token_count`       | int              | Cached token count of active_observations                  |
| `sealed_message_ids`    | json             | IDs of messages already observed (prevents re-observation) |
| `created_at`            | datetime         | Row creation timestamp                                     |
| `updated_at`            | datetime         | Last write timestamp                                       |
| `expires_at`            | datetime \| null | Optional TTL for retention policy                          |

Operations (all acquire per-session-key mutex before write):

- `get_state(session_key)` — read all fields
- `append_buffered(session_key, observations)` — append to buffered_observations
- `swap_buffered_to_active(session_key)` — atomic swap via SQLite transaction
- `store_buffered_reflection(session_key, reflection)` — store Reflector output
- `seal_message_ids(session_key, ids)` — mark messages as observed
- `purge_session(session_key)` — hard delete all rows for session (for compliance)
- `purge_expired()` — delete rows where `expires_at < now()` (scheduled cleanup)

---

### 2.6 Observer Agent Prompt Contract

Input: raw messages (post-ContentRouter compression)
Output: structured observations in Mastra's actual format (date-grouped, time-stamped bullet points):

```
Date: Feb 28, 2026 (2 days ago)
* 🔴 (14:30) User reported auth token expiry causing production outage. Root cause
  traced to certificate rotation skipping env vars.

Date: Mar 1, 2026 (1 day ago)
* 🟡 (11:22) Implemented fix in token_refresh.py line 142. Tests passing locally,
  awaiting CI confirmation.
* 🟢 (11:45) Discussed refactor of refresh logic into dedicated AuthService class
  for future maintainability.
```

Relative time annotations (e.g., "3 weeks ago") are appended by post-processing,
not generated inline by the LLM.

Priority tagging rules (matching Mastra's definitions):

- 🔴 High priority: user facts, preferences, goals, critical context
- 🟡 Medium priority: project details, learned info, tool results
- 🟢 Low priority: minor details, uncertain observations

---

### 2.7 Reflector Agent Compression Levels

Matches Mastra's implementation — level selected based on how far observations
exceed the reflection threshold:

| Level | Trigger                    | Behaviour                                                                                   |
| ----- | -------------------------- | ------------------------------------------------------------------------------------------- |
| 0     | Just above buffer ratio    | No compression guidance — minor cleanup only                                                |
| 1     | Moderately above threshold | "Aim for 8/10 detail level" — slight reduction                                              |
| 2     | Well above threshold       | "Aim for 6/10 detail level" — moderate reduction                                            |
| 3     | Hard threshold breach      | "Aim for 4/10 detail level" — maximum compression, oldest 50-70% summarised into paragraphs |

---

### 2.8 Threshold Recommendations by Model Class

| Model class                     | Context window | Recommended message threshold | Recommended obs threshold |
| ------------------------------- | -------------- | ----------------------------- | ------------------------- |
| Small (GPT-3.5, Mistral 7B)     | 8k–16k         | absolute: 3,000–4,000         | absolute: 4,000–5,000     |
| Medium (Llama 3.1 8B)           | 32k            | ratio 0.25 → ~8,000           | ratio 0.35 → ~11,200      |
| Standard (GPT-4o, Claude Haiku) | 128k–200k      | ratio 0.25 → 32k–50k          | ratio 0.35 → 45k–70k      |
| Large (Claude Sonnet/Opus)      | 200k           | ratio 0.25 → 50,000           | ratio 0.35 → 70,000       |
| Very large (Gemini 2.5 Flash)   | 1M             | ratio 0.25 → 250,000          | ratio 0.35 → 350,000      |

For small-context models, the absolute override is strongly recommended —
the ratio defaults may produce thresholds too close to the usable window after
accounting for system prompt and output buffer.

---

### 2.9 Session Identity Contract (resolves R1)

The design previously referenced `session_id` without defining its origin. The
proxy today exposes only `x-headroom-user-id` (`server.py:1622`). OM needs a
stable, per-conversation key to partition ObservationStore rows correctly.

**Resolution — `session_key` derivation (in priority order):**

1. **Explicit header (preferred):** `x-headroom-session-id` — caller supplies a
   stable identifier (e.g., a UUID generated at conversation start). This is the
   recommended path for production integrations.

2. **Derived from user + thread:** If no session header is present but
   `x-headroom-user-id` and `x-headroom-thread-id` are both set:
   `session_key = sha256(user_id + ":" + thread_id)[:16]`

3. **Deterministic message fingerprint (fallback, non-default):** If no headers
   are present, derive from multiple entropy sources to reduce collision risk.
   Plain `sha256(first_user_message_content)` is insufficient — common openers
   ("Hello", "Hi there") collide across unrelated conversations. Instead:
   `session_key = sha256(user_id + ":" + model + ":" + timestamp_bucket + ":" + first_user_message_content)[:16]`
   where `timestamp_bucket = unix_timestamp // 3600` (1-hour granularity).
   This is **disabled by default in production** — set `session_fallback = "fingerprint"`
   only in single-user or trusted-network deployments.

4. **Default sentinel:** If the request has no messages, or no fallback strategy
   resolves a key, `session_key = "default"`. This matches existing
   `memory_user_id` fallback behaviour. Multiple users sharing the `"default"`
   key will share observations — acceptable for local/dev use only.

**New CLI flags and config fields:**

```bash
headroom proxy --om-session-header x-headroom-session-id  # default
```

```python
@dataclass
class ObservableMemoryConfig:
    ...
    session_header: str = "x-headroom-session-id"
    # Production default is "thread" — requires x-headroom-thread-id.
    # "fingerprint" is available but must be opted into explicitly.
    session_fallback: Literal["thread", "fingerprint", "default"] = "thread"
```

**Sealed message ID fallback:** When message objects lack an `id` field,
`sealed_message_ids` stores `sha256(role + content)[:16]` fingerprints instead.
This is deterministic and collision-resistant for practical conversation sizes.

---

### 2.10 CacheAligner Interaction and Injection Ordering (resolves R2)

**The conflict:** Design Section 6 originally stated "CacheAligner — still runs
first, unmodified." This is incorrect. `CacheAligner.apply()` iterates all
`role == "system"` messages and calls `_extract_dynamic_content()` which strips
inline date strings (`cache_aligner.py:163–176`). The observation format
contains inline dates (`[2026-03-02 | ref: 2026-02-28 | +2d]`) that CacheAligner
would extract and relocate, breaking observation semantics and destroying prefix
stability.

**Resolution — two-part fix:**

**Part A — Tag-based exemption in CacheAligner (minimal change):**
The observation block is injected as a system message with metadata:
`{"source": "observational-memory"}`. CacheAligner is updated to skip dynamic
content extraction for any system message carrying this tag:

```python
# cache_aligner.py — apply(), system message loop
for msg in result_messages:
    if msg.get("role") == "system":
        if msg.get("metadata", {}).get("source") == "observational-memory":
            continue  # preserve observation block intact
        ...existing extraction logic...
```

This is the only change to CacheAligner. All other system message processing is
unchanged.

**Part B — Injection point moves to after CacheAligner:**
The active observation block is injected between Step 1 (CacheAligner) and
Step 2 (ContentRouter), not before the pipeline. This eliminates the dependency
on CacheAligner seeing the tag at all and is safer long-term.

Updated pipeline order:

```
Step 1 — CacheAligner           (unchanged, processes original system messages)
Step 1.5 — OM prefix injection  (NEW: insert active_obs system message with tag)
Step 2 — ContentRouter          (unchanged)
Step 3 — ObservationalMemory    (threshold checking, Observer/Reflector dispatch)
```

Both Part A and Part B are required: Part B ensures CacheAligner never sees the
observation block; Part A is a defensive guard in case ordering ever shifts.

---

### 2.11 Async Concurrency Model (resolves R3)

**The conflict:** The design specified fire-and-forget async Observer/Reflector
calls inside `ObservationalMemoryTransform.apply()`, which is a synchronous
`def` (`base.py:17–35`). Under concurrent proxy requests for the same session,
this produces race conditions on ObservationStore writes with no defined
backpressure or lifecycle.

**Resolution — proxy-level OM worker:**

Async OM operations are handled by a **dedicated proxy-level worker**, not
inside the transform. The transform only enqueues work and reads state.

```
ObservationalMemoryTransform.apply()
  ├── READS from ObservationStore (no lock needed — reads are safe)
  ├── If sync threshold hit → calls ObserverAgent inline (blocks apply())
  └── If buffer threshold hit → enqueues (session_key, slice) to OMWorker

OMWorker (proxy-level singleton)
  ├── Bounded asyncio queue (max depth: configurable, default 32 tasks)
  ├── Per-session-key asyncio.Lock — prevents duplicate work for same session
  ├── Max concurrent LLM calls: configurable (default 4)
  ├── Timeout: configurable (default 30s per Observer call)
  └── On failure: log + emit metric om_observer_error; do not crash request
```

**ObservationStore write safety:**
All write operations (`append_buffered`, `swap_buffered_to_active`,
`seal_message_ids`) are serialised through the OMWorker's per-session
`asyncio.Lock` before touching the database. SQLite's `BEGIN IMMEDIATE`
acquires a **database-level reserved lock** (not row-level — SQLite has no
row-level locking). This means at most one OM write is in-flight at any
moment across all sessions on a single SQLite file. SQLite WAL mode is
required to allow concurrent reads during writes.

**Expected throughput limits:** A single SQLite file in WAL mode supports
roughly 100–500 writes/second under local conditions. OM write frequency is
bounded by `worker_max_concurrency` (default 4) × observer LLM latency (typically
2–10 seconds), so realistic OM write rate is well under 10/second — comfortably
within SQLite limits. If write contention is observed (SQLITE_BUSY on retry),
the OMWorker backs off with exponential jitter (base 50ms, max 2s, 3 retries)
before failing the observation for that turn.

**Failure policy:**

- Observer timeout or error → observation skipped for this turn; unobserved
  tokens remain eligible for observation on the next turn.
- Reflector timeout or error → reflection skipped; buffered observations
  accumulate until the next successful Reflector run.
- Repeated failures (≥3 consecutive) → circuit-break OM for that session for
  60s; emit metric `om_circuit_open`.

**New config fields:**

```python
@dataclass
class ObservableMemoryConfig:
    ...
    worker_queue_depth: int = 32
    worker_max_concurrency: int = 4
    observer_timeout_seconds: int = 30
    reflector_timeout_seconds: int = 60
    circuit_break_threshold: int = 3
    circuit_break_cooldown_seconds: int = 60
```

---

### 2.12 Emergency Terminal Cap (resolves R4)

**The concern:** Removing the RollingWindow fallback eliminates the hard budget
guarantee present in both `RollingWindow` (`rolling_window.py:115`) and
`IntelligentContextManager` (`intelligent_context.py:362`). The sync Observer
guarantee covers the common case, but edge cases remain: a very large system
prompt plus a large active observation block could push total tokens over the
model's context window even after observation compression.

**Resolution — retain a minimal hard cap as a named emergency path:**

After the Observer and Reflector complete, the transform checks total token
count one final time. If still over the model limit, it applies a simple
position-based hard drop (oldest non-system, non-observation messages first) and
emits a metric:

```python
# After Observer/Reflector complete
remaining_tokens = tokenizer.count_messages(messages)
if remaining_tokens > model_limit - output_buffer:
    messages = _emergency_hard_cap(messages, model_limit, output_buffer)
    metrics.increment("om_emergency_cap_triggered")
    logger.warning("OM emergency cap triggered for session %s", session_key)
```

This path is not part of the normal OM flow — it is an explicit safety valve.
The metric makes it visible so operators can tune thresholds if it fires
frequently.

The design comparison matrix row "No unnecessary fallback strategy" remains ✓
because this cap is a named, instrumented emergency path, not a silent fallback.

---

### 2.13 Data Retention and Compliance (resolves R5)

**The gap:** ObservationStore persists user conversations compressed into
observations indefinitely. No retention policy, purge API, or encryption
boundary was defined.

**Resolution:**

**Retention policy (config-driven):**

```python
@dataclass
class ObservableMemoryConfig:
    ...
    # Set to None for no TTL (default).
    # If set, rows with expires_at < now() are purged by the scheduled cleaner.
    session_ttl_seconds: int | None = None
    # How often the purge job runs (default: every hour)
    purge_interval_seconds: int = 3600
```

When `session_ttl_seconds` is set, `expires_at` is written as
`created_at + ttl` on row creation and updated on each write.

**Purge APIs:**

- `purge_session(session_key)` — immediately hard-deletes all observation rows
  for a session. Exposed via `headroom memory purge --session <key>` CLI command
  and as a proxy endpoint `DELETE /v1/om/session/<key>` (authenticated).
- `purge_expired()` — scheduled background job, runs every `purge_interval_seconds`.

**Encryption at rest:**
ObservationStore lives in the same SQLite file as `SQLiteMemoryStore`. The
encryption-at-rest posture is the same as the existing memory system — at-rest
encryption is the operator's responsibility (filesystem encryption, SQLCipher,
or equivalent). No new requirement is introduced.

**Access boundaries:**
Observations are partitioned by `session_key`. No cross-session read is possible
via the store API.

**Trust model clarification (N1):** The proxy has **no request authentication
layer** for `x-headroom-*` headers (`server.py:1438–1440` accepts them as
raw client input). The isolation guarantee is therefore: a client cannot read
another session's observations through the OM API, but a client that knows or
guesses another session's `x-headroom-session-id` value can write observations
into it. This is the same trust posture as the existing memory system.

For **single-tenant or trusted-network deployments** this is acceptable.
For **multi-tenant production deployments** the recommended hardening is:

- Derive `session_key` server-side from an authenticated principal (e.g., JWT
  sub-claim + session nonce) rather than trusting the client header directly.
- Or: HMAC-sign the session ID server-side and verify on each request before
  deriving `session_key`. Implementation is an operator responsibility and
  out of scope for the initial OM feature, but the design must not claim
  authentication protection that does not exist.

**PII guidance (documentation, not enforcement):**
Observations are LLM-generated summaries of conversation content. If the
underlying conversation contains PII, the observations may too. Operators
handling regulated data should set `session_ttl_seconds` and use the purge API
in their data deletion workflows.

---

### 2.14 Context Strategy Selection and Logging

**Defaults:**

| Strategy | Default | How to enable |
|----------|---------|---------------|
| `RollingWindow` | **on** | always active unless overridden |
| `IntelligentContextManager` | off | `intelligent_context.enabled=True` / `--intelligent-context` |
| `ObservationalMemory` | off | `observable_memory.enabled=True` / `--observable-memory` |

RollingWindow is the baseline. ICM and OM are both opt-in and replace it when
enabled. OM is not a successor to ICM — they are independent alternatives.

**Precedence order when multiple strategies are enabled (highest → lowest):**

```
ObservationalMemory       (opt-in)
IntelligentContextManager (opt-in)
RollingWindow             (default)
```

Only one strategy runs. If multiple are enabled, the highest-priority one is
selected and the rest are silently skipped.

**Startup log — always emit, regardless of which strategy wins:**

```python
logger.info("Pipeline context strategy: %s", selected_strategy_name)
```

Examples:
```
Pipeline context strategy: ObservationalMemory
Pipeline context strategy: IntelligentContextManager (COMPRESS_FIRST -> SUMMARIZE -> DROP_BY_SCORE)
Pipeline context strategy: RollingWindow
Pipeline context strategy: none (pass-through)
```

**Warn when a lower-priority strategy is overridden:**

```python
if om_enabled and (icm_enabled or rw_enabled):
    overridden = "IntelligentContextManager" if icm_enabled else "RollingWindow"
    logger.warning(
        "ObservationalMemory is enabled — %s config present but will not run. "
        "Disable it explicitly to suppress this warning.",
        overridden,
    )
if icm_enabled and rw_enabled:
    logger.warning(
        "IntelligentContextManager is enabled — RollingWindow config present "
        "but will not run. Disable it explicitly to suppress this warning."
    )
```

This aligns with the existing ICM log line in `pipeline.py` while extending it
to cover all strategies consistently, including the currently-unlogged
`RollingWindow` case.

---

## 3. What Is Not Changing (and One Correction)

The following headroom systems are **unchanged** by this integration:

- `ContentRouter` / `SmartCrusher` / `CodeCompressor` etc. — still run at Step 2, unmodified
- `IntelligentContextManager` — still available as an alternative strategy
- `RollingWindow` — still available as an alternative strategy
- `CompressionStore` (CCR) — unchanged; OM does not replace CCR
- `MemoryHandler` (episodic memory) — unchanged; coexists with OM observations
- `hooks.py` (`pre_compress`, `compute_biases`, `post_compress`) — unchanged

**One correction from review (R2):** `CacheAligner` requires a minimal change.
Its system message processing loop must skip messages tagged with
`{"source": "observational-memory"}` to prevent date extraction from distorting
the observation block. Additionally, OM prefix injection moves to Step 1.5
(after CacheAligner, before ContentRouter) rather than before the pipeline.
See §2.10 for the full analysis and fix.

---

## 4. Open Questions for Reviewers

1. **Small-context model handling.** ~~Should the config validation emit a warning
   (or error) when the resolved threshold exceeds 50% of the model's context
   window?~~ **Decided: both.** Emit a `logger.warning` at 50% and raise a
   `ConfigurationError` at a higher threshold (e.g. 75%), so users get an early
   heads-up before hitting a hard failure.

2. **Observer/Reflector model cost.** Defaulting to the configured provider
   means a Sonnet-class model doing background observation compression. Should
   the default be a cheaper model (Haiku, GPT-4o-mini) with the user opting
   into a more capable one? Or does observation quality matter enough to
   justify the same model?

3. **Interaction with CCR.** OM observations give the LLM a semantic view of
   history; CCR gives it the raw data on demand. Both can coexist. Is there
   value in the ObserverAgent annotating which observations are backed by a CCR
   reference (so the LLM knows it can retrieve exact data for 🔴 observations)?

4. **Scope — thread vs resource.** Mastra supports both thread-scoped and
   resource-scoped observations (shared across threads for the same user). The
   initial implementation targets thread-scoped only. Resource-scoped is a
   follow-on. Agreement needed.

5. **Observation persistence.** Should the ObservationStore live in the same
   SQLite file as the existing `MemoryStore`, or in a separate file? Separate
   files are easier to clear independently; same file simplifies backup.

_The following questions raised by the design review are now resolved in the
design and no longer require reviewer input:_

- ~~Session identity model undefined (R1)~~ → resolved in §2.9
- ~~CacheAligner conflict with date-tagged observations (R2)~~ → resolved in §2.10
- ~~Async concurrency model undefined (R3)~~ → resolved in §2.11
- ~~No hard budget safety net after removing fallback (R4)~~ → resolved in §2.12
- ~~No retention/compliance controls for ObservationStore (R5)~~ → resolved in §2.13

_Round 2 review findings (N1–N3) now also resolved:_

- ~~Session header trust model overstated — claimed non-existent auth layer (N1)~~ → corrected in §2.9 (fingerprint fallback hardened with multi-source entropy, production default changed from `"fingerprint"` to `"thread"`) and §2.13 (trust model clarified — no auth layer exists, multi-tenant hardening is operator responsibility)
- ~~SQLite "row-level lock" terminology incorrect (N2)~~ → corrected in §2.11 — now accurately describes database-level reserved lock, WAL requirement, throughput limits, and SQLITE_BUSY backoff policy
- ~~Fingerprint fallback collides on common prompts (N3)~~ → corrected in §2.9 — fingerprint now includes `user_id + model + timestamp_bucket + message_content`; disabled by default in production

---

## 5. Implementation Plan

### Phase 1 — Port `@mastra/memory` OM to Python (isolated within headroom repo)

**Goal:** Produce a standalone, framework-agnostic Python package that any project
can use without depending on headroom, co-located in the headroom repo and
installable as its own extra.

**Location in repo:**

```
headroom/
└── observable_memory/        ← self-contained, no imports from headroom core
    ├── __init__.py
    ├── processor.py          ← ObservableMemoryProcessor
    ├── observer.py           ← ObserverAgent + prompt templates
    ├── reflector.py          ← ReflectorAgent + prompt templates
    ├── store.py              ← ObservationStore ABC + SQLiteObservationStore
    ├── token_counter.py      ← TokenCounter
    ├── types.py              ← ObservableMemoryConfig, ObservationState, etc.
    └── worker.py             ← OMWorker (async queue + per-session mutex)
```

**Install:**

```bash
pip install "headroom-ai[observable-memory]"
```

**Source material:**

| File | Lines | Action |
|------|-------|--------|
| `observational-memory.ts` | 6,128 | Port core logic |
| `observer-agent.ts` | 882 | Port — prompt templates copy verbatim |
| `reflector-agent.ts` | 337 | Port — prompt templates copy verbatim |
| `token-counter.ts` | 138 | Port — `js-tiktoken` → Python `tiktoken` |
| `types.ts` | 646 | Port — TS interfaces → Python `dataclass` / `Protocol` |
| `__tests__/*.ts` | 10,081 | Port alongside implementation as correctness spec |

**Key adaptations from Mastra:**

- Replace `@mastra/core` types with a minimal `LLMProvider` Protocol.
  Observer and Reflector call the LLM with a system prompt + user prompt and
  receive raw text back — all parsing is done by the OM package itself:
  ```python
  class LLMProvider(Protocol):
      async def complete(
          self,
          system: str,
          prompt: str,
          model: str | None = None,
      ) -> str: ...
      def count_tokens(self, text: str, model: str | None = None) -> int: ...
  ```
- Replace Mastra's storage abstraction with a built-in `SQLiteObservationStore`
  (host passes `db_path`; custom backends implement `ObservationStore` ABC)
- Replace `async-mutex` with `asyncio.Lock`
- Replace TypeScript `Map` static state with Python class-level `dict`
- Replace Mastra's AI SDK client calls with `await llm.complete(...)`
- Default models: `None` (inherit from caller) instead of Mastra's hardcoded
  `google/gemini-2.5-flash`
- Thresholds: context-window-relative ratios (0.25 / 0.35) instead of Mastra's
  hardcoded 30k / 40k absolutes

**Isolation constraint:** `headroom/observable_memory/` must not import anything
from `headroom/` core. The boundary is enforced by the `LLMProvider` Protocol
and `db_path` — headroom wires those up in Phase 2.

**Deliverable:** A working Python port with all Mastra OM tests ported and
passing, installable as `headroom-ai[observable-memory]` independently of the
headroom pipeline.

---

### Phase 2 — Integrate OM into headroom

**Goal:** Wire the Phase 1 package into headroom's pipeline, proxy, and CLI.

**Changes to headroom:**

- `pip install headroom-ai[om]` pulls in the Phase 1 package as a dependency
- `headroom/transforms/pipeline.py` — add OM as the highest-priority context
  strategy (see §2.14 precedence order)
- `headroom/transforms/pipeline.py` — inject active observation block at Step 1.5
  (after CacheAligner, before ContentRouter — see §2.10)
- `headroom/transforms/cache_aligner.py` — add tag-based exemption for
  `{"source": "observational-memory"}` messages (see §2.10)
- `headroom/proxy/server.py` — start/stop `OMWorker` singleton with proxy
  lifecycle; wire `LLMProvider` implementation using proxy's configured client
- `headroom/config.py` — add `ObservableMemoryConfig` to `HeadroomConfig`
- `headroom/cli/proxy.py` — add `--observable-memory` and `--om-*` CLI flags
  (see §2.4)
- `headroom/storage/` — pass existing SQLite DB path to `SQLiteObservationStore`
  so `om_observations` table co-locates with existing memory tables (see §2.5)

**Deliverable:** `headroom proxy --observable-memory` works end-to-end, all
headroom OM integration tests pass (see §6 Test Plan), published as
`headroom-ai[om]` extra.

---

## 6. Test Plan (Phase 2)

### 6.1 Unit Tests

#### `ObservationStore` (`tests/test_observable_memory/test_observation_store.py`)

| Test                                            | What it verifies                                                                        |
| ----------------------------------------------- | --------------------------------------------------------------------------------------- |
| `test_get_state_returns_empty_for_new_session`  | Fresh session returns zero-state defaults                                               |
| `test_append_buffered_accumulates`              | Multiple `append_buffered` calls concatenate correctly                                  |
| `test_swap_buffered_to_active_is_atomic`        | Swap moves buffered → active and clears buffered in a single SQLite transaction         |
| `test_seal_message_ids_prevents_re_observation` | Sealed IDs excluded from next unobserved slice                                          |
| `test_seal_uses_fingerprint_when_no_id_field`   | `sha256(role+content)[:16]` fingerprint used when message has no `id`                   |
| `test_purge_session_hard_deletes`               | `purge_session` removes all rows for that key, leaves others intact                     |
| `test_purge_expired_respects_ttl`               | Rows with `expires_at < now()` removed; unexpired rows preserved                        |
| `test_concurrent_writes_serialised`             | Two coroutines writing to the same session do not corrupt state (per-session mutex)     |
| `test_sqlite_busy_backoff`                      | Simulated `SQLITE_BUSY` triggers exponential backoff (≤3 retries) then fails gracefully |

#### Session Key Derivation (`tests/test_observable_memory/test_session_key.py`)

| Test                                                     | What it verifies                                                            |
| -------------------------------------------------------- | --------------------------------------------------------------------------- |
| `test_explicit_header_takes_priority`                    | `x-headroom-session-id` header wins over all other strategies               |
| `test_thread_fallback_derives_from_user_and_thread`      | `sha256(user_id + ":" + thread_id)[:16]` computed correctly                 |
| `test_fingerprint_fallback_includes_all_entropy_sources` | `user_id + model + timestamp_bucket + message_content` all hashed           |
| `test_fingerprint_fallback_disabled_by_default`          | `session_fallback="thread"` default — fingerprint not used unless opted in  |
| `test_common_greeting_no_collision_with_fingerprint`     | "Hello" from two different users at different hours produces different keys |
| `test_default_sentinel_on_empty_messages`                | No messages → `session_key = "default"`                                     |

#### Threshold Resolution (`tests/test_observable_memory/test_threshold_resolution.py`)

| Test                                                    | What it verifies                                                |
| ------------------------------------------------------- | --------------------------------------------------------------- |
| `test_ratio_applied_to_context_window`                  | `0.25 × 128000 = 32000`                                         |
| `test_absolute_override_wins_over_ratio`                | When `message_threshold_tokens` set, ratio ignored              |
| `test_buffer_ratio_applied_to_hard_threshold`           | `buffer_ratio=0.20` → async fires at 20% of hard threshold      |
| `test_config_validation_warns_on_threshold_above_50pct` | Warning emitted when resolved threshold > 50% of context window |
| `test_cli_absolute_flag_wins_over_config_ratio`         | CLI `--om-message-threshold-tokens` beats config file ratio     |

#### `ObservationalMemoryTransform.apply()` (`tests/test_observable_memory/test_transform.py`)

| Test                                             | What it verifies                                                                                    |
| ------------------------------------------------ | --------------------------------------------------------------------------------------------------- |
| `test_injects_active_obs_as_system_prefix`       | Active observations appear as system message with `{"source": "observational-memory"}` tag          |
| `test_no_injection_when_obs_empty`               | No system message injected when `active_observations` is empty                                      |
| `test_async_enqueue_at_buffer_threshold`         | OMWorker enqueued (not called inline) when buffer threshold crossed                                 |
| `test_sync_observer_called_at_hard_threshold`    | ObserverAgent called synchronously when hard threshold crossed                                      |
| `test_emergency_cap_fires_when_still_over_limit` | Hard drop applied and `om_emergency_cap_triggered` metric emitted when post-OM tokens > model limit |
| `test_emergency_cap_not_fired_in_normal_path`    | Metric NOT emitted when tokens fit within model limit                                               |
| `test_om_disabled_passes_through`                | When `enabled=False`, messages pass through unchanged                                               |

#### `CacheAligner` Exemption (`tests/test_cache/test_cache_aligner_om_exemption.py`)

| Test                                           | What it verifies                                                                                         |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `test_observation_block_not_stripped`          | System message tagged `{"source": "observational-memory"}` passes through CacheAligner with dates intact |
| `test_regular_system_messages_still_processed` | Non-OM system messages still have dynamic content extracted (existing behaviour unchanged)               |

#### `OMWorker` (`tests/test_observable_memory/test_om_worker.py`)

| Test                                              | What it verifies                                                                                   |
| ------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `test_per_session_mutex_prevents_parallel_writes` | Two async tasks for same session serialised — second waits for first                               |
| `test_different_sessions_run_concurrently`        | Tasks for different sessions do not block each other                                               |
| `test_queue_bounded_at_max_depth`                 | Enqueueing beyond `worker_queue_depth` raises or drops cleanly                                     |
| `test_observer_timeout_emits_metric_and_skips`    | Observer call exceeding `observer_timeout_seconds` → `om_observer_error` metric, request continues |
| `test_circuit_breaker_opens_after_threshold`      | ≥3 consecutive failures → `om_circuit_open` metric, OM skipped for session for cooldown period     |
| `test_circuit_breaker_resets_after_cooldown`      | After `circuit_break_cooldown_seconds`, OM re-enabled for session                                  |

#### Observer / Reflector Prompt Contract (`tests/test_observable_memory/test_agent_prompts.py`)

| Test                                        | What it verifies                                                               |
| ------------------------------------------- | ------------------------------------------------------------------------------ |
| `test_observer_output_format_parsed`        | Output with `[date \| ref: date \| +Nd] 🔴/🟡/🟢 text` correctly parsed        |
| `test_reflector_level_selected_by_overflow` | Level 0/1/2/3 selected based on how far obs_tokens exceed reflection threshold |
| `test_observer_preserves_critical_priority` | 🔴 items retained by Reflector even at level 3 compression                     |

---

### 6.2 Integration Tests

#### Pipeline Integration (`tests/test_observable_memory/test_pipeline_integration.py`)

| Test                                              | What it verifies                                                                    |
| ------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `test_injection_order_step_1_5`                   | Observation block injected after CacheAligner, before ContentRouter                 |
| `test_om_coexists_with_smartcrusher`              | SmartCrusher still compresses tool outputs; observation block untouched             |
| `test_om_not_active_when_rolling_window_selected` | Selecting `RollingWindow` strategy → OM transform not in pipeline                   |
| `test_om_not_active_when_icm_selected`            | Selecting `IntelligentContextManager` → OM transform not in pipeline                |
| `test_episodic_memory_coexists_with_om`           | `MemoryHandler` injections and OM observations both present in messages sent to LLM |

#### Proxy Integration (`tests/test_observable_memory/test_proxy_integration.py`)

| Test                                     | What it verifies                                                       |
| ---------------------------------------- | ---------------------------------------------------------------------- |
| `test_proxy_om_enabled_via_cli_flag`     | `--observable-memory` flag activates OM in proxy                       |
| `test_session_key_extracted_from_header` | `x-headroom-session-id` header propagated to ObservationStore          |
| `test_purge_endpoint_deletes_session`    | `DELETE /v1/om/session/<key>` removes observations and returns 200     |
| `test_om_worker_starts_with_proxy`       | OMWorker singleton initialised on proxy startup, shut down on teardown |

---

### 6.3 End-to-End Tests (Live LLM, marked `@pytest.mark.e2e`)

These require a real LLM API key and are excluded from CI by default (`pytest -m "not e2e"`).

| Test                                                 | What it verifies                                                                        |
| ---------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `test_observer_compresses_long_conversation`         | After 20+ turns, ObserverAgent produces valid observation block; token count reduced    |
| `test_reflector_compresses_large_obs_block`          | Observation block exceeding reflection threshold is compressed by ReflectorAgent        |
| `test_kv_cache_prefix_stability`                     | Observation block prefix byte-identical across consecutive turns (KV cache eligible)    |
| `test_emergency_cap_not_triggered_in_normal_session` | 50-turn conversation with standard model completes without `om_emergency_cap_triggered` |
| `test_needle_retained_after_observation`             | Critical fact planted in early turns recoverable from observation block in later turns  |

---

### 6.4 Test Fixtures and Helpers

- **`om_config()`** — pytest fixture returning a default `ObservableMemoryConfig` with small thresholds (e.g., 500 tokens) for fast test execution without large message history.
- **`mock_observer_agent(observations)`** — returns a mock that produces a fixed observation string, avoiding LLM calls in unit/integration tests.
- **`mock_reflector_agent(level)`** — returns a mock that echoes input with a level annotation for Reflector level selection tests.
- **`observation_store(tmp_path)`** — pytest fixture providing an `ObservationStore` backed by a temporary SQLite file, torn down after each test.
- **`proxy_with_om()`** — pytest fixture that starts a proxy with `--observable-memory` and tears it down after each integration test.

---

### 6.5 Test Coverage Targets

| Component                      | Target                                               |
| ------------------------------ | ---------------------------------------------------- |
| `ObservationStore`             | 95% — all store operations and error paths           |
| Session key derivation         | 100% — all 4 priority levels and edge cases          |
| Threshold resolution           | 100% — all precedence levels                         |
| `ObservationalMemoryTransform` | 90% — all threshold branches including emergency cap |
| `OMWorker`                     | 85% — concurrency paths require async test harness   |
| `CacheAligner` exemption       | 100% — two-line change, full coverage required       |
| Agent prompt parsing           | 80% — format validation, not LLM output quality      |

---

## 7. Out of Scope (Follow-on Work)

- Resource-scoped (cross-thread) observations
- Exposing ObservationStore contents via the headroom dashboard
- Export of observations to the existing markdown bridge (`memory_bridge`)
- Eval suite additions for Observable Memory accuracy benchmarking
- CCR annotation of 🔴 observations (open question #3 above)

---

## Appendix A: Design Iteration History

Five distinct designs were explored before converging on the final approach.
Understanding why each was rejected clarifies the final choice.

### Design Comparison Matrix

| Criterion                               | Design 1 (Transforms) | Design 2 (Processor Layer) | Design 3 (ICM-Embedded) | Design 4 (Third Option + Fallback) | **Design 5 (Final)** |
| --------------------------------------- | --------------------- | -------------------------- | ----------------------- | ---------------------------------- | -------------------- |
| Proactive (fires before budget crisis)  | ✗                     | ✓                          | ✗                       | ✓                                  | ✓                    |
| Stable KV-cacheable prefix              | ✗                     | ✓                          | Partial                 | ✓                                  | ✓                    |
| Async background (no hot-path latency)  | ✗                     | ✓                          | Partial                 | ✓                                  | ✓                    |
| Separate observation storage            | ✗                     | ✓                          | ✗                       | ✓                                  | ✓                    |
| Faithful to Mastra's design             | ✗                     | ✓                          | Partial                 | Partial                            | ✓                    |
| Reuses existing headroom infrastructure | ✓                     | Partial                    | ✓                       | Partial                            | Partial              |
| No new cross-turn state                 | ✓                     | ✗                          | ✓                       | ✗                                  | ✗ (necessary)        |
| Context-window-relative thresholds      | ✗                     | ✗                          | ✗                       | ✗                                  | ✓                    |
| CLI + config configurable thresholds    | ✗                     | ✗                          | ✗                       | ✗                                  | ✓                    |
| No unnecessary fallback strategy        | N/A                   | N/A                        | N/A                     | ✗                                  | ✓                    |
| Compatible with all models              | ✗                     | ✗                          | ✓                       | ✗                                  | ✓                    |
| **Overall rating**                      | **4/10**              | **6/10**                   | **7/10**                | **7.5/10**                         | **9/10**             |

---

### Design 1 — Transforms in the Compression Pipeline

**Concept:** Add `ObserverTransform` and `ReflectorTransform` directly into
`headroom/transforms/` alongside `SmartCrusher`, `CodeCompressor`, etc. They
fire at fixed token thresholds, call an LLM, and store observations in
headroom's existing `Memory` system.

**Rating: 4 / 10**

| Strength                                                      | Weakness                                                                                         |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Follows existing transform pattern — minimal new architecture | Runs synchronously in the hot path — adds 200–800ms LLM latency to every request above threshold |
| No new pipeline stages                                        | Observations mixed into general `Memory` store — indistinguishable from episodic memories        |
|                                                               | No stable prefix achievable — general memory injection is dynamic, not positionally stable       |
|                                                               | Transform interface `apply(messages) → TransformResult` is not designed for cross-turn state     |

**Why rejected:** The transform pipeline is designed for stateless, per-request
content manipulation. Observable Memory is inherently stateful across turns. The
mismatch forces awkward workarounds, and the synchronous execution destroys the
latency profile.

---

### Design 2 — Standalone Processor Layer (Option B)

**Concept:** A dedicated `headroom/memory/observation/` module with
`ObserverProcessor` and `ReflectorProcessor` running as async background tasks.
Triggered by per-session token thresholds tracked by a new cross-turn
`TokenCounter`. Observations stored in a separate `ObservationStore`. Stable
observation block injected as system prefix before the pipeline.

**Rating: 6 / 10**

| Strength                                       | Weakness                                                                                                           |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| True to Mastra's original design intent        | Per-session `TokenCounter` with sealed message ID tracking — entirely new cross-turn state not present in headroom |
| Async background avoids hot-path latency       | Start-of-turn activation is synchronous — coordination between async background and sync activation is non-trivial |
| Stable prefix for KV caching                   | Injection of stable prefix before `CacheAligner` requires careful pipeline sequencing to avoid conflicts           |
| Separate `ObservationStore` — clean separation | Two new infrastructure pieces: `TokenCounter` + `ObservationStore`                                                 |

**Initial assessment of cons was overstated:** After reading Mastra source,
cons #2 and #4 were less severe than described. Mastra uses a static
`asyncBufferingOps` Map and synchronous Step 0 activation — the async
coordination is straightforward. Injection uses a dedicated
`'observational-memory'` system message tag that doesn't conflict with
`CacheAligner`. The real complexity is the `TokenCounter` and the fact that
observations are triggered independently of budget pressure.

**Why rejected:** The independent token counter was the decisive issue. Headroom
must already compute token counts for budget management — adding a parallel
counter for a different threshold is redundant. A design that reuses existing
token accounting is preferable.

---

### Design 3 — ICM-Embedded (Observer as Phase 2 Summarization)

**Concept:** Implement Observable Memory inside `IntelligentContextManager` as
the fulfillment of its existing Phase 2 `summarize_fn` hook. When ICM decides
to drop messages (DROP_BY_SCORE strategy), instead of inserting a plain marker,
it calls the ObserverAgent on the about-to-be-dropped messages. The observation
replaces the plain `"[N messages dropped]"` marker.

**Rating: 7 / 10**

| Strength                                                                        | Weakness                                                                                                                                              |
| ------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| Leverages the `summarize_fn` callback — already exists for exactly this purpose | **Reactive, not proactive.** Observations only generated when messages are dropped — no observations accumulate until the budget is already exhausted |
| No separate token counter — ICM already knows what it's dropping                | Tied to ICM — users on `RollingWindow` get no benefit                                                                                                 |
| No new pipeline stages                                                          | Observation content is only the dropped portion — not a holistic view of conversation history                                                         |
| Natural integration: observation IS the drop summary                            | `summarize_fn` signature `(messages, context) → str` means the call is synchronous in the drop path — blocks until the LLM responds                   |

**Why rejected:** The fundamental issue is proactivity. ICM-embedded only
observes what it's discarding — it never compresses history ahead of budget
pressure. This means the LLM always operates with a full raw message history
until the crisis point, then suddenly gets observations. Mastra's value is in
the gradual, incremental compression that happens well before the budget is
tight. The ICM-embedded design captures perhaps 30% of the benefit.

---

### Design 4 — ObservableMemory as Third Pipeline Strategy (with Fallback)

**Concept:** `ObservationalMemoryTransform` becomes a peer context management
strategy alongside `RollingWindow` and `ICM`, selected via config at Step 3 of
the pipeline. Runs its own token threshold logic. Includes a `RollingWindow`
fallback for when the transform is still over budget after observations.

**Rating: 7.5 / 10**

| Strength                                                                       | Weakness                                                                                                         |
| ------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| Clean architectural separation — one of three options, not embedded in another | `RollingWindow` fallback not present in Mastra's design — adds complexity without precedent                      |
| Proactive — fires at threshold ratio, not at budget exhaustion                 | Fallback creates ambiguity: if OM fires and is still over budget, it silently falls back to a different strategy |
| Async/sync split matches Mastra's approach                                     |                                                                                                                  |
| Self-contained — its own store, its own thresholds                             |                                                                                                                  |

**Key correction from Mastra source:** The `RollingWindow` fallback was added
defensively, but Mastra's sync threshold guarantee makes it unnecessary. The
Observer at the 30k hard threshold runs synchronously and _guarantees_ the
result before the LLM call — that IS the safety mechanism. No fallback needed.

**Why revised:** User correctly identified that the RollingWindow fallback does
not exist in Mastra's design. Removing it makes the design cleaner and more
faithful.

---

### Design 5 — Final Design (Configurable Ratios, No Fallback)

**Concept:** Same as Design 4, with two critical improvements:

1. **No RollingWindow fallback** — sync thresholds guarantee the observation
   block is ready before the LLM call, exactly as Mastra does.
2. **Context-window-relative thresholds** — after discovering that Mastra's
   30k/40k defaults were hardcoded with `model: 'google/gemini-2.5-flash'` as
   the default for both Observer and Reflector, with no documented rationale for
   the specific token values, thresholds are expressed as fractions of the
   model's actual context window. Both ratio and absolute values are configurable
   via config file and CLI.

**Rating: 9 / 10**

---

### Why the Rating Improved Across Iterations

The progression was driven by three things:

**1. Reading the actual Mastra source.**
Initial assumptions about Mastra's design were largely correct but two were
wrong: compression ratios are LLM-guided (not programmatic guarantees), and the
async/sync coordination is simpler than assumed. Reading the real code also
revealed that `bufferActivation: 0.8` and `bufferTokens: 0.2` create a
pre-warming layer that makes the hard-threshold sync path fast in practice.

**2. Correcting the RollingWindow fallback.**
Design 4 introduced a fallback that felt prudent but was unnecessary and not
present in Mastra's implementation. The user's observation prompted its removal.
The sync hard threshold IS the safety guarantee — no fallback needed.

**3. Investigating the threshold origins.**
Discovering that Mastra's 30k/40k defaults were hardcoded for Gemini 2.5 Flash
(1M context window) with no documented rationale — and that they would be
useless or counter-productive for small-context models — motivated the
context-window-relative threshold design. This is the single most significant
improvement over Mastra's approach, making the feature safe to deploy across
headroom's full model range.

## Known remaining work (from design doc, out of scope for Phase 2):
  - Streaming support
  - Transform pipeline integration (OM as peer strategy to RollingWindow/ICM)
  - HeadroomClient SDK integration
  - Google Gemini + OpenAI Responses API handlers
  - Purge endpoint, advanced OMWorker config, TTL/retention