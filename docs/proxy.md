# Proxy Server Documentation

The Headroom proxy server is a production-ready HTTP server that applies context optimization to all requests passing through it.

## Starting the Proxy

```bash
# Basic usage
headroom proxy

# Custom port
headroom proxy --port 8080

# With all options
headroom proxy \
  --host 0.0.0.0 \
  --port 8787 \
  --log-file /var/log/headroom.jsonl \
  --budget 100.0
```

## Command Line Options

### Core Options

| Option | Default | Description |
|--------|---------|-------------|
| `--host` | `127.0.0.1` | Host to bind to |
| `--port` | `8787` | Port to bind to |
| `--no-optimize` | `false` | Disable optimization (passthrough mode) |
| `--no-cache` | `false` | Disable semantic caching |
| `--no-rate-limit` | `false` | Disable rate limiting |
| `--log-file` | None | Path to JSONL log file |
| `--budget` | None | Daily budget limit in USD |
| `--openai-api-url` | `https://api.openai.com` | Custom OpenAI API URL endpoint |

### Context Management Options

| Option | Default | Description |
|--------|---------|-------------|
| `--no-intelligent-context` | `false` | Disable IntelligentContextManager (fall back to RollingWindow) |
| `--no-intelligent-scoring` | `false` | Disable multi-factor importance scoring (use position-based) |
| `--no-compress-first` | `false` | Disable trying deeper compression before dropping messages |

By default, the proxy uses **IntelligentContextManager** which scores messages by multiple factors (recency, semantic similarity, TOIN-learned patterns, error indicators, forward references) and drops lowest-scored messages first. This is smarter than simple age-based truncation.

**CCR Integration:** When messages are dropped, they're stored in CCR so the LLM can retrieve them if needed. The inserted marker includes the CCR reference. Drops are also recorded to TOIN, so the system learns which message patterns are important across all users.

```bash
# Use legacy RollingWindow (drops oldest first)
headroom proxy --no-intelligent-context

# Disable semantic scoring (faster, but less intelligent)
headroom proxy --no-intelligent-scoring
```

### Observable Memory Options

| Option | Default | Description |
|--------|---------|-------------|
| `--observable-memory` | `false` | Enable Observable Memory proactive compression |
| `--observable-memory-observer-model` | `None` | LLM for Observer agent (inherits upstream model if not set) |
| `--observable-memory-reflector-model` | `None` | LLM for Reflector agent (inherits observer model if not set) |
| `--observable-memory-db-path` | `":memory:"` | SQLite path for observations, or `":memory:"` for ephemeral |
| `--observable-memory-message-threshold-ratio` | `0.25` | Min context usage before observing (0.0–1.0) |
| `--observable-memory-observation-threshold-ratio` | `0.35` | Observation size that triggers Reflector (0.0–1.0) |
| `--observable-memory-instruction` | `None` | Custom instructions appended to Observer and Reflector prompts |
| `--observable-memory-observer-api-key` | `None` | API key for Observer LLM (if using a different provider) |

**Note:** Observable Memory requires additional dependencies: `pip install headroom-ai[observable-memory]`

```bash
# Enable Observable Memory with a cheap observer model
headroom proxy --observable-memory --observable-memory-observer-model gpt-4o-mini

# With SQLite persistence
headroom proxy \
  --observable-memory \
  --observable-memory-observer-model gpt-4o-mini \
  --observable-memory-db-path /var/data/observations.db

# With custom observer instructions
headroom proxy \
  --observable-memory \
  --observable-memory-instruction "Focus on technical decisions and error resolutions."
```

### LLMLingua Options (ML Compression)

| Option | Default | Description |
|--------|---------|-------------|
| `--llmlingua` | `false` | Enable LLMLingua-2 ML-based compression |
| `--llmlingua-device` | `auto` | Device for model: `auto`, `cuda`, `cpu`, `mps` |
| `--llmlingua-rate` | `0.3` | Target compression rate (0.3 = keep 30% of tokens) |

**Note:** LLMLingua requires additional dependencies: `pip install headroom-ai[llmlingua]`

```bash
# Enable LLMLingua with GPU acceleration
headroom proxy --llmlingua --llmlingua-device cuda

# More aggressive compression (keep only 20%)
headroom proxy --llmlingua --llmlingua-rate 0.2

# Conservative compression for code (keep 50%)
headroom proxy --llmlingua --llmlingua-rate 0.5
```

## API Endpoints

### Health Check

```bash
curl http://localhost:8787/health
```

Response:
```json
{
  "status": "healthy",
  "optimize": true,
  "stats": {
    "total_requests": 42,
    "tokens_saved": 15000,
    "savings_percent": 45.2
  }
}
```

### Detailed Statistics

```bash
curl http://localhost:8787/stats
```

### Prometheus Metrics

```bash
curl http://localhost:8787/metrics
```

### LLM APIs

The proxy supports both Anthropic and OpenAI API formats:

```bash
# Anthropic format
POST /v1/messages

# OpenAI format
POST /v1/chat/completions
```

## Using with Claude Code

```bash
# Start proxy
headroom proxy --port 8787

# In another terminal
ANTHROPIC_BASE_URL=http://localhost:8787 claude
```

## Using with Cursor

1. Start the proxy: `headroom proxy`
2. In Cursor settings, set the base URL to `http://localhost:8787`

## Using with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8787/v1",
    api_key="your-api-key",  # Still needed for upstream
)
```

## Features

### Observable Memory (Opt-In)

When enabled, the proxy proactively compresses message history using Observer/Reflector LLM agents. Every conversation passing through the proxy automatically receives background summarization — no client changes required.

```bash
headroom proxy --observable-memory --observable-memory-observer-model gpt-4o-mini
```

**How it works:**
- **Before each request** — stored observations from prior turns are injected into the system prompt as `<memory>...</memory>`
- **After each response** — a background Observer LLM reads the new turn and writes structured observations (facts, decisions, blockers, preferences)
- **When observations grow large** — a Reflector LLM compresses them with escalating detail reduction

**Thread ID resolution** — the proxy auto-derives a thread ID for each conversation from request headers (first match wins):

1. `x-headroom-thread-id` — explicit Headroom ID
2. `helicone-session-id` — Helicone session
3. `x-portkey-trace-id` — Portkey trace
4. `mcp-session-id` — MCP session
5. Content hash of `user_id + system[:200] + first_user_msg[:300]` — uninstrumented clients (Claude Code, Codex CLI, etc.)

**Startup feedback:**

```
  Obs. Memory:  ENABLED

Observable Memory:
  - Observer model:  gpt-4o-mini
  - Reflector model: gpt-4o-mini
  - Storage:         /var/data/observations.db
  - Thresholds:      message=0.25, observation=0.35
```

See [Observable Memory](observable-memory.md) for full configuration details and examples.

### LLMLingua ML Compression (Opt-In)

When enabled, the proxy uses Microsoft's LLMLingua-2 model for ML-based token compression:

```bash
headroom proxy --llmlingua
```

**How it works:**
- LLMLinguaCompressor is added to the transform pipeline (before RollingWindow)
- Automatically detects content type (JSON, code, text) and adjusts compression
- Stores original content in CCR for retrieval if needed

**Startup feedback:**

```
# When enabled and available:
LLMLingua: ENABLED  (device=cuda, rate=0.3)

# When installed but not enabled (helpful hint):
LLMLingua: available (enable with --llmlingua for ML compression)

# When enabled but not installed:
WARNING: LLMLingua requested but not installed. Install with: pip install headroom-ai[llmlingua]
```

**Why opt-in?**
| Concern | Default Proxy | With LLMLingua |
|---------|---------------|----------------|
| Dependencies | ~50MB | +2GB (torch, transformers) |
| Cold start | <1s | 10-30s (model load) |
| Memory | ~100MB | +1GB (model in RAM) |
| Overhead | <5ms | 50-200ms per request |

Enable LLMLingua when maximum compression justifies the resource cost.

### Semantic Caching

The proxy caches responses for repeated queries:

- LRU eviction with configurable max entries
- TTL-based expiration
- Cache key based on message content hash

### Rate Limiting

Token bucket rate limiting protects against runaway costs:

- Configurable requests per minute
- Configurable tokens per minute
- Per-API-key tracking

### Cost Tracking

Track spending and enforce budgets:

- Real-time cost estimation
- Budget periods: hourly, daily, monthly
- Automatic request rejection when over budget

### Prometheus Metrics

Export metrics for monitoring:

```
headroom_requests_total
headroom_tokens_saved_total
headroom_cost_usd_total
headroom_latency_ms_sum
```

## Configuration via Environment

```bash
export HEADROOM_HOST=0.0.0.0
export HEADROOM_PORT=8787
export HEADROOM_BUDGET=100.0
export OPENAI_TARGET_API_URL=https://custom.openai.endpoint.com
headroom proxy
```

## Running in Production

For production deployments:

```bash
# Use a process manager
pip install gunicorn

# Run with gunicorn
gunicorn headroom.proxy.server:app \
  --workers 4 \
  --bind 0.0.0.0:8787 \
  --worker-class uvicorn.workers.UvicornWorker
```

Or with Docker:

```dockerfile
FROM python:3.11-slim
RUN pip install headroom[proxy]
EXPOSE 8787
CMD ["headroom", "proxy", "--host", "0.0.0.0"]
```
