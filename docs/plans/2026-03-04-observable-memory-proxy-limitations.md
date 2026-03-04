# Observable Memory: Proxy Limitations & Future Directions

**Date:** 2026-03-04

## Context

This document captures a design discussion about the fundamental limitations of headroom's
Observable Memory (OM) feature when used with agentic clients like Claude Code and Codex,
and how those limitations compare to agent-native memory architectures like Mastra.

---

## What OM Does (and Doesn't Do)

### What `inject_observations` actually does

`inject_observations()` **only appends** a `<memory>…</memory>` block to the system prompt.
It never removes, compresses, or replaces any messages from `body["messages"]`.

```
Request arrives with full history from Claude Code
         ↓
RollingWindow / IntelligentContextManager  →  may drop old messages
         ↓
SmartCrusher                               →  compresses large tool outputs
         ↓
inject_observations()                      →  ADDS <memory> block to system prompt
         ↓
Optimized (but larger) request sent to LLM
```

OM **increases** the token count of each request by the size of the memory block.
Token reduction comes from the transform pipeline (RollingWindow, SmartCrusher), not OM.

---

## The Proxy Boundary Problem

### Why OM cannot prevent Claude Code from compacting

Claude Code and Codex maintain a full in-process message array locally.
Every turn appends a new user + assistant message to that array.
The client compacts when its internal token counter (fed by API-reported usage) crosses
a threshold.

Headroom is a **proxy** — it sits between the client and the LLM API:

```
Claude Code local buffer  →  grows every turn, headroom cannot touch this
         ↓
headroom proxy            →  compresses what gets SENT to the LLM
         ↓
Anthropic API             →  sees compressed tokens, reports lower usage
         ↓
Claude Code               →  sees lower usage count → compaction is delayed
```

The delay is real: if the transform pipeline reduces tokens, API-reported usage is lower,
so Claude Code's compaction trigger fires later. But the local buffer still grows.
**Headroom cannot prevent compaction — only delay it.**

---

## Comparison: Mastra's Approach

Mastra ships "Mastra Code: a coding agent that never compacts." Their architecture:

- Two background agents (Observer + Reflector) compress messages into a rolling
  observation log — the same Observer/Reflector pattern as headroom's OM.
- When raw messages hit ~30k tokens, the observer compresses them into observations
  in the background (async buffering). When observations get large, the reflector
  garbage-collects them.
- Because Mastra **is** the agent runtime (not a proxy in front of it), it owns the
  message store. It can replace old messages with observations, keeping the context
  window genuinely stable.

| | Headroom | Mastra |
|---|---|---|
| Layer | Proxy (between client and API) | Agent framework (owns message store) |
| Can modify client's local buffer? | ❌ No | ✅ Yes |
| Can prevent compaction? | ❌ No — only delay | ✅ Yes |
| OM concept | Observer/Reflector pipeline | Same Observer/Reflector pipeline |

The OM concept is architecturally identical. The difference is purely where it runs.

---

## Claude Code Extension Mechanisms

Claude Code provides no mechanism for an external service to read or modify its
internal message buffer:

| Mechanism | Read messages? | Modify messages? |
|---|---|---|
| Hooks | ❌ Only `last_assistant_message` + `transcript_path` | ❌ No |
| MCP servers | ❌ No | ❌ No |
| `ANTHROPIC_BASE_URL` proxy (headroom today) | ✅ Sees full history per-request | ❌ Can't modify local copy |
| `transcript_path` (hook field) | ✅ Read-only JSONL | ❌ Unsafe to modify externally |

### The `PreCompact` hook — the most promising surface

Claude Code exposes a `PreCompact` hook that fires immediately before its built-in
compaction runs. This is the closest integration point for headroom's use case:

- Headroom's OM has already computed a high-quality observation summary.
- If `PreCompact` allows injecting a custom summary, Claude Code could use headroom's
  observations instead of running its own compaction.
- If `PreCompact` can signal "skip compaction", it could be suppressed entirely.

Whether the current `PreCompact` payload supports this is not yet documented.
**This is worth investigating / filing a feature request with Anthropic.**

---

## Paths to Eliminating Compaction

Three viable approaches, in order of effort:

1. **`PreCompact` hook integration** — lowest friction. Headroom injects its OM
   observations into the `PreCompact` hook output. Claude Code uses them as the
   compaction summary. Requires Anthropic to document/extend the hook payload.

2. **Claude Code exposes a conversation state MCP resource** — allows read/write
   access to the message buffer. Headroom could trim old messages and replace them
   with observations directly. Requires Anthropic to build this.

3. **Own the agent runtime (Mastra approach)** — build a Claude Code replacement
   that uses headroom's OM as its native memory layer. Complete control, maximum
   effort.

---

## References

- [Mastra: Announcing Mastra Code — never compacts](https://mastra.ai/blog/announcing-mastra-code)
- [Mastra: Observational Memory docs](https://mastra.ai/docs/memory/observational-memory)
- [Mastra: Observational Memory research (95% on LongMemEval)](https://mastra.ai/research/observational-memory)
- [VentureBeat: Observational memory cuts AI agent costs 10x](https://venturebeat.com/data/observational-memory-cuts-ai-agent-costs-10x-and-outscores-rag-on-long)
- [Claude Code Hooks Reference](https://code.claude.com/docs/en/hooks)
