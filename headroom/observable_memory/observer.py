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
_TRUNCATION_SUFFIX = " … [truncated]"


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

    from datetime import datetime, timezone  # stdlib, deferred to keep module-level imports minimal

    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "unknown")
        role_display = role.capitalize()

        # Format timestamp if present
        timestamp_str = ""
        created_at = msg.get("created_at")
        if created_at:
            try:
                if isinstance(created_at, str):
                    try:
                        dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    except ValueError:
                        # Try parsing as unix epoch string (e.g., "1733322600.0")
                        dt = datetime.fromtimestamp(float(created_at), tz=timezone.utc)
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
            lines[i] = line[:_MAX_OBSERVATION_LINE_CHARS - len(_TRUNCATION_SUFFIX)] + _TRUNCATION_SUFFIX
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
