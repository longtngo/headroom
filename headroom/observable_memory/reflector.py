"""Reflector LLM agent for Observable Memory.

Ported from @mastra/memory reflector-agent.ts.
The Reflector consolidates growing observations into a compressed, coherent
memory that serves as the assistant's complete record of past interactions.
"""
from __future__ import annotations

import re

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
- Collapse verbose multi-line tool sequences into a single outcome-focused sentence

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
