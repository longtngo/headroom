"""Tests for the Observer agent — prompt building and output parsing."""
from __future__ import annotations

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
