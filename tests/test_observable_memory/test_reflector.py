"""Tests for the Reflector agent — compression and output parsing."""
from __future__ import annotations

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
