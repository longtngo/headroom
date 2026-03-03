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
