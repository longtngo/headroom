"""Core types for Observable Memory."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


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
