"""ObservableMemoryProcessor — main entry point for Observable Memory.

Orchestrates the observe/reflect cycle:
1. validate_context_window — warn/error if model context is too small
2. observe — determine if threshold is met, run Observer LLM, maybe run Reflector
3. get_observations — retrieve stored observations to inject into prompts
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from .reflector import (
    build_reflector_prompt,
    build_reflector_system_prompt,
    parse_reflector_output,
    validate_compression,
)
from .store import InMemoryObservationStore, ObservationStore
from .token_counter import count_string
from .types import LLMProvider, ObservableMemoryConfig
from .worker import OMWorker

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when Observable Memory cannot operate safely with the given config."""


class ObservableMemoryProcessor:
    """Main entry point for Observable Memory.

    Wires together the OMWorker (Observer) and Reflector into a single
    interface that the host application calls per-turn.

    Args:
        config: ObservableMemoryConfig controlling all OM behaviour.
        llm: LLMProvider for Observer and Reflector calls.
            Required when config.enabled=True, ignored otherwise.
        store: ObservationStore for persistence.
            Defaults to InMemoryObservationStore (ephemeral).
    """

    def __init__(
        self,
        config: ObservableMemoryConfig | None = None,
        llm: LLMProvider | None = None,
        store: ObservationStore | None = None,
    ) -> None:
        self._config = config or ObservableMemoryConfig()
        self._store = store or InMemoryObservationStore()

        if self._config.enabled and llm is None:
            raise ValueError(
                "LLMProvider is required when ObservableMemoryConfig.enabled=True. "
                "Provide an LLMProvider or set enabled=False."
            )

        self._llm = llm
        self._worker: OMWorker | None = None
        if llm is not None:
            self._worker = OMWorker(llm, self._store, self._config)

    def validate_context_window(self, context_window: int) -> None:
        """Check whether the model's context window is large enough for OM.

        - Warning: context < 2x min_context_window (OM overhead is significant)
        - ConfigurationError: context < min_context_window (unsafe to run OM)

        Args:
            context_window: Model's total context window in tokens.

        Raises:
            ConfigurationError: If context_window < config.min_context_window.
        """
        min_win = self._config.min_context_window

        if context_window < min_win:
            raise ConfigurationError(
                f"Observable Memory requires a context window of at least {min_win} tokens, "
                f"but model reports {context_window} tokens. "
                f"Disable OM or use a model with a larger context window."
            )

        if context_window < 2 * min_win:
            logger.warning(
                "Observable Memory: model context window (%d tokens) is small. "
                "OM overhead may consume a significant fraction of the context. "
                "Consider using a model with a context window >= %d tokens.",
                context_window,
                2 * min_win,
            )

    async def observe(
        self,
        thread_id: str,
        messages: list[dict[str, Any]],
        model: str,
        context_window: int,
        current_token_count: int | None = None,
        abort: Any = None,
    ) -> None:
        """Run one observe/reflect cycle for new messages.

        Steps:
        1. Skip if OM is disabled.
        2. Check message_threshold_ratio — skip if not enough context is used.
        3. Run Observer LLM via OMWorker.
        4. Check observation_threshold_ratio — run Reflector if observations are large.

        Args:
            thread_id: Conversation/thread identifier.
            messages: New messages to observe.
            model: Model identifier for token counting and LLM calls.
            context_window: Model's total context window in tokens.
            current_token_count: Current total tokens in the conversation.
                If None, the threshold check is skipped and observation always runs.
            abort: Optional asyncio.Event for cancellation.
        """
        if not self._config.enabled or self._worker is None:
            return

        # Check message threshold — only when a full conversation token count is supplied.
        # When current_token_count is None the caller did not provide context usage data,
        # so we always proceed with observation rather than making a potentially wrong
        # estimate from only the new messages.
        if current_token_count is not None:
            usage_ratio = current_token_count / max(context_window, 1)
            if usage_ratio < self._config.message_threshold_ratio:
                logger.debug(
                    "OM.observe skipped: usage_ratio=%.2f < threshold=%.2f for thread=%s",
                    usage_ratio,
                    self._config.message_threshold_ratio,
                    thread_id,
                )
                return

        # Run Observer
        abort_event = abort if isinstance(abort, asyncio.Event) else None
        await self._worker.observe(thread_id, messages, model=model, abort=abort_event)

        # Check reflection threshold
        await self._maybe_reflect(thread_id, model, context_window)

    async def _maybe_reflect(
        self,
        thread_id: str,
        model: str,
        context_window: int,
    ) -> None:
        """Trigger Reflector if observations exceed the observation threshold."""
        observations = await self._store.load(thread_id)
        if not observations:
            return

        obs_tokens = count_string(observations, model)
        obs_threshold = int(context_window * self._config.observation_threshold_ratio)

        if obs_tokens < obs_threshold:
            return

        logger.info(
            "OM.reflect triggered: obs_tokens=%d >= threshold=%d for thread=%s",
            obs_tokens,
            obs_threshold,
            thread_id,
        )

        await self._run_reflector(thread_id, observations, model, obs_threshold)

    async def _run_reflector(
        self,
        thread_id: str,
        observations: str,
        model: str,
        target_threshold: int,
        max_attempts: int = 4,
    ) -> None:
        """Run the Reflector LLM with escalating compression levels."""
        assert self._llm is not None

        reflector_model = self._config.reflector_model or model
        system_prompt = build_reflector_system_prompt(instruction=self._config.instruction)

        for attempt in range(max_attempts):
            compression_level = min(attempt, 3)
            user_prompt = build_reflector_prompt(
                observations,
                compression_level=compression_level,
            )

            try:
                raw_output = await self._llm.complete(system_prompt, user_prompt, reflector_model)
            except Exception as exc:
                logger.exception("OM.reflect LLM call failed (attempt %d): %s", attempt, exc)
                return

            result = parse_reflector_output(raw_output)

            if result.degenerate:
                logger.warning(
                    "OM.reflect: degenerate output on attempt %d for thread=%s",
                    attempt,
                    thread_id,
                )
                continue

            reflected_tokens = count_string(result.observations, model)
            result.token_count = reflected_tokens

            if validate_compression(reflected_tokens, target_threshold):
                await self._store.save(thread_id, result.observations)
                logger.info(
                    "OM.reflect: compressed %d → %d tokens for thread=%s",
                    count_string(observations, model),
                    reflected_tokens,
                    thread_id,
                )
                return

            logger.info(
                "OM.reflect: attempt %d did not compress enough (%d tokens, target=%d). "
                "Escalating compression.",
                attempt,
                reflected_tokens,
                target_threshold,
            )

        # All attempts failed — keep the original observations
        logger.error(
            "OM.reflect: all %d compression attempts failed for thread=%s. "
            "Keeping original observations.",
            max_attempts,
            thread_id,
        )

    async def get_observations(self, thread_id: str) -> str | None:
        """Retrieve stored observations for a thread.

        Call this before each LLM turn to inject memory into the system prompt.

        Args:
            thread_id: Conversation/thread identifier.

        Returns:
            Observations string, or None if no observations exist.
        """
        return await self._store.load(thread_id)

    async def clear_observations(self, thread_id: str) -> None:
        """Delete all observations for a thread.

        Args:
            thread_id: Conversation/thread identifier.
        """
        await self._store.delete(thread_id)
