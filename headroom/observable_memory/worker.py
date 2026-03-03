"""OMWorker — async background observation worker for Observable Memory.

Manages the observe/reflect cycle:
1. Load existing observations for the thread from the store
2. Call the Observer LLM to extract new observations from recent messages
3. Append new observations to the store
4. (Reflect step is triggered by ObservableMemoryProcessor when threshold exceeded)

Includes:
- CircuitBreaker: stops hammering the LLM after repeated failures
- Abort signal: asyncio.Event-based cancellation (analogous to AbortSignal in JS)
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from .observer import build_observer_prompt, build_observer_system_prompt, parse_observer_output
from .store import ObservationStore
from .types import LLMProvider, ObservableMemoryConfig

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """Simple circuit breaker to prevent LLM spam on repeated failures.

    States:
    - Closed (normal): requests pass through
    - Open (tripped): requests are dropped until reset

    Args:
        failure_threshold: Number of consecutive failures before opening.
        reset_after_seconds: Seconds after opening before allowing one retry.
    """

    def __init__(self, failure_threshold: int = 5, reset_after_seconds: float = 60.0) -> None:
        self._threshold = failure_threshold
        self._reset_after = reset_after_seconds
        self._failures = 0
        self._opened_at: float | None = None

    @property
    def is_open(self) -> bool:
        """True if the circuit is open (requests should be dropped)."""
        if self._failures < self._threshold:
            return False
        if self._opened_at is not None:
            elapsed = time.monotonic() - self._opened_at
            if elapsed >= self._reset_after:
                # Allow one retry (half-open state: reset failures by 1)
                self._failures = self._threshold - 1
                self._opened_at = None
                return False
        return True

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self._threshold and self._opened_at is None:
            self._opened_at = time.monotonic()
            logger.warning(
                "OMWorker circuit breaker OPEN after %d consecutive failures. "
                "Observation disabled for %ds.",
                self._failures,
                int(self._reset_after),
            )

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None


class OMWorker:
    """Async worker that runs the Observer step for a single session.

    Usage:
        worker = OMWorker(llm, store, config, session_id="my-session")
        await worker.observe("thread-id", messages, model="gpt-4o")

    Args:
        llm: LLMProvider for making Observer/Reflector calls.
        store: ObservationStore for persisting observations.
        config: ObservableMemoryConfig controlling models and thresholds.
        session_id: Unique session identifier. Defaults to a new UUID.
    """

    def __init__(
        self,
        llm: LLMProvider,
        store: ObservationStore,
        config: ObservableMemoryConfig,
        session_id: str | None = None,
    ) -> None:
        self._llm = llm
        self._store = store
        self._config = config
        self.session_id = session_id or str(uuid.uuid4())
        self._circuit_breaker = CircuitBreaker()
        self._lock = asyncio.Lock()

    async def observe(
        self,
        thread_id: str,
        messages: list[dict[str, Any]],
        model: str,
        abort: asyncio.Event | None = None,
    ) -> None:
        """Run the Observer step for new messages in a thread.

        Loads existing observations, calls the Observer LLM, and saves
        the updated observations. Skips silently if:
        - The circuit breaker is open
        - The abort event is set

        Args:
            thread_id: Conversation/thread identifier.
            messages: New messages to observe.
            model: Model identifier to pass to the Observer.
            abort: Optional asyncio.Event. If set before or during observation,
                   the operation is cancelled and nothing is saved.
        """
        # Check abort signal first
        if abort is not None and abort.is_set():
            logger.debug("OMWorker.observe aborted (abort event set) for thread=%s", thread_id)
            return

        # Check circuit breaker
        if self._circuit_breaker.is_open:
            logger.debug("OMWorker.observe skipped (circuit open) for thread=%s", thread_id)
            return

        async with self._lock:
            try:
                await self._do_observe(thread_id, messages, model, abort)
            except asyncio.CancelledError:
                logger.debug("OMWorker.observe cancelled for thread=%s", thread_id)
                raise
            except Exception as exc:
                self._circuit_breaker.record_failure()
                logger.exception(
                    "OMWorker.observe failed for thread=%s: %s", thread_id, exc
                )

    async def _do_observe(
        self,
        thread_id: str,
        messages: list[dict[str, Any]],
        model: str,
        abort: asyncio.Event | None,
    ) -> None:
        """Internal: run the Observer LLM and persist results."""
        # Check abort again after acquiring lock
        if abort is not None and abort.is_set():
            return

        # Load existing observations for context
        existing = await self._store.load(thread_id)

        # Resolve model
        observer_model = self._config.observer_model or model

        # Build prompts
        system_prompt = build_observer_system_prompt(instruction=self._config.instruction)
        user_prompt = build_observer_prompt(
            existing_observations=existing,
            messages_to_observe=messages,
        )

        # Call Observer LLM
        raw_output = await self._llm.complete(system_prompt, user_prompt, observer_model)

        # Check abort after potentially slow LLM call
        if abort is not None and abort.is_set():
            logger.debug(
                "OMWorker.observe aborted after LLM call for thread=%s", thread_id
            )
            return

        # Parse output
        result = parse_observer_output(raw_output)

        if result.degenerate:
            self._circuit_breaker.record_failure()
            logger.warning(
                "OMWorker detected degenerate observer output for thread=%s. "
                "Discarding and recording failure.",
                thread_id,
            )
            return

        # Append new observations to existing
        if result.observations:
            if existing:
                combined = f"{existing}\n{result.observations}"
            else:
                combined = result.observations
            await self._store.save(thread_id, combined)
            self._circuit_breaker.record_success()
            logger.debug(
                "OMWorker saved %d chars of observations for thread=%s",
                len(combined),
                thread_id,
            )
        else:
            logger.debug(
                "OMWorker: observer returned empty observations for thread=%s", thread_id
            )
