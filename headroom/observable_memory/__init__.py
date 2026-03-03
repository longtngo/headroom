"""Observable Memory — proactive background compression of message history.

Install: pip install "headroom-ai[observable-memory]"

Quick start:
    from headroom.observable_memory import ObservableMemoryProcessor, ObservableMemoryConfig

    # Provide your own LLMProvider implementation
    class MyLLM:
        async def complete(self, system: str, prompt: str, model: str) -> str:
            # call your preferred LLM API
            ...
        def count_tokens(self, text: str, model: str) -> int:
            # count tokens (use tiktoken or your model's tokenizer)
            ...

    proc = ObservableMemoryProcessor(
        config=ObservableMemoryConfig(enabled=True, db_path="/tmp/memory.db"),
        llm=MyLLM(),
    )

    # On each turn, pass new messages and retrieve observations for the system prompt
    await proc.observe(thread_id="conv-1", messages=messages, model="gpt-4o", context_window=128_000)
    observations = await proc.get_observations("conv-1")
    if observations:
        system_prompt = f"{base_system_prompt}\\n\\n<memory>\\n{observations}\\n</memory>"
"""

from .processor import ConfigurationError, ObservableMemoryProcessor
from .store import InMemoryObservationStore, ObservationStore, SQLiteObservationStore
from .token_counter import count_string
from .types import LLMProvider, ObservableMemoryConfig, ObserverResult, ReflectorResult

__all__ = [
    # Main entry point
    "ObservableMemoryProcessor",
    # Configuration
    "ObservableMemoryConfig",
    # Types
    "LLMProvider",
    "ObserverResult",
    "ReflectorResult",
    # Store implementations
    "ObservationStore",
    "InMemoryObservationStore",
    "SQLiteObservationStore",
    # Utilities
    "count_string",
    # Errors
    "ConfigurationError",
]
