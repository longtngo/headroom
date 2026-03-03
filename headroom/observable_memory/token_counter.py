"""Token counting for Observable Memory.

Uses tiktoken (already a headroom core dependency) with a shared encoder cache.
Default encoding: o200k_base (used by GPT-4o and the Mastra JS port).
Unknown models fall back to cl100k_base.
"""
from __future__ import annotations

import tiktoken

# Encoder cache — shared across all OM calls in a process
_encoder_cache: dict[str, tiktoken.Encoding] = {}

# Models that use o200k_base encoding (GPT-4o family)
_O200K_MODELS = {
    "gpt-4o",
    "gpt-4o-mini",
    "o1",
    "o1-mini",
    "o1-preview",
    "o3",
    "o3-mini",
    "o4-mini",
}


def _get_encoding_name(model: str) -> str:
    """Map a model name to a tiktoken encoding name."""
    model_lower = model.lower()
    if model_lower in _O200K_MODELS:
        return "o200k_base"
    if "gpt-4" in model_lower or "gpt-3.5" in model_lower:
        return "cl100k_base"
    # Claude, Gemini, Llama, and unknown models: default to cl100k_base
    # (reasonable approximation; exact counts vary by model)
    return "cl100k_base"


def get_encoder(model: str = "gpt-4o") -> tiktoken.Encoding:
    """Get a cached tiktoken encoder for the given model.

    Args:
        model: Model name. Used to select the right encoding.

    Returns:
        Cached tiktoken.Encoding instance.
    """
    encoding_name = _get_encoding_name(model)
    if encoding_name not in _encoder_cache:
        _encoder_cache[encoding_name] = tiktoken.get_encoding(encoding_name)
    return _encoder_cache[encoding_name]


def count_string(text: str, model: str = "gpt-4o") -> int:
    """Count tokens in a string.

    Args:
        text: Text to count tokens for.
        model: Model name for encoding selection.

    Returns:
        Token count as an integer.
    """
    if not text:
        return 0
    encoder = get_encoder(model)
    return len(encoder.encode(text))
