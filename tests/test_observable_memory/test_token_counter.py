"""Tests for the Observable Memory token counter."""
from __future__ import annotations

from headroom.observable_memory.token_counter import count_string, get_encoder


def test_count_string_basic():
    """Basic token counting works."""
    count = count_string("Hello world")
    assert count > 0
    assert isinstance(count, int)


def test_count_string_empty():
    """Empty string returns 0."""
    assert count_string("") == 0


def test_count_string_model_variants():
    """Different model names all return a positive count for the same text."""
    text = "The quick brown fox jumps over the lazy dog"
    for model in ["gpt-4o", "gpt-4", "claude-opus-4-6", "unknown-model"]:
        count = count_string(text, model)
        assert count > 0, f"Expected > 0 tokens for model={model}"


def test_count_string_longer_text_has_more_tokens():
    short = count_string("Hello")
    long = count_string("Hello world this is a longer sentence with more tokens")
    assert long > short


def test_get_encoder_singleton():
    """get_encoder returns the same object on repeated calls."""
    enc1 = get_encoder("gpt-4o")
    enc2 = get_encoder("gpt-4o")
    assert enc1 is enc2


def test_get_encoder_different_models_cached_separately():
    enc_4o = get_encoder("gpt-4o")
    enc_4 = get_encoder("gpt-4")
    # Different encoding names → different encoder objects
    # (o200k_base vs cl100k_base)
    assert enc_4o is not enc_4


def test_count_string_special_tokens():
    """Special tokens in input do not raise; they are counted."""
    count = count_string("<|endoftext|>")
    assert isinstance(count, int)
    assert count >= 1
