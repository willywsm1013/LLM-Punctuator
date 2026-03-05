"""Tests for request/response schema models."""

import pytest

from llm_punctuator.schema import PunctuateRequest, PunctuateResponse


def test_punctuate_request_defaults() -> None:
    req = PunctuateRequest(text="hello world")
    assert req.text == "hello world"
    assert req.language == "zh"
    assert req.chunk_size == 50


def test_punctuate_request_custom() -> None:
    req = PunctuateRequest(text="hello", language="en", chunk_size=100)
    assert req.language == "en"
    assert req.chunk_size == 100


def test_punctuate_request_empty_text_rejected() -> None:
    with pytest.raises(Exception):
        PunctuateRequest(text="")


def test_punctuate_response() -> None:
    resp = PunctuateResponse(text="hello, world.", language="en", model="test-model")
    assert resp.text == "hello, world."
    assert resp.model == "test-model"
