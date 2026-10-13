"""Tests for example.py."""

import runpy
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from example import clean_zh_text

EXAMPLE_PATH = Path(__file__).resolve().parent.parent / "example.py"


class FullStopPunctuator:
    """Fake punctuator that ends each text with a full stop."""

    def __init__(self, model_name_or_path: str) -> None:
        """Skip loading a model."""

    def add_punctuation(self, texts: list[str], **kwargs: object) -> list[str]:
        """Append a full stop to each text."""
        return [f"{text}。" for text in texts]


class TestCleanZhText:
    """Test line break and whitespace cleanup for Chinese input."""

    @pytest.mark.parametrize(
        "text, expected",
        [
            (
                "今天天氣很好\r\n我們去公園散步\n然後回家吃飯",
                "今天天氣很好我們去公園散步然後回家吃飯",
            ),
            ("今天 天氣  很好", "今天天氣很好"),
            ("他說hello\nworld", "他說hello world"),
            ("我覺得很 OK\nThank you 大家", "我覺得很 OK Thank you 大家"),
            ("今天天氣很好\n", "今天天氣很好"),
        ],
        ids=[
            "cjk_line_breaks",
            "cjk_spaces",
            "ascii_line_break",
            "mixed_line_break",
            "trailing_newline",
        ],
    )
    def test_clean_zh_text(self, text: str, expected: str) -> None:
        """Breaks and spaces between CJK characters go; a break between ASCII words becomes a space."""
        assert clean_zh_text(text) == expected


def test_text_option_prints_one_punctuated_text(capsys: pytest.CaptureFixture[str]) -> None:
    """--text prints the punctuated text itself, not a list holding it."""
    argv = ["example.py", "--text", "今天天氣很好"]
    with (
        patch.object(sys, "argv", argv),
        patch("llm_punctuator.punctuator.TransformersLLMPunctuator", FullStopPunctuator),
    ):
        runpy.run_path(str(EXAMPLE_PATH), run_name="__main__")

    assert capsys.readouterr().out == "今天天氣很好。\n"
