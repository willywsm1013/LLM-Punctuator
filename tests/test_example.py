"""Tests for example.py text cleanup."""

import pytest

from example import clean_zh_text


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
        assert clean_zh_text(text) == expected
