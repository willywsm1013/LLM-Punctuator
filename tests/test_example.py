"""Tests for example.py."""

import json
import runpy
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from example import clean_zh_text

EXAMPLE_PATH = Path(__file__).resolve().parent.parent / "example.py"


class FullStopPunctuator:
    """Fake punctuator that ends each text with a full stop and keeps the last kwargs."""

    last_kwargs: dict[str, object] = {}

    def __init__(self, model_name_or_path: str) -> None:
        """Skip loading a model."""

    def add_punctuation(self, texts: list[str], **kwargs: object) -> list[str]:
        """Append a full stop to each text."""
        FullStopPunctuator.last_kwargs = kwargs
        return [f"{text}。" for text in texts]


class UnloadablePunctuator:
    """Fake punctuator whose model fails to load, like a missing -m path."""

    def __init__(self, model_name_or_path: str) -> None:
        """Fail the way a missing model path does."""
        raise OSError(f"{model_name_or_path} not found")


def run_example(argv: list[str], punctuator: type = FullStopPunctuator) -> None:
    """Run example.py as a script with argv and a fake punctuator."""
    with (
        patch.object(sys, "argv", ["example.py", *argv]),
        patch("llm_punctuator.punctuator.TransformersLLMPunctuator", punctuator),
    ):
        runpy.run_path(str(EXAMPLE_PATH), run_name="__main__")


def write_list(list_path: Path, lines: list[str]) -> Path:
    """Write a JSONL list file and return its path."""
    list_path.parent.mkdir(parents=True, exist_ok=True)
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return list_path


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


class TestListOption:
    """Test punctuating the texts in a JSONL list."""

    @pytest.fixture
    def texts_dir(self, tmp_path: Path) -> Path:
        """A directory holding three text files, one with a line break."""
        texts_dir = tmp_path / "texts"
        texts_dir.mkdir()
        (texts_dir / "a.txt").write_text("今天天氣\n很好", encoding="utf-8")
        (texts_dir / "b.txt").write_text("我們去公園", encoding="utf-8")
        (texts_dir / "c.txt").write_text("然後回家", encoding="utf-8")
        (texts_dir / "big5.txt").write_bytes("今天".encode("big5"))
        return texts_dir

    def test_list_prints_one_jsonl_line_per_entry_in_list_order(
        self, texts_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Each entry comes out as {"id", "text"} in list order, text cleaned like --file."""
        list_path = write_list(
            texts_dir / "list.jsonl",
            [
                '{"id": "003", "path": "c.txt"}',
                "",
                '{"id": "001", "path": "a.txt", "speaker": "ignored"}',
                '{"id": "002", "path": "b.txt"}',
            ],
        )

        run_example(["--list", str(list_path)])

        lines = capsys.readouterr().out.splitlines()
        assert [json.loads(line) for line in lines] == [
            {"id": "003", "text": "然後回家。"},
            {"id": "001", "text": "今天天氣很好。"},
            {"id": "002", "text": "我們去公園。"},
        ]

    def test_list_relative_path_is_relative_to_list_file_not_cwd(
        self,
        texts_dir: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A relative path resolves against the list file's directory when run from elsewhere."""
        list_path = write_list(
            tmp_path / "lists" / "list.jsonl", ['{"id": "a", "path": "../texts/a.txt"}']
        )
        other_dir = tmp_path / "elsewhere"
        other_dir.mkdir()
        monkeypatch.chdir(other_dir)

        run_example(["--list", str(list_path)])

        assert json.loads(capsys.readouterr().out) == {"id": "a", "text": "今天天氣很好。"}

    def test_single_entry_list_punctuates_same_as_file(
        self, texts_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--file and a one-line --list give the same punctuated text."""
        list_path = write_list(texts_dir / "list.jsonl", ['{"id": "a", "path": "a.txt"}'])
        run_example(["--file", str(texts_dir / "a.txt")])
        file_text = capsys.readouterr().out.rstrip("\n")

        run_example(["--list", str(list_path)])

        assert json.loads(capsys.readouterr().out)["text"] == file_text

    def test_list_passes_batch_size_to_add_punctuation(
        self, texts_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """-b reaches add_punctuation as batch_size."""
        list_path = write_list(texts_dir / "list.jsonl", ['{"id": "a", "path": "a.txt"}'])

        run_example(["--list", str(list_path), "-b", "2"])

        assert FullStopPunctuator.last_kwargs["batch_size"] == 2

    def test_list_without_batch_size_leaves_add_punctuation_default(
        self, texts_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Without -b, add_punctuation keeps its own batch_size default."""
        list_path = write_list(texts_dir / "list.jsonl", ['{"id": "a", "path": "a.txt"}'])

        run_example(["--list", str(list_path)])

        assert "batch_size" not in FullStopPunctuator.last_kwargs

    def test_list_output_file_ends_every_record_with_newline(self, texts_dir: Path) -> None:
        """-o gets the same bytes as stdout, so two output files concatenate into valid JSONL."""
        list_path = write_list(
            texts_dir / "list.jsonl",
            ['{"id": "a", "path": "a.txt"}', '{"id": "b", "path": "b.txt"}'],
        )
        output_path = texts_dir / "out.jsonl"

        run_example(["--list", str(list_path), "-o", str(output_path)])

        assert output_path.read_text(encoding="utf-8") == (
            '{"id": "a", "text": "今天天氣很好。"}\n{"id": "b", "text": "我們去公園。"}\n'
        )

    @pytest.mark.parametrize(
        "bad_line, message",
        [
            ('{"id": "b", "path": ', "invalid JSON"),
            ('{"path": "b.txt"}', '"id" and "path"'),
            ('{"id": "b"}', '"id" and "path"'),
            ('{"id": "a", "path": "b.txt"}', "duplicate id 'a'"),
            ('{"id": "b", "path": "missing.txt"}', "file not found"),
            ('{"id": "b", "path": "big5.txt"}', "not UTF-8"),
        ],
        ids=[
            "invalid_json",
            "missing_id",
            "missing_path",
            "duplicate_id",
            "missing_file",
            "non_utf8_file",
        ],
    )
    def test_list_error_exits_1_with_line_number_before_loading_model(
        self, texts_dir: Path, capsys: pytest.CaptureFixture[str], bad_line: str, message: str
    ) -> None:
        """A bad list exits 1 naming the line, even when the model cannot load."""
        list_path = write_list(
            texts_dir / "list.jsonl", ['{"id": "a", "path": "a.txt"}', "", bad_line]
        )

        with pytest.raises(SystemExit) as exc_info:
            run_example(["--list", str(list_path), "-m", "/no/such/model"], UnloadablePunctuator)

        assert exc_info.value.code == 1
        stderr = capsys.readouterr().err
        assert f"{list_path}:3:" in stderr
        assert message in stderr

    @pytest.mark.parametrize(
        "argv",
        [["--file", "a.txt"], ["--text", "今天"]],
        ids=["with_file", "with_text"],
    )
    def test_list_with_another_input_exits_2(self, argv: list[str]) -> None:
        """--list cannot be combined with --file or --text."""
        with pytest.raises(SystemExit) as exc_info:
            run_example(["--list", "list.jsonl", *argv])

        assert exc_info.value.code == 2

    def test_batch_size_below_1_exits_nonzero_with_stderr(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--batch_size 0 is rejected with a message."""
        with pytest.raises(SystemExit) as exc_info:
            run_example(["--text", "今天", "-b", "0"])

        assert exc_info.value.code != 0
        assert "--batch_size must be a positive integer, got 0" in capsys.readouterr().err
