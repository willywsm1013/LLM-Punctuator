"""Tests for benchmark evaluation."""

from pathlib import Path

import pytest

from llm_punctuator.benchmark import (
    compute_metrics,
    extract_punctuation_labels,
    load_file_pairs,
    produce_markdown_table,
)


class TestExtractPunctuationLabels:
    """Test extracting punctuation labels from punctuated text."""

    def test_simple_sentence_with_comma_and_period(self) -> None:
        text = "今天天氣很好，適合出門。"
        plain, labels = extract_punctuation_labels(text)

        assert plain == "今天天氣很好適合出門"
        # Labels: one per gap between chars + one after last char
        # 今_天_天_氣_很_好_適_合_出_門
        #                  ，            。
        expected_labels = [None, None, None, None, None, "，", None, None, None, "。"]
        assert labels == expected_labels

    def test_question_mark(self) -> None:
        text = "你好嗎？"
        plain, labels = extract_punctuation_labels(text)

        assert plain == "你好嗎"
        assert labels == [None, None, "？"]

    def test_no_punctuation(self) -> None:
        text = "今天天氣很好"
        plain, labels = extract_punctuation_labels(text)

        assert plain == "今天天氣很好"
        assert labels == [None, None, None, None, None, None]

    def test_multiple_punctuation_types(self) -> None:
        text = "你好！今天天氣很好，要出門嗎？"
        plain, labels = extract_punctuation_labels(text)

        assert plain == "你好今天天氣很好要出門嗎"
        #                 你    好    今   天   天   氣   很   好    要   出   門   嗎
        assert labels == [None, "！", None, None, None, None, None, "，", None, None, None, "？"]

    def test_empty_string(self) -> None:
        text = ""
        plain, labels = extract_punctuation_labels(text)

        assert plain == ""
        assert labels == []

    def test_consecutive_punctuation_keeps_first(self) -> None:
        # Edge case: two punctuation marks in a row (unlikely but handle gracefully)
        text = "你好！？世界"
        plain, labels = extract_punctuation_labels(text)

        assert plain == "你好世界"
        # After 好: ！, then ？ also after 好 — keep both? No, each gap gets one label.
        # Actually ！？ are both after 好, so we should keep the first one.
        assert labels == [None, "！", None, None]


class TestComputeMetrics:
    """Test Precision/Recall/F1 computation."""

    def test_perfect_match(self) -> None:
        ref_labels = [None, "，", None, None, "。"]
        pred_labels = [None, "，", None, None, "。"]

        metrics = compute_metrics(ref_labels, pred_labels)

        assert metrics["overall"]["precision"] == pytest.approx(1.0)
        assert metrics["overall"]["recall"] == pytest.approx(1.0)
        assert metrics["overall"]["f1"] == pytest.approx(1.0)

    def test_no_predictions(self) -> None:
        ref_labels = [None, "，", None, None, "。"]
        pred_labels = [None, None, None, None, None]

        metrics = compute_metrics(ref_labels, pred_labels)

        assert metrics["overall"]["recall"] == pytest.approx(0.0)
        # Precision is undefined (0 predictions), treat as 0.0
        assert metrics["overall"]["precision"] == pytest.approx(0.0)

    def test_all_false_positives(self) -> None:
        ref_labels = [None, None, None]
        pred_labels = [None, "，", None]

        metrics = compute_metrics(ref_labels, pred_labels)

        assert metrics["overall"]["precision"] == pytest.approx(0.0)
        # Recall is undefined (0 reference), treat as 0.0
        assert metrics["overall"]["recall"] == pytest.approx(0.0)

    def test_partial_match_with_per_punctuation_breakdown(self) -> None:
        # ref: 今天，天氣。很好？
        ref_labels = [None, "，", None, "。", None, "？"]
        # pred: 今天，天氣，很好？  (句號 predicted as 逗號)
        pred_labels = [None, "，", None, "，", None, "？"]

        metrics = compute_metrics(ref_labels, pred_labels)

        # Overall: TP=2 (，at pos1, ？at pos5), FP=1 (，at pos3), FN=1 (。at pos3)
        assert metrics["overall"]["precision"] == pytest.approx(2 / 3)
        assert metrics["overall"]["recall"] == pytest.approx(2 / 3)

        # Per-punctuation: ，
        assert metrics["，"]["precision"] == pytest.approx(1 / 2)  # 1 TP, 1 FP
        assert metrics["，"]["recall"] == pytest.approx(1.0)  # 1 TP, 0 FN

        # Per-punctuation: 。
        assert metrics["。"]["precision"] == pytest.approx(0.0)  # 0 TP, 0 FP (no 。 predicted)
        assert metrics["。"]["recall"] == pytest.approx(0.0)  # 0 TP, 1 FN

        # Per-punctuation: ？
        assert metrics["？"]["precision"] == pytest.approx(1.0)
        assert metrics["？"]["recall"] == pytest.approx(1.0)

    def test_mismatched_lengths_raises_error(self) -> None:
        with pytest.raises(ValueError, match="length"):
            compute_metrics([None, "，"], [None])

    def test_both_empty(self) -> None:
        metrics = compute_metrics([], [])

        assert metrics["overall"]["precision"] == pytest.approx(0.0)
        assert metrics["overall"]["recall"] == pytest.approx(0.0)
        assert metrics["overall"]["f1"] == pytest.approx(0.0)


class TestLoadFilePairs:
    """Test loading and pairing files from directories."""

    def test_matching_files_are_paired(self, tmp_path: Path) -> None:
        ref_dir = tmp_path / "reference"
        out_dir = tmp_path / "output"
        ref_dir.mkdir()
        out_dir.mkdir()

        (ref_dir / "01_news.txt").write_text("你好，世界。", encoding="utf-8")
        (out_dir / "01_news.txt").write_text("你好，世界。", encoding="utf-8")

        pairs = load_file_pairs(ref_dir, out_dir)

        assert len(pairs) == 1
        assert pairs[0][0] == ref_dir / "01_news.txt"
        assert pairs[0][1] == out_dir / "01_news.txt"

    def test_missing_output_file_raises_error(self, tmp_path: Path) -> None:
        ref_dir = tmp_path / "reference"
        out_dir = tmp_path / "output"
        ref_dir.mkdir()
        out_dir.mkdir()

        (ref_dir / "01_news.txt").write_text("你好，世界。", encoding="utf-8")

        with pytest.raises(FileNotFoundError, match="01_news.txt"):
            load_file_pairs(ref_dir, out_dir)

    def test_files_sorted_by_name(self, tmp_path: Path) -> None:
        ref_dir = tmp_path / "reference"
        out_dir = tmp_path / "output"
        ref_dir.mkdir()
        out_dir.mkdir()

        for name in ["03_conv.txt", "01_news.txt", "02_speech.txt"]:
            (ref_dir / name).write_text("test", encoding="utf-8")
            (out_dir / name).write_text("test", encoding="utf-8")

        pairs = load_file_pairs(ref_dir, out_dir)

        assert [p[0].name for p in pairs] == ["01_news.txt", "02_speech.txt", "03_conv.txt"]


class TestProduceMarkdownTable:
    """Test markdown table output."""

    def test_table_has_correct_format(self) -> None:
        metrics = {
            "overall": {"precision": 0.85, "recall": 0.83, "f1": 0.84},
            "，": {"precision": 0.82, "recall": 0.80, "f1": 0.81},
            "。": {"precision": 0.90, "recall": 0.88, "f1": 0.89},
        }

        table = produce_markdown_table(metrics)

        lines = table.strip().split("\n")
        assert len(lines) == 5  # header + separator + 3 data rows
        assert "Overall" in lines[0]
        assert "，" in lines[0]
        assert "。" in lines[0]
        assert "Precision" in lines[2]
        assert "Recall" in lines[3]
        assert "F1-score" in lines[4]

    def test_table_is_valid_markdown(self) -> None:
        metrics = {
            "overall": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
        }

        table = produce_markdown_table(metrics)

        lines = table.strip().split("\n")
        # Every line should have pipes
        for line in lines:
            assert line.startswith("|")
            assert line.endswith("|")
        # Separator line should have dashes
        assert all(c in "|-: " for c in lines[1])
