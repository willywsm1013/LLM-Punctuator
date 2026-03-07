"""Evaluate punctuation restoration quality against benchmark references.

Usage:
    python scripts/evaluate_benchmark.py --model-output <directory>

The script compares model output files against reference files in data/benchmark/reference/.
Files are matched by name. Both reference and output files should contain punctuated text.

Output: Markdown table with Precision, Recall, F1-score (overall + per-punctuation).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from llm_punctuator.benchmark import (
    compute_metrics,
    extract_punctuation_labels,
    load_file_pairs,
    produce_markdown_table,
)

BENCHMARK_DIR = Path(__file__).resolve().parent.parent / "data" / "benchmark"


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate punctuation restoration quality against benchmark references.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        required=True,
        help="Directory containing model output files (matched by name with reference files).",
    )
    return parser.parse_args()


def main() -> int:
    """Run benchmark evaluation."""
    args = get_args()
    output_dir: Path = args.model_output
    reference_dir = BENCHMARK_DIR / "reference"

    if not output_dir.is_dir():
        print(f"Error: Not a directory: {output_dir}", file=sys.stderr)
        return 1

    if not reference_dir.is_dir():
        print(f"Error: Reference directory not found: {reference_dir}", file=sys.stderr)
        return 1

    try:
        pairs = load_file_pairs(reference_dir, output_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if not pairs:
        print("Error: No reference files found in benchmark directory.", file=sys.stderr)
        return 1

    all_ref_labels: list[str | None] = []
    all_pred_labels: list[str | None] = []

    for ref_path, out_path in pairs:
        try:
            ref_text = ref_path.read_text(encoding="utf-8")
            out_text = out_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            print(f"Error: Encoding error in {e.reason}: {ref_path}", file=sys.stderr)
            return 1

        ref_plain, ref_labels = extract_punctuation_labels(ref_text)
        out_plain, out_labels = extract_punctuation_labels(out_text)

        if ref_plain != out_plain:
            print(
                f"Error: Mismatch between reference and output files: "
                f"plain text differs in {ref_path.name}",
                file=sys.stderr,
            )
            return 1

        all_ref_labels.extend(ref_labels)
        all_pred_labels.extend(out_labels)

    metrics = compute_metrics(all_ref_labels, all_pred_labels)
    table = produce_markdown_table(metrics)
    print(table)

    return 0


if __name__ == "__main__":
    sys.exit(main())
