"""Evaluate punctuation restoration quality against benchmark references.

Usage:
    python scripts/evaluate_benchmark.py --model-output <directory>

The script compares model output files against reference files in
data/benchmark/{category}/. Files are matched by category and name.
Both reference and output files should contain punctuated text.

Output: Per-category and overall markdown tables with Precision, Recall, F1-score.
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
        help="Directory containing model output files (matched by category and name).",
    )
    return parser.parse_args()


def _collect_labels(
    pairs: list[tuple[Path, Path]],
) -> tuple[list[str | None], list[str | None]] | str:
    """Extract and validate labels from file pairs.

    Returns (ref_labels, pred_labels) on success, or an error message string.
    """
    ref_labels: list[str | None] = []
    pred_labels: list[str | None] = []

    for ref_path, out_path in pairs:
        try:
            ref_text = ref_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            return f"Cannot decode reference file {ref_path}: {e}"
        try:
            out_text = out_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            return f"Cannot decode output file {out_path}: {e}"

        ref_plain, ref_lab = extract_punctuation_labels(ref_text.strip())
        out_plain, out_lab = extract_punctuation_labels(out_text.strip())

        if ref_plain != out_plain:
            return f"Mismatch between reference and output files: plain text differs in {ref_path}"

        ref_labels.extend(ref_lab)
        pred_labels.extend(out_lab)

    return ref_labels, pred_labels


def main() -> int:
    """Run benchmark evaluation."""
    args = get_args()
    output_dir: Path = args.model_output

    if not output_dir.is_dir():
        print(f"Error: Not a directory: {output_dir}", file=sys.stderr)
        return 1

    if not BENCHMARK_DIR.is_dir():
        print(f"Error: Benchmark directory not found: {BENCHMARK_DIR}", file=sys.stderr)
        return 1

    try:
        categorized_pairs = load_file_pairs(BENCHMARK_DIR, output_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if not categorized_pairs:
        print("Error: No category subdirectories found in benchmark directory.", file=sys.stderr)
        return 1

    all_ref_labels: list[str | None] = []
    all_pred_labels: list[str | None] = []

    for category, pairs in sorted(categorized_pairs.items()):
        result = _collect_labels(pairs)
        if isinstance(result, str):
            print(f"Error: {result}", file=sys.stderr)
            return 1

        cat_ref, cat_pred = result
        all_ref_labels.extend(cat_ref)
        all_pred_labels.extend(cat_pred)

        metrics = compute_metrics(cat_ref, cat_pred)
        print(f"### {category}\n")
        print(produce_markdown_table(metrics))

    if len(categorized_pairs) > 1:
        overall_metrics = compute_metrics(all_ref_labels, all_pred_labels)
        print("### overall\n")
        print(produce_markdown_table(overall_metrics))

    return 0


if __name__ == "__main__":
    sys.exit(main())
