"""Run punctuation model on benchmark inputs and save outputs.

Usage:
    python scripts/run_benchmark.py --output-dir <directory> [--model Qwen/Qwen3-1.7B]

Reads reference files from data/benchmark/{category}/, strips punctuation to create
inputs, runs the model, and saves outputs to the specified directory mirroring
the category subdirectory structure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from llm_punctuator.benchmark import extract_punctuation_labels
from llm_punctuator.punctuator import TransformersLLMPunctuator

BENCHMARK_DIR = Path(__file__).resolve().parent.parent / "data" / "benchmark"


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run punctuation model on benchmark inputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save model output files.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help="Model name or path. Default: Qwen/Qwen3-1.7B",
    )
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        choices=["zh", "en"],
        default="zh",
        help="Language: zh or en. Default: zh",
    )
    parser.add_argument(
        "-c",
        "--chunk-size",
        type=int,
        default=50,
        help="Chunk size for processing. Default: 50",
    )
    return parser.parse_args()


def main() -> int:
    args = get_args()

    if not BENCHMARK_DIR.is_dir():
        print(f"Error: Benchmark directory not found: {BENCHMARK_DIR}", file=sys.stderr)
        return 1

    categories = sorted(d for d in BENCHMARK_DIR.iterdir() if d.is_dir())
    if not categories:
        print("Error: No category subdirectories found.", file=sys.stderr)
        return 1

    output_dir: Path = args.output_dir

    print(f"Loading model: {args.model}")
    punctuator = TransformersLLMPunctuator(args.model)

    for category_dir in categories:
        category = category_dir.name
        ref_files = sorted(category_dir.glob("*.txt"))
        if not ref_files:
            continue

        cat_output_dir = output_dir / category
        cat_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{category}] {len(ref_files)} files")
        for ref_file in ref_files:
            out_file = cat_output_dir / ref_file.name
            if out_file.exists():
                print(f"  Skipping: {ref_file.name} (output exists)")
                continue

            ref_text = ref_file.read_text(encoding="utf-8")
            plain_text, _ = extract_punctuation_labels(ref_text)

            print(f"  Processing: {ref_file.name} ({len(plain_text)} chars)")
            result = punctuator.add_punctuation(
                plain_text, language=args.language, chunk_size=args.chunk_size
            )

            out_file.write_text(result, encoding="utf-8")
            print(f"    -> {out_file}")

    print(f"\nDone. Outputs saved to: {output_dir}")
    print(
        f"Run evaluation with:\n  python scripts/evaluate_benchmark.py --model-output {output_dir}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
