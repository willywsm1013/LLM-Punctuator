"""Run punctuation model on benchmark inputs and save outputs.

Usage:
    python scripts/run_benchmark.py --output-dir <directory> [--model Qwen/Qwen3-1.7B]

Reads reference files from data/benchmark/reference/, strips punctuation to create
inputs, runs the model, and saves outputs to the specified directory.
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

    reference_dir = BENCHMARK_DIR / "reference"
    if not reference_dir.is_dir():
        print(f"Error: Reference directory not found: {reference_dir}", file=sys.stderr)
        return 1

    ref_files = sorted(reference_dir.glob("*.txt"))
    if not ref_files:
        print("Error: No reference files found.", file=sys.stderr)
        return 1

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
    punctuator = TransformersLLMPunctuator(args.model)

    for ref_file in ref_files:
        ref_text = ref_file.read_text(encoding="utf-8")
        plain_text, _ = extract_punctuation_labels(ref_text)

        print(f"Processing: {ref_file.name} ({len(plain_text)} chars)")
        result = punctuator.add_punctuation(
            plain_text, language=args.language, chunk_size=args.chunk_size
        )

        out_file = output_dir / ref_file.name
        out_file.write_text(result, encoding="utf-8")
        print(f"  -> {out_file}")

    print(f"\nDone. Outputs saved to: {output_dir}")
    print(
        f"Run evaluation with:\n  python scripts/evaluate_benchmark.py --model-output {output_dir}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
