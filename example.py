"""Example script for using LLM Punctuator to add punctuation to text."""

import argparse
import logging
import re
from pathlib import Path

from llm_punctuator.punctuator import TransformersLLMPunctuator


def get_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed command line arguments containing model path, text input,
        chunk size, and language settings.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help="Model name or path to the model. Default: Qwen/Qwen3-1.7B",
    )
    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument("--file", type=str, help="Path to the file")
    text_group.add_argument("--text", type=str, help="Text to punctuate")
    parser.add_argument(
        "-o", "--output_file", type=str, help="Path to the output file. Default: print to stdout"
    )
    parser.add_argument(
        "-c", "--chunk_size", type=int, default=50, help="Chunk size for processing. Default: 50"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=-1,
        help="Most mark positions to decide per forward pass; -1 for the whole chunk. Default: -1",
    )
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        choices=["zh", "en"],
        default="zh",
        help="Language: zh (Chinese) or en (English). Default: zh",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()
    return args


def clean_zh_text(text: str) -> str:
    """Remove line breaks and whitespace between Chinese characters.

    A line break becomes a space, so non-CJK tokens on either side stay separated.

    Args:
        text: Raw Chinese text, possibly spanning multiple lines.

    Returns:
        Single-line text with no whitespace between two CJK characters.
    """
    clean_text = text.replace("\r", " ").replace("\n", " ").strip()
    return re.sub(r"(?<=[\u4e00-\u9fa5])\s+(?=[\u4e00-\u9fa5])", "", clean_text)


if __name__ == "__main__":
    args = get_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    # Load the punctuator (no language needed at initialization)
    punctuator = TransformersLLMPunctuator(args.model_name_or_path)

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            text = f.read()
    else:
        text = args.text
    logging.info(f"Original text: {text}")

    if args.language == "zh":
        clean_text = clean_zh_text(text)
    else:
        clean_text = text

    logging.info(f"Clean text: {clean_text}")

    # Language and punctuations are now handled automatically by add_punctuation
    result = punctuator.add_punctuation(
        clean_text, language=args.language, chunk_size=args.chunk_size, k=args.k
    )

    if args.output_file is None:
        print(result)
    else:
        output_file = Path(args.output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(result, encoding="utf-8")
