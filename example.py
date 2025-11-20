"""Example script for using LLM Punctuator to add punctuation to text."""

import argparse
import logging
import re

from llm_punctuator.punctuator import TransformersLLMPunctuator

logging.basicConfig(level=logging.INFO)


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
        "-c", "--chunk_size", type=int, default=50, help="Chunk size for processing. Default: 50"
    )
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        choices=["zh", "en"],
        default="zh",
        help="Language: zh (Chinese) or en (English). Default: zh",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    # Load the punctuator (no language needed at initialization)
    punctuator = TransformersLLMPunctuator(args.model_name_or_path)

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            text = f.read()
    else:
        text = args.text
    logging.info(f"Original text: {text}")

    # Remove space between chinese characters (only for Chinese)
    if args.language == "zh":
        clean_text = re.sub(r"(?<=[\u4e00-\u9fa5])\s+(?=[\u4e00-\u9fa5])", "", text)
    else:
        clean_text = text

    logging.info(f"Clean text: {clean_text}")

    # Language and punctuations are now handled automatically by add_punctuation
    result = punctuator.add_punctuation(
        clean_text, language=args.language, chunk_size=args.chunk_size
    )

    print(result)
