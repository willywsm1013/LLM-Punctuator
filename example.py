"""Example script for using LLM Punctuator to add punctuation to text."""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

from llm_punctuator.asr_punctuator import ASR_MODELS, ASRPunctuator
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
    text_group.add_argument(
        "--list",
        type=str,
        help='Path to a JSONL list, one {"id": ..., "path": ...} per line; an ASR model also '
        'needs "audio". A relative path is relative to the list file. '
        "Output is JSONL in the same order",
    )
    parser.add_argument(
        "-o", "--output_file", type=str, help="Path to the output file. Default: print to stdout"
    )
    parser.add_argument(
        "-c",
        "--chunk_size",
        type=int,
        help="Chunk size for a text LLM. Default: 50. An ASR model does not take it",
    )
    parser.add_argument(
        "--k",
        type=int,
        help="Most mark positions to decide per forward pass; -1 for the whole chunk. "
        "Default: the default of add_punctuation",
    )
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        choices=["zh", "en"],
        default="zh",
        help="Language: zh (Chinese) or en (English). Default: zh",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        help="Most texts to decode together; only used with --list and a text LLM. "
        "Default: the default of add_punctuation",
    )
    parser.add_argument(
        "--audio",
        type=str,
        help="Audio file of the text; required with an ASR model and --file or --text",
    )
    parser.add_argument(
        "--context",
        type=str,
        help="Context template for an ASR model; each {text} becomes the text. "
        "Default: the default of add_punctuation",
    )
    parser.add_argument(
        "--mark_bonus_per_char",
        type=float,
        help="For an ASR model, add this times the characters since the last mark to every "
        "mark's score. See README. Default: the default of add_punctuation",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()
    if args.model_name_or_path in ASR_MODELS:
        if args.list is not None and args.audio is not None:
            parser.error('--audio does not apply to --list; put an "audio" field on each line')
        if args.list is None and args.audio is None:
            parser.error("--audio is required with an ASR model")
        if args.chunk_size is not None:
            parser.error("--chunk_size does not apply to an ASR model")
        if args.batch_size is not None:
            parser.error("--batch_size does not apply to an ASR model")
    elif args.audio is not None or args.context is not None:
        parser.error("--audio and --context need an ASR model")
    elif args.mark_bonus_per_char is not None:
        parser.error("--mark_bonus_per_char needs an ASR model")
    if args.batch_size is not None and args.batch_size < 1:
        parser.error(f"--batch_size must be a positive integer, got {args.batch_size}")
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


def read_list(list_path: Path, with_audio: bool) -> list[tuple[object, str, str | None]]:
    """Read a JSONL list of texts to punctuate.

    Args:
        list_path: JSONL file, one {"id": ..., "path": ...} object per line. Other fields and
            blank lines are ignored.
        with_audio: Whether each line must also name an existing "audio" file.

    Returns:
        (id, text, audio) per line in file order. audio is None without with_audio. A
        relative path or audio is resolved against the list file's directory.

    Raises:
        ValueError: If a line is not valid JSON, lacks id or a string path, repeats an earlier id, or
            points to a missing or non-UTF-8 file. With with_audio, also if a line lacks a string
            audio or its audio file is missing. The message names the line number.
    """
    entries = []
    seen_ids = set()
    with list_path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            where = f"{list_path}:{line_number}"
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{where}: invalid JSON: {e.msg}") from e
            if (
                not isinstance(item, dict)
                or "id" not in item
                or not isinstance(item.get("path"), str)
            ):
                raise ValueError(f'{where}: expected a JSON object with "id" and "path"')
            if item["id"] in seen_ids:
                raise ValueError(f"{where}: duplicate id {item['id']!r}")
            path = list_path.parent / item["path"]
            if not path.is_file():
                raise ValueError(f"{where}: file not found: {path}")
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError as e:
                raise ValueError(f"{where}: not UTF-8: {path}: {e.reason}") from e
            audio = resolve_audio(item, list_path, where) if with_audio else None
            seen_ids.add(item["id"])
            entries.append((item["id"], text, audio))
    return entries


def resolve_audio(item: dict, list_path: Path, where: str) -> str:
    """Return the existing audio file a list line names, or raise ValueError naming where."""
    if not isinstance(item.get("audio"), str):
        raise ValueError(f'{where}: expected "audio" with an ASR model')
    audio = list_path.parent / item["audio"]
    if not audio.is_file():
        raise ValueError(f"{where}: audio file not found: {audio}")
    return str(audio)


def preprocess(text: str, language: str) -> str:
    """Clean a text the way every input is cleaned before punctuation."""
    return clean_zh_text(text) if language == "zh" else text


if __name__ == "__main__":
    args = get_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    is_asr = args.model_name_or_path in ASR_MODELS
    ids = None
    audios = [args.audio]
    if args.list:
        try:
            entries = read_list(Path(args.list), with_audio=is_asr)
        except ValueError as e:
            print(f"Invalid list: {e}", file=sys.stderr)
            sys.exit(1)
        ids = [entry_id for entry_id, _, _ in entries]
        texts = [text for _, text, _ in entries]
        audios = [audio for _, _, audio in entries]
    elif args.file:
        texts = [Path(args.file).read_text(encoding="utf-8")]
    else:
        texts = [args.text]

    clean_texts = [preprocess(text, args.language) for text in texts]
    for text, clean_text in zip(texts, clean_texts, strict=True):
        logging.info(f"Original text: {text}")
        logging.info(f"Clean text: {clean_text}")

    # Language and punctuations are now handled automatically by add_punctuation
    options = {} if args.k is None else {"k": args.k}
    if is_asr:
        if args.context is not None:
            options["context"] = args.context
        if args.mark_bonus_per_char is not None:
            options["mark_bonus_per_char"] = args.mark_bonus_per_char
        asr_punctuator = ASRPunctuator(args.model_name_or_path)
        results = asr_punctuator.add_punctuation(
            clean_texts, audios, language=args.language, **options
        )
    else:
        if ids is not None and args.batch_size is not None:
            options["batch_size"] = args.batch_size
        # Load the punctuator (no language needed at initialization)
        punctuator = TransformersLLMPunctuator(args.model_name_or_path)
        chunk_size = 50 if args.chunk_size is None else args.chunk_size
        results = punctuator.add_punctuation(
            clean_texts, language=args.language, chunk_size=chunk_size, **options
        )

    if ids is None:
        [output] = results
    else:
        output = "".join(
            json.dumps({"id": entry_id, "text": result}, ensure_ascii=False) + "\n"
            for entry_id, result in zip(ids, results, strict=True)
        )

    if args.output_file is None:
        print(output, end="\n" if ids is None else "")
    else:
        output_file = Path(args.output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(output, encoding="utf-8")
