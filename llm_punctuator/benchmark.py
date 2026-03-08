"""Benchmark evaluation for punctuation restoration quality."""

from __future__ import annotations

from pathlib import Path

from llm_punctuator.schema import ALLOWED_PUNCTUATIONS


def extract_punctuation_labels(text: str) -> tuple[str, list[str | None]]:
    """Extract plain text and per-character punctuation labels from punctuated text.

    Each label corresponds to the position after a plain text character.
    If a punctuation mark follows a character, that position gets the punctuation mark.
    Otherwise, it gets None.

    Args:
        text: Punctuated text string.

    Returns:
        A tuple of (plain_text, labels) where labels[i] is the punctuation
        after the i-th plain text character, or None.
    """
    plain_chars: list[str] = []
    labels: list[str | None] = []

    for char in text:
        if char in ALLOWED_PUNCTUATIONS:
            # Attach punctuation to the last plain character
            if plain_chars and labels[-1] is None:
                labels[-1] = char
            # If there's already a punctuation at this position, skip (keep first)
        elif char in "\n\r":
            continue
        else:
            plain_chars.append(char)
            labels.append(None)

    return "".join(plain_chars), labels


def compute_metrics(
    ref_labels: list[str | None],
    pred_labels: list[str | None],
) -> dict[str, dict[str, float]]:
    """Compute Precision, Recall, and F1-score for punctuation prediction.

    Args:
        ref_labels: Reference punctuation labels (one per character position).
        pred_labels: Predicted punctuation labels (one per character position).

    Returns:
        Dict mapping category name to {"precision", "recall", "f1"}.
        Keys include "overall" and each individual punctuation mark found.

    Raises:
        ValueError: If ref_labels and pred_labels have different lengths.
    """
    if len(ref_labels) != len(pred_labels):
        msg = f"Label lists must have the same length, got {len(ref_labels)} vs {len(pred_labels)}"
        raise ValueError(msg)

    # Collect all punctuation types present in either list
    all_puncts: set[str] = set()
    for label in ref_labels:
        if label is not None:
            all_puncts.add(label)
    for label in pred_labels:
        if label is not None:
            all_puncts.add(label)

    # Per-punctuation TP/FP/FN
    tp: dict[str, int] = dict.fromkeys(all_puncts, 0)
    fp: dict[str, int] = dict.fromkeys(all_puncts, 0)
    fn: dict[str, int] = dict.fromkeys(all_puncts, 0)

    for ref, pred in zip(ref_labels, pred_labels, strict=True):
        if ref is not None and pred == ref:
            tp[ref] += 1
        elif pred is not None and (ref is None or ref != pred):
            fp[pred] += 1
        if ref is not None and pred != ref:
            fn[ref] += 1

    def _prf(tp_val: int, fp_val: int, fn_val: int) -> dict[str, float]:
        precision = tp_val / (tp_val + fp_val) if (tp_val + fp_val) > 0 else 0.0
        recall = tp_val / (tp_val + fn_val) if (tp_val + fn_val) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {"precision": precision, "recall": recall, "f1": f1}

    result: dict[str, dict[str, float]] = {}

    # Per-punctuation metrics
    for p in sorted(all_puncts):
        result[p] = _prf(tp[p], fp[p], fn[p])

    # Overall (micro-average)
    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())
    result["overall"] = _prf(total_tp, total_fp, total_fn)

    return result


def load_file_pairs(
    benchmark_dir: Path,
    output_dir: Path,
) -> dict[str, list[tuple[Path, Path]]]:
    """Load matching file pairs from benchmark category subdirectories.

    Scans subdirectories of benchmark_dir as categories (e.g. asr/, news/, wiki/).
    For each category, matches reference files with output files by relative path.

    Args:
        benchmark_dir: Directory containing category subdirectories with reference files.
        output_dir: Directory containing model output files in matching structure.

    Returns:
        Dict mapping category name to sorted list of (reference_path, output_path) tuples.

    Raises:
        FileNotFoundError: If an output file is missing for a reference file.
    """
    result: dict[str, list[tuple[Path, Path]]] = {}

    categories = sorted(d for d in benchmark_dir.iterdir() if d.is_dir())
    for category_dir in categories:
        category = category_dir.name
        ref_files = sorted(category_dir.glob("*.txt"))
        if not ref_files:
            continue

        pairs: list[tuple[Path, Path]] = []
        for ref_file in ref_files:
            out_file = output_dir / category / ref_file.name
            if not out_file.exists():
                msg = f"Output file not found: {category}/{ref_file.name}"
                raise FileNotFoundError(msg)
            pairs.append((ref_file, out_file))

        result[category] = pairs

    return result


def produce_markdown_table(metrics: dict[str, dict[str, float]]) -> str:
    """Produce a markdown table from metrics dict.

    Args:
        metrics: Dict mapping category to {"precision", "recall", "f1"}.

    Returns:
        Formatted markdown table string.
    """
    # Column order: Overall first, then sorted punctuation marks
    columns = ["overall"]
    punct_cols = sorted(k for k in metrics if k != "overall")
    columns.extend(punct_cols)

    # Use fixed column width for consistent alignment
    # "Overall" is 7 chars; punctuation marks may be fullwidth (2 display cols)
    col_width = 7  # display width for all columns

    def _display_width(s: str) -> int:
        """Approximate display width accounting for fullwidth characters."""
        w = 0
        for ch in s:
            if "\u2e80" <= ch <= "\U0001f9ff":
                w += 2
            else:
                w += 1
        return w

    def _pad_center(s: str, width: int) -> str:
        """Center-pad string accounting for display width."""
        dw = _display_width(s)
        pad = width - dw
        if pad <= 0:
            return s
        left = pad // 2
        right = pad - left
        return " " * left + s + " " * right

    # Header
    header_names = {"overall": "Overall"}
    header_names.update({p: p for p in punct_cols})
    header_cells = [_pad_center(header_names[c], col_width) for c in columns]
    header = "| Metric    | " + " | ".join(header_cells) + " |"

    # Separator
    sep = "|-----------|-" + "-|-".join("-" * col_width for _ in columns) + "-|"

    # Data rows
    rows = []
    for metric_name, key in [("Precision", "precision"), ("Recall", "recall"), ("F1-score", "f1")]:
        values = [f"{metrics[c][key]:.2f}" for c in columns]
        padded = [_pad_center(v, col_width) for v in values]
        row = f"| {metric_name:<9} | " + " | ".join(padded) + " |"
        rows.append(row)

    return "\n".join([header, sep, *rows]) + "\n"
