"""Tests for TransformersLLMPunctuator: decoding and where marks may go."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from transformers import AutoTokenizer, DynamicCache

from llm_punctuator.logits_processor import CustomLogitsProcessor
from llm_punctuator.punctuator import TransformersLLMPunctuator
from llm_punctuator.schema import EN_PUNCTUATIONS, ZH_PUNCTUATIONS

EOS_ID = 0
QWEN_MODEL = "Qwen/Qwen3-1.7B"
K_VALUES = pytest.mark.parametrize("k", [-1, 1], ids=["k_all", "k_1"])


class CharTokenizer:
    """One token per character, token id = code point."""

    eos_token_id = EOS_ID
    bos_token_id = None

    def encode(self, text: str) -> list[int]:
        """Map each character to its code point."""
        return [ord(c) for c in text]

    def decode(self, tokens: list[int], skip_special_tokens: bool = True) -> str:
        """Map code points back to characters, dropping EOS."""
        return "".join(chr(t) for t in tokens if t != EOS_ID)

    def apply_chat_template(self, messages: list[dict], **kwargs: object) -> str:
        """Join the messages into one tagged string."""
        return "".join(f"[{m['role']}]{m['content']}<end>" for m in messages)

    def __call__(self, text: str, return_tensors: str | None = None, **kwargs: object) -> dict:
        """Encode text as a batch of one, or as a list with each character's offsets."""
        if return_tensors is None:
            return {
                "input_ids": self.encode(text),
                "offset_mapping": [(i, i + 1) for i in range(len(text))],
            }
        ids = torch.tensor([self.encode(text)])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class ConstantScoreModel:
    """Fake LM that gives every position the same scores, and counts its forwards."""

    device = torch.device("cpu")

    def __init__(self, scores: torch.Tensor) -> None:
        """Keep the scores of one position, shaped (1, vocab_size)."""
        self.scores = scores
        self.forwards = 0

    def __call__(self, input_ids: torch.Tensor, **kwargs: object) -> SimpleNamespace:
        """Return the scores at every fed position."""
        self.forwards += 1
        return SimpleNamespace(logits=self.scores.expand(1, input_ids.shape[1], -1))


class PrefixScoreModel:
    """Fake LM whose scores at a position depend on every token before it.

    It keeps the token IDs in the cache, so a cropped cache changes what it sees.
    """

    device = torch.device("cpu")
    vocab_size = 0x10000

    def __call__(
        self, input_ids: torch.Tensor, past_key_values: DynamicCache, **kwargs: object
    ) -> SimpleNamespace:
        """Score text 11 and each mark p by (31 * sum of the prefix + p) % 13."""
        fed = input_ids[:, None, :, None]
        sequence = past_key_values.update(fed, fed, layer_idx=0)[0][0, 0, :, 0].tolist()
        logits = torch.full((1, input_ids.shape[1], self.vocab_size), 11.0)
        marks = torch.tensor([ord(p) for p in ZH_PUNCTUATIONS])
        for row in range(input_ids.shape[1]):
            prefix_sum = sum(sequence[: len(sequence) - input_ids.shape[1] + row + 1])
            logits[0, row, marks] = ((31 * prefix_sum + marks) % 13).float()
        return SimpleNamespace(logits=logits)


@pytest.fixture
def punctuator() -> TransformersLLMPunctuator:
    """Char tokenizer with a fake model that puts a mark wherever one is allowed."""
    punctuation_scores = torch.ones(1, 0x10000)
    punctuation_scores[0, EOS_ID] = 0.0
    punctuation_scores[0, [ord(p) for p in ZH_PUNCTUATIONS]] = 2.0
    with (
        patch("llm_punctuator.punctuator.AutoTokenizer") as tokenizer_cls,
        patch("llm_punctuator.punctuator.AutoModelForCausalLM") as model_cls,
    ):
        tokenizer_cls.from_pretrained.return_value = CharTokenizer()
        model_cls.from_pretrained.return_value = ConstantScoreModel(punctuation_scores)
        return TransformersLLMPunctuator("fake-model")


@pytest.fixture(scope="module")
def qwen_punctuator() -> TransformersLLMPunctuator:
    """Qwen3 tokenizer with a fake model that picks the first mark whenever one is allowed."""
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL)
    scores = torch.ones(1, len(tokenizer))
    scores[0, tokenizer.eos_token_id] = 0.0
    for punctuations in (ZH_PUNCTUATIONS, EN_PUNCTUATIONS):
        scores[0, tokenizer.encode(punctuations[0])[-1]] = 2.0
    with patch("llm_punctuator.punctuator.AutoModelForCausalLM") as model_cls:
        model_cls.from_pretrained.return_value = ConstantScoreModel(scores)
        return TransformersLLMPunctuator(QWEN_MODEL)


@K_VALUES
@pytest.mark.parametrize("chunk_size", [50, 5], ids=["single_chunk", "continued_chunks"])
def test_add_punctuation_keeps_all_text_when_every_position_is_punctuated(
    punctuator: TransformersLLMPunctuator,
    chunk_size: int,
    k: int,
) -> None:
    """No text is dropped even when every allowed position gets a mark."""
    text = "今天天氣很好我們去公園散步"

    result = punctuator.add_punctuation(text, language="zh", chunk_size=chunk_size, k=k)

    assert "".join(c for c in result if c not in ZH_PUNCTUATIONS) == text


@K_VALUES
@pytest.mark.parametrize(
    ("text", "language", "chunk_size", "expected"),
    [
        ("鄉東至水社大", "zh", 50, "鄉，東，至，水，社，大，"),
        ("鄉東至水社大", "zh", 4, "鄉，東，至，水，社，大，"),
        ("hello world how are you", "en", 50, "hello, world, how, are, you,"),
        ("hello world how are you", "en", 2, "hello, world, how, are, you,"),
    ],
    ids=["zh_single_chunk", "zh_continued_chunks", "en_single_chunk", "en_continued_chunks"],
)
def test_add_punctuation_keeps_trailing_mark_when_every_gap_is_punctuated(
    qwen_punctuator: TransformersLLMPunctuator,
    text: str,
    language: str,
    chunk_size: int,
    expected: str,
    k: int,
) -> None:
    """The trailing mark survives even when every gap in the last chunk gets a mark."""
    result = qwen_punctuator.add_punctuation(text, language=language, chunk_size=chunk_size, k=k)

    assert result == expected


@K_VALUES
@pytest.mark.parametrize(
    ("text", "chunk_size", "expected"),
    [
        ("鄉東至水社大山西至", 50, "鄉，東，至，水，社，大，山西，至，"),
        ("中共2017年", 50, "中共，2017，年，"),
        ("鼓勵長輩", 50, "鼓，勵，長，輩，"),
        ("他說hello world今天", 50, "他，說，hello， world，今天，"),
        ("買iPhone15花２０１７％", 50, "買，iPhone15，花，２０１７，％，"),
        ("口負成長新生兒", 50, "口，負，成，長新，生，兒，"),
        ("鼓勵長輩", 2, "鼓，勵，長，輩，"),
        ("中共2017年", 3, "中共，2017，年，"),
        ("口負成長新生兒", 4, "口，負，成，長新，生，兒，"),
    ],
    ids=[
        "merged_characters_stay_together",
        "digit_run",
        "rare_character",
        "english_words_with_space",
        "full_width_run_and_symbol",
        "token_straddles_two_characters",
        "chunk_cut_inside_rare_character",
        "chunk_cut_inside_digit_run",
        "chunk_cut_before_straddling_token",
    ],
)
def test_zh_marks_never_split_a_unit(
    qwen_punctuator: TransformersLLMPunctuator,
    text: str,
    chunk_size: int,
    expected: str,
    k: int,
) -> None:
    """A mark goes after every token that ends a unit, and never inside a unit."""
    result = qwen_punctuator.add_punctuation(text, language="zh", chunk_size=chunk_size, k=k)

    assert result == expected


@K_VALUES
@pytest.mark.parametrize("chunk_size", [50, 3], ids=["single_chunk", "continued_chunks"])
def test_en_marks_go_between_tokens_as_before(
    qwen_punctuator: TransformersLLMPunctuator,
    chunk_size: int,
    k: int,
) -> None:
    """English keeps one mark position per token, even inside a word or number."""
    result = qwen_punctuator.add_punctuation(
        "the punctuator works in 2017", language="en", chunk_size=chunk_size, k=k
    )

    assert result == "the, punct,uator, works, in, ,2,0,1,7,"


def step_by_step_greedy(prompt: list[int], rule: CustomLogitsProcessor) -> list[int]:
    """Decode one token per forward with the rule as a logits processor, without EOS."""
    model = PrefixScoreModel()
    ids = torch.tensor([prompt])
    while ids[0, -1].item() != EOS_ID:
        scores = model(ids, past_key_values=DynamicCache()).logits[:, -1]
        next_id = rule(ids, scores).argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)
    return ids[0, len(prompt) : -1].tolist()


@pytest.mark.parametrize("k", [-1, 1, 2], ids=["k_all", "k_1", "k_2"])
@pytest.mark.parametrize("has_prev_input", [False, True], ids=["first_chunk", "continued_chunk"])
def test_greedy_search_matches_step_by_step_greedy(
    punctuator: TransformersLLMPunctuator, k: int, has_prev_input: bool
) -> None:
    """Deciding several positions per forward gives the same tokens as one per forward."""
    punctuator.model = PrefixScoreModel()
    text = punctuator.encode_text("今天在2017年天氣很好我們去公園")
    prompt = punctuator.encode_text("[user]")
    unit_ends = {i for i in range(1, len(text) + 1) if i not in {4, 5, 6}}

    def new_rule() -> CustomLogitsProcessor:
        return CustomLogitsProcessor(
            text + [EOS_ID], [ord(p) for p in ZH_PUNCTUATIONS], has_prev_input, unit_ends
        )

    result = punctuator.greedy_search(prompt, new_rule(), k)

    assert result == step_by_step_greedy(prompt, new_rule())


@pytest.mark.parametrize(("k", "forwards"), [(-1, 1), (1, 5), (2, 3)], ids=["k_all", "k_1", "k_2"])
def test_k_caps_mark_positions_decided_per_forward(
    punctuator: TransformersLLMPunctuator, k: int, forwards: int
) -> None:
    """With no mark ever winning, 5 mark positions take ceil(5 / k) forwards, or 1 for k=-1."""
    text_scores = torch.ones(1, 0x10000)
    text_scores[0, [ord(p) for p in ZH_PUNCTUATIONS]] = 0.0
    punctuator.model = ConstantScoreModel(text_scores)

    punctuator.add_punctuation("今天天氣好", language="zh", chunk_size=50, k=k)

    assert punctuator.model.forwards == forwards


@pytest.mark.parametrize("k", [0, -2], ids=["zero", "below_minus_one"])
def test_add_punctuation_rejects_invalid_k(punctuator: TransformersLLMPunctuator, k: int) -> None:
    """Any k other than -1 or a positive integer raises ValueError."""
    with pytest.raises(ValueError, match=f"k must be -1 or a positive integer, got {k}"):
        punctuator.add_punctuation("今天天氣很好", k=k)
