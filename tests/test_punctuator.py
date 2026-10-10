"""Tests for TransformersLLMPunctuator: generation budget and where marks may go."""

from unittest.mock import patch

import pytest
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from llm_punctuator.punctuator import TransformersLLMPunctuator
from llm_punctuator.schema import EN_PUNCTUATIONS, ZH_PUNCTUATIONS

EOS_ID = 0
QWEN_MODEL = "Qwen/Qwen3-1.7B"


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


class PunctuationGreedyModel:
    """Fake LM that scores punctuation highest, so it punctuates every allowed position."""

    device = torch.device("cpu")
    vocab_size = 0x10000

    def generate(
        self,
        input_ids: torch.Tensor,
        logits_processor: list,
        max_length: int,
        **kwargs: object,
    ) -> torch.Tensor:
        """Greedy decoding with the processors applied, up to max_length or EOS."""
        scores = torch.ones(1, self.vocab_size)
        scores[0, EOS_ID] = 0.0
        scores[0, [ord(p) for p in ZH_PUNCTUATIONS]] = 2.0
        ids = input_ids
        while ids.shape[1] < max_length:
            step = scores.clone()
            for processor in logits_processor:
                step = processor(ids, step)
            next_id = step.argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
            if next_id.item() == EOS_ID:
                break
        return ids


@pytest.fixture
def punctuator() -> TransformersLLMPunctuator:
    """Punctuator built on the fake tokenizer and model."""
    with (
        patch("llm_punctuator.punctuator.AutoTokenizer") as tokenizer_cls,
        patch("llm_punctuator.punctuator.AutoModelForCausalLM") as model_cls,
    ):
        tokenizer_cls.from_pretrained.return_value = CharTokenizer()
        model_cls.from_pretrained.return_value = PunctuationGreedyModel()
        return TransformersLLMPunctuator("fake-model")


class FirstMarkModel:
    """Fake LM that picks the first mark of the language whenever a mark is allowed."""

    device = torch.device("cpu")

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        """Score the first zh and en marks above text, and text above EOS."""
        self.eos_token_id = tokenizer.eos_token_id
        self.scores = torch.ones(1, len(tokenizer))
        self.scores[0, self.eos_token_id] = 0.0
        for punctuations in (ZH_PUNCTUATIONS, EN_PUNCTUATIONS):
            self.scores[0, tokenizer.encode(punctuations[0])[-1]] = 2.0

    def generate(
        self,
        input_ids: torch.Tensor,
        logits_processor: list,
        max_length: int,
        **kwargs: object,
    ) -> torch.Tensor:
        """Greedy decoding with the processors applied, up to max_length or EOS."""
        ids = input_ids
        while ids.shape[1] < max_length:
            step = self.scores.clone()
            for processor in logits_processor:
                step = processor(ids, step)
            next_id = step.argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
            if next_id.item() == self.eos_token_id:
                break
        return ids


@pytest.fixture(scope="module")
def qwen_punctuator() -> TransformersLLMPunctuator:
    """Punctuator built on the Qwen3 tokenizer and the first-mark fake model."""
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL)
    with patch("llm_punctuator.punctuator.AutoModelForCausalLM") as model_cls:
        model_cls.from_pretrained.return_value = FirstMarkModel(tokenizer)
        return TransformersLLMPunctuator(QWEN_MODEL)


@pytest.mark.parametrize("chunk_size", [50, 5], ids=["single_chunk", "continued_chunks"])
def test_add_punctuation_keeps_all_text_when_every_position_is_punctuated(
    punctuator: TransformersLLMPunctuator, chunk_size: int
) -> None:
    """No text is dropped even when every allowed position gets a mark."""
    text = "今天天氣很好我們去公園散步"

    result = punctuator.add_punctuation(text, language="zh", chunk_size=chunk_size)

    assert "".join(c for c in result if c not in ZH_PUNCTUATIONS) == text


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
) -> None:
    """The trailing mark survives even when every gap in the last chunk gets a mark."""
    result = qwen_punctuator.add_punctuation(text, language=language, chunk_size=chunk_size)

    assert result == expected


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
    qwen_punctuator: TransformersLLMPunctuator, text: str, chunk_size: int, expected: str
) -> None:
    """A mark goes after every token that ends a unit, and never inside a unit."""
    result = qwen_punctuator.add_punctuation(text, language="zh", chunk_size=chunk_size)

    assert result == expected


@pytest.mark.parametrize("chunk_size", [50, 3], ids=["single_chunk", "continued_chunks"])
def test_en_marks_go_between_tokens_as_before(
    qwen_punctuator: TransformersLLMPunctuator, chunk_size: int
) -> None:
    """English keeps one mark position per token, even inside a word or number."""
    result = qwen_punctuator.add_punctuation(
        "the punctuator works in 2017", language="en", chunk_size=chunk_size
    )

    assert result == "the, punct,uator, works, in, ,2,0,1,7,"
