"""Tests for TransformersLLMPunctuator generation budget."""

from unittest.mock import patch

import pytest
import torch

from llm_punctuator.punctuator import TransformersLLMPunctuator
from llm_punctuator.schema import ZH_PUNCTUATIONS

EOS_ID = 0


class CharTokenizer:
    """One token per character, token id = code point."""

    eos_token_id = EOS_ID
    bos_token_id = None

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, tokens: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(chr(t) for t in tokens if t != EOS_ID)

    def apply_chat_template(self, messages: list[dict], **kwargs: object) -> str:
        return "".join(f"[{m['role']}]{m['content']}<end>" for m in messages)

    def __call__(self, text: str, return_tensors: str) -> dict[str, torch.Tensor]:
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
    with (
        patch("llm_punctuator.punctuator.AutoTokenizer") as tokenizer_cls,
        patch("llm_punctuator.punctuator.AutoModelForCausalLM") as model_cls,
    ):
        tokenizer_cls.from_pretrained.return_value = CharTokenizer()
        model_cls.from_pretrained.return_value = PunctuationGreedyModel()
        return TransformersLLMPunctuator("fake-model")


@pytest.mark.parametrize("chunk_size", [50, 5], ids=["single_chunk", "continued_chunks"])
def test_add_punctuation_keeps_all_text_when_every_position_is_punctuated(
    punctuator: TransformersLLMPunctuator, chunk_size: int
) -> None:
    text = "今天天氣很好我們去公園散步"

    result = punctuator.add_punctuation(text, language="zh", chunk_size=chunk_size)

    assert "".join(c for c in result if c not in ZH_PUNCTUATIONS) == text
