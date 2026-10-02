"""Tests for CustomLogitsProcessor."""

import pytest
import torch

from llm_punctuator.logits_processor import CustomLogitsProcessor

TEXT = [10, 11]
PUNCTUATION = [1, 2]
EOS_ID = 99
PROMPT_ID = 5
VOCAB_SIZE = 100


@pytest.fixture
def processor() -> CustomLogitsProcessor:
    """Processor for a two-token text, starting a new chunk."""
    return CustomLogitsProcessor(TEXT + [EOS_ID], PUNCTUATION, has_prev_input=False)


@pytest.mark.parametrize(
    "last_token_id, expected",
    [(1, {EOS_ID}), (11, {1, 2, EOS_ID})],
    ids=["after_punctuation", "after_text"],
)
def test_allowed_tokens_after_all_text_allow_at_most_one_mark(
    processor: CustomLogitsProcessor, last_token_id: int, expected: set[int]
) -> None:
    """After the last text token, a mark is allowed only if the previous token is not one."""
    allowed = processor._get_allowed_tokens(len(TEXT), last_token_id)

    assert allowed == expected


def test_generation_preferring_punctuation_ends_with_one_mark_then_eos(
    processor: CustomLogitsProcessor,
) -> None:
    """A model that always prefers punctuation still stops after one trailing mark."""
    scores = torch.ones(1, VOCAB_SIZE)
    scores[0, EOS_ID] = 0.0
    scores[0, PUNCTUATION] = 2.0
    ids = torch.tensor([[PROMPT_ID]])
    max_steps = 4 * len(TEXT)

    for _ in range(max_steps):
        next_id = processor(ids, scores.clone()).argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)
        if next_id.item() == EOS_ID:
            break

    assert ids[0, 1:].tolist() == [10, 1, 11, 1, EOS_ID]
