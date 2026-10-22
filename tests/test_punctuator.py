"""Tests for TransformersLLMPunctuator: decoding and where marks may go."""

import itertools
import re
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from transformers import AutoTokenizer, DynamicCache, Qwen3Config, Qwen3ForCausalLM

from llm_punctuator.logits_processor import CustomLogitsProcessor, DistanceBonusProcessor
from llm_punctuator.punctuator import ChunkJob, TransformersLLMPunctuator
from llm_punctuator.schema import EN_PUNCTUATIONS, ZH_PUNCTUATIONS

EOS_ID = 3
QWEN_MODEL = "Qwen/Qwen3-1.7B"
K_VALUES = pytest.mark.parametrize("k", [-1, 1], ids=["k_all", "k_1"])

ZH_UNIT_CASES = [
    pytest.param(
        "鄉東至水社大山西至",
        50,
        "鄉，東，至，水，社，大，山西，至，",
        id="merged_characters_stay_together",
    ),
    pytest.param("中共2017年", 50, "中共，2017，年，", id="digit_run"),
    pytest.param("鼓勵長輩", 50, "鼓，勵，長，輩，", id="rare_character"),
    pytest.param(
        "他說hello world今天", 50, "他，說，hello， world，今天，", id="english_words_with_space"
    ),
    pytest.param(
        "買iPhone15花２０１７％",
        50,
        "買，iPhone15，花，２０１７，％，",
        id="full_width_run_and_symbol",
    ),
    pytest.param(
        "口負成長新生兒", 50, "口，負，成，長新，生，兒，", id="token_straddles_two_characters"
    ),
    pytest.param("鼓勵長輩", 2, "鼓，勵，長，輩，", id="chunk_cut_inside_rare_character"),
    pytest.param("中共2017年", 3, "中共，2017，年，", id="chunk_cut_inside_digit_run"),
    pytest.param(
        "口負成長新生兒", 4, "口，負，成，長新，生，兒，", id="chunk_cut_before_straddling_token"
    ),
    pytest.param("買AK-47和F-16", 50, "買，AK-47，和，F-16，", id="dash_joins_letters_and_digits"),
    pytest.param("COVID-19疫情", 50, "COVID-19，疫情，", id="dash_joins_word_and_number"),
    pytest.param("圓周率3.14", 50, "圓，周，率，3.14，", id="dot_joins_digits"),
    pytest.param("2024-10-10", 50, "2024-10-10，", id="dashes_join_date"),
    pytest.param("self-driving車", 50, "self-driving，車，", id="dash_joins_words"),
    pytest.param("v1.2.3版", 50, "v1.2.3，版，", id="dots_join_version"),
    pytest.param("U.S.政府", 50, "U.S，.，政府，", id="dot_before_chinese_stands_alone"),
    pytest.param("3-月", 50, "3，-，月，", id="dash_before_chinese_stands_alone"),
    pytest.param("ＡＫ－４７", 50, "ＡＫ，－，４７，", id="full_width_dash_stands_alone"),
    pytest.param(
        "買AK-47和F-16", 2, "買，AK-47，和，F-16，", id="chunk_cut_inside_dash_joined_run"
    ),
]


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
        return SimpleNamespace(logits=self.scores.expand(*input_ids.shape, -1))


class PrefixScoreModel:
    """Fake LM whose scores at a position depend on that position and every token it attends to.

    It keeps the token IDs in the cache, so a cropped cache or a masked token changes what it
    sees.
    """

    device = torch.device("cpu")
    vocab_size = 0x10000

    def __call__(
        self,
        input_ids: torch.Tensor,
        past_key_values: DynamicCache,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        **kwargs: object,
    ) -> SimpleNamespace:
        """Score text 11 and each mark p by (31 * sum of attended tokens + 7 * position + p) % 13.

        Like a transformers model given a 2D mask, a token attends to the cached tokens up to
        it that the mask keeps. Without a mask it keeps all. Without position IDs, positions
        count from the start of the cache.
        """
        fed = input_ids[:, None, :, None]
        sequence = past_key_values.update(fed, fed, layer_idx=0)[0][:, 0, :, 0]
        total, width = sequence.shape[1], input_ids.shape[1]
        if attention_mask is None:
            attention_mask = torch.ones_like(sequence)
        if position_ids is None:
            position_ids = torch.arange(total - width, total)[None]
        causal = torch.ones(total, total, dtype=torch.long).tril()[-width:]
        attended_sums = (attention_mask[:, None, :] * causal * sequence[:, None, :]).sum(dim=-1)
        marks = torch.tensor([ord(p) for p in ZH_PUNCTUATIONS])
        logits = torch.full((*input_ids.shape, self.vocab_size), 11.0)
        logits[:, :, marks] = ((31 * attended_sums + 7 * position_ids)[..., None] + marks) % 13.0
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

    [result] = punctuator.add_punctuation([text], language="zh", chunk_size=chunk_size, k=k)

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
    [result] = qwen_punctuator.add_punctuation(
        [text], language=language, chunk_size=chunk_size, k=k
    )

    assert result == expected


@K_VALUES
@pytest.mark.parametrize(("text", "chunk_size", "expected"), ZH_UNIT_CASES)
def test_zh_marks_never_split_a_unit(
    qwen_punctuator: TransformersLLMPunctuator,
    text: str,
    chunk_size: int,
    expected: str,
    k: int,
) -> None:
    """A mark goes after every token that ends a unit, and never inside a unit."""
    [result] = qwen_punctuator.add_punctuation([text], language="zh", chunk_size=chunk_size, k=k)

    assert result == expected


@K_VALUES
@pytest.mark.parametrize("chunk_size", [50, 3], ids=["single_chunk", "continued_chunks"])
def test_en_marks_go_between_tokens_as_before(
    qwen_punctuator: TransformersLLMPunctuator,
    chunk_size: int,
    k: int,
) -> None:
    """English keeps one mark position per token, even inside a word or number."""
    [result] = qwen_punctuator.add_punctuation(
        ["the punctuator works in 2017"], language="en", chunk_size=chunk_size, k=k
    )

    assert result == "the, punct,uator, works, in, ,2,0,1,7,"


def step_by_step_greedy(
    prompt: list[int],
    rule: CustomLogitsProcessor,
    model: object | None = None,
    distance_bonus: float = 0.0,
) -> list[int]:
    """Decode one token per forward with the rule as a logits processor, without EOS.

    Before the rule, every mark gets distance_bonus times the non-whitespace characters
    generated since the last mark, with one character per token as in CharTokenizer. The model
    defaults to PrefixScoreModel. Each forward starts from an empty cache.
    """
    model = model or PrefixScoreModel()
    marks = sorted(rule.punctuation_tokens)
    ids = torch.tensor([prompt])
    while ids[0, -1].item() != rule.original_text_tokens[-1]:
        scores = model(ids, past_key_values=DynamicCache()).logits[:, -1]
        generated = ids[0, len(prompt) :].tolist()
        since_mark = itertools.takewhile(lambda t: t not in marks, reversed(generated))
        scores[:, marks] += distance_bonus * sum(not chr(t).isspace() for t in since_mark)
        next_id = rule(ids, scores).argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)
    return ids[0, len(prompt) : -1].tolist()


@pytest.mark.parametrize("k", [-1, 1, 2], ids=["k_all", "k_1", "k_2"])
def test_greedy_search_matches_step_by_step_greedy(
    punctuator: TransformersLLMPunctuator, k: int
) -> None:
    """Rows decoded together each get the tokens that one token per forward gives them alone."""
    punctuator.model = PrefixScoreModel()
    text = punctuator.encode_text("今天在2017年天氣很好我們去公園")
    short_text = punctuator.encode_text("去公園散步")
    text_unit_ends = {i for i in range(1, len(text) + 1) if i not in {4, 5, 6}}
    rows = [
        ("[user]", text, False, text_unit_ends),
        ("[system]說明[user]", text, True, text_unit_ends),
        ("[user]", short_text, True, set(range(1, len(short_text) + 1))),
    ]

    def new_job(
        prompt: str, tokens: list[int], has_prev_input: bool, unit_ends: set[int]
    ) -> ChunkJob:
        rule = CustomLogitsProcessor(
            tokens + [EOS_ID], [ord(p) for p in ZH_PUNCTUATIONS], has_prev_input, unit_ends
        )
        return punctuator.encode_text(prompt), rule

    results = punctuator.greedy_search([new_job(*row) for row in rows], k)

    assert results == [step_by_step_greedy(*new_job(*row)) for row in rows]


@pytest.mark.parametrize("distance_bonus", [0.0, 0.25], ids=["no_bonus", "bonus"])
@pytest.mark.parametrize("k", [-1, 1, 16], ids=["k_all", "k_1", "k_16"])
def test_greedy_search_with_distance_bonus_matches_step_by_step_greedy(
    punctuator: TransformersLLMPunctuator, k: int, distance_bonus: float
) -> None:
    """Rows of a distance bonus rule get what one token per forward gives with the same bonus."""
    punctuator.model = PrefixScoreModel()
    marks = [ord(p) for p in ZH_PUNCTUATIONS]
    texts = [
        ("[user]", "今天在2017年 天氣很好我們去公園散步然後回家吃飯再去睡覺", False),
        ("[system]說明[user]", "去公園散步然後回家吃飯", True),
    ]
    jobs, references = [], []
    for prompt, text, has_prev_input in texts:
        tokens = punctuator.encode_text(text)
        unit_ends = set(range(1, len(tokens) + 1))
        char_counts = [sum(not c.isspace() for c in text[:p]) for p in range(len(text) + 1)]
        prompt_ids = punctuator.encode_text(prompt)
        rule = DistanceBonusProcessor(
            tokens + [EOS_ID], marks, has_prev_input, unit_ends, char_counts, distance_bonus
        )
        jobs.append((prompt_ids, rule))
        plain = CustomLogitsProcessor(tokens + [EOS_ID], marks, has_prev_input, unit_ends)
        references.append(step_by_step_greedy(prompt_ids, plain, distance_bonus=distance_bonus))

    results = punctuator.greedy_search(jobs, k)

    assert results == references


@pytest.mark.parametrize("k", [-1, 1], ids=["k_all", "k_1"])
@pytest.mark.parametrize("attn_implementation", ["eager", "sdpa"])
def test_greedy_search_masks_rows_under_each_attention_backend(
    qwen_punctuator: TransformersLLMPunctuator, attn_implementation: str, k: int
) -> None:
    """With real transformers attention, rows decoded together match one token per forward."""
    tokenizer = qwen_punctuator.tokenizer
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
    )
    config._attn_implementation = attn_implementation
    model = Qwen3ForCausalLM(config).eval()
    punctuator = TransformersLLMPunctuator.__new__(TransformersLLMPunctuator)
    punctuator.tokenizer, punctuator.model, punctuator.device = tokenizer, model, model.device
    marks = [tokenizer.encode(p)[-1] for p in ZH_PUNCTUATIONS]
    rows = [
        ("系統提示", "今天天氣很好我們去公園散步"),
        ("提示", "鄉東至水社大山西至"),
        ("", "鼓勵長輩"),
    ]
    jobs = []
    for prompt, text in rows:
        tokens = tokenizer.encode(text)
        unit_ends = set(range(1, len(tokens) + 1))
        rule = CustomLogitsProcessor(tokens + [tokenizer.eos_token_id], marks, True, unit_ends)
        jobs.append((tokenizer.encode(f"<|im_start|>{prompt}"), rule))

    with torch.no_grad():
        results = punctuator.greedy_search(jobs, k)
        expected = [step_by_step_greedy(prompt, rule, model) for prompt, rule in jobs]

    assert results == expected


@pytest.mark.parametrize(("k", "forwards"), [(-1, 1), (1, 5), (2, 3)], ids=["k_all", "k_1", "k_2"])
def test_k_caps_mark_positions_decided_per_forward(
    punctuator: TransformersLLMPunctuator, k: int, forwards: int
) -> None:
    """With no mark ever winning, 5 mark positions take ceil(5 / k) forwards, or 1 for k=-1."""
    text_scores = torch.ones(1, 0x10000)
    text_scores[0, [ord(p) for p in ZH_PUNCTUATIONS]] = 0.0
    punctuator.model = ConstantScoreModel(text_scores)

    punctuator.add_punctuation(["今天天氣好"], language="zh", chunk_size=50, k=k)

    assert punctuator.model.forwards == forwards


@pytest.mark.parametrize("k", [0, -2], ids=["zero", "below_minus_one"])
def test_add_punctuation_rejects_invalid_k(punctuator: TransformersLLMPunctuator, k: int) -> None:
    """Any k other than -1 or a positive integer raises ValueError."""
    with pytest.raises(ValueError, match=f"k must be -1 or a positive integer, got {k}"):
        punctuator.add_punctuation(["今天天氣很好"], k=k)


ZH_INPUTS = [
    "鄉東至水社大山西至",
    "中共2017年",
    "鼓勵長輩",
    "他說hello world今天",
    "買iPhone15花２０１７％",
    "鄉東至水社大",
]
EN_INPUTS = ["hello world how are you", "the punctuator works in 2017"]


@K_VALUES
@pytest.mark.parametrize(
    ("texts", "language", "chunk_size"),
    [
        *[(ZH_INPUTS, "zh", chunk_size) for chunk_size in (50, 2, 3, 4)],
        *[(EN_INPUTS, "en", chunk_size) for chunk_size in (50, 2, 3)],
    ],
    ids=["zh_50", "zh_2", "zh_3", "zh_4", "en_50", "en_2", "en_3"],
)
def test_batch_gives_each_text_its_own_output(
    qwen_punctuator: TransformersLLMPunctuator,
    texts: list[str],
    language: str,
    chunk_size: int,
    k: int,
) -> None:
    """Texts punctuated together get what each gets alone, in input order."""
    options = {"language": language, "chunk_size": chunk_size, "k": k}
    alone = [qwen_punctuator.add_punctuation([text], batch_size=1, **options)[0] for text in texts]

    together = qwen_punctuator.add_punctuation(texts, batch_size=8, **options)

    assert together == alone


def test_add_punctuation_rejects_a_single_str(punctuator: TransformersLLMPunctuator) -> None:
    """A bare str raises TypeError that says to wrap it in a list."""
    with pytest.raises(
        TypeError,
        match=re.escape("texts must be a list of str, got str. Wrap a single text as [text]."),
    ):
        punctuator.add_punctuation("今天天氣很好")  # type: ignore[arg-type]


@pytest.mark.parametrize("batch_size", [0, -1], ids=["zero", "negative"])
def test_add_punctuation_rejects_batch_size_below_one(
    punctuator: TransformersLLMPunctuator, batch_size: int
) -> None:
    """batch_size below 1 raises ValueError."""
    with pytest.raises(
        ValueError, match=f"batch_size must be a positive integer, got {batch_size}"
    ):
        punctuator.add_punctuation(["今天天氣很好"], batch_size=batch_size)
