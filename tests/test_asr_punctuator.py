"""Tests for ASRPunctuator: audio input, prompt, and forced decoding."""

import copy
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import soundfile
import torch
from transformers import AutoTokenizer, DynamicCache, Qwen3Config, Qwen3ForCausalLM
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoderConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeAudioEncoder

from llm_punctuator.asr_punctuator import (
    SAMPLE_RATE,
    ASRPunctuator,
    build_prompt,
    load_audio,
    mask_attention_windows,
)
from llm_punctuator.schema import EN_PUNCTUATIONS, ZH_PUNCTUATIONS
from tests.test_punctuator import ZH_UNIT_CASES

ASR_MODEL = "Qwen/Qwen3-ASR-1.7B"
AUDIO_TOKEN_ID = 151676
HIDDEN = 8
N_AUDIO = 3
K_VALUES = pytest.mark.parametrize("k", [-1, 1], ids=["k_all", "k_1"])


def tone(seconds: float, sample_rate: int, peak: float = 0.5) -> np.ndarray:
    """A 440 Hz sine as float32."""
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    return (peak * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def test_path_and_array_load_the_same(tmp_path: Path) -> None:
    """A file and the same samples as (array, 16000) give the same waveform."""
    signal = tone(1.0, SAMPLE_RATE)
    path = tmp_path / "tone.wav"
    soundfile.write(path, signal, SAMPLE_RATE, subtype="FLOAT")

    np.testing.assert_array_equal(load_audio(str(path)), load_audio((signal, SAMPLE_RATE)))


def test_48k_audio_is_resampled_to_16k() -> None:
    """One second at 48 kHz becomes 16000 samples."""
    assert load_audio((tone(1.0, 48000), 48000)).shape == (SAMPLE_RATE,)


def test_stereo_file_is_averaged_to_mono(tmp_path: Path) -> None:
    """Each sample is the mean of the two channels."""
    left, right = tone(1.0, SAMPLE_RATE, 0.2), tone(1.0, SAMPLE_RATE, 0.6)
    path = tmp_path / "stereo.wav"
    soundfile.write(path, np.stack([left, right], axis=1), SAMPLE_RATE, subtype="FLOAT")

    np.testing.assert_allclose(load_audio(str(path)), (left + right) / 2, atol=1e-7)


def test_peak_above_one_is_scaled_to_one() -> None:
    """An int-scaled array is divided by its peak."""
    signal = tone(1.0, SAMPLE_RATE, peak=0.5)

    np.testing.assert_allclose(load_audio((signal * 4, SAMPLE_RATE)), signal * 2, atol=1e-7)


class FakeAudioEncoder:
    """Turns any features into N_AUDIO zero vectors, and counts its calls."""

    def __init__(self) -> None:
        """Start with no calls."""
        self.calls = 0

    def __call__(self, input_features: torch.Tensor, feature_lens: torch.Tensor) -> SimpleNamespace:
        """Return N_AUDIO vectors of size HIDDEN."""
        self.calls += 1
        return SimpleNamespace(last_hidden_state=torch.zeros(N_AUDIO, HIDDEN))


class ConstantScoreLM:
    """Fake LM fed by embeddings that gives every position the same scores."""

    device = torch.device("cpu")
    dtype = torch.float32

    def __init__(self, scores: torch.Tensor) -> None:
        """Keep the scores of one position, shaped (1, vocab_size)."""
        self.scores = scores
        self.embedding = torch.nn.Embedding(scores.shape[1], HIDDEN)

    def get_input_embeddings(self) -> torch.nn.Embedding:
        """Return the token embedding."""
        return self.embedding

    def __call__(self, inputs_embeds: torch.Tensor, **kwargs: object) -> SimpleNamespace:
        """Return the scores at every fed position."""
        return SimpleNamespace(logits=self.scores.expand(*inputs_embeds.shape[:2], -1))


@pytest.fixture(scope="module")
def asr_punctuator() -> ASRPunctuator:
    """Real Qwen3-ASR tokenizer with a fake model that picks the first mark whenever one is allowed."""
    tokenizer = AutoTokenizer.from_pretrained(ASR_MODEL)
    scores = torch.ones(1, len(tokenizer))
    scores[0, tokenizer.eos_token_id] = 0.0
    for punctuations in (ZH_PUNCTUATIONS, EN_PUNCTUATIONS):
        scores[0, tokenizer.encode(punctuations[0])[-1]] = 2.0
    with patch("llm_punctuator.asr_punctuator.load_model") as load_model:
        load_model.return_value = (FakeAudioEncoder(), ConstantScoreLM(scores), AUDIO_TOKEN_ID)
        return ASRPunctuator(ASR_MODEL)


@pytest.fixture
def wav_path(tmp_path: Path) -> Path:
    """One second of tone at 16 kHz."""
    path = tmp_path / "tone.wav"
    soundfile.write(path, tone(1.0, SAMPLE_RATE), SAMPLE_RATE, subtype="FLOAT")
    return path


@K_VALUES
@pytest.mark.parametrize(
    ("text", "expected"),
    [pytest.param(*case.values[::2], id=case.id) for case in ZH_UNIT_CASES if case.values[1] == 50],
)
def test_zh_marks_never_split_a_unit(
    asr_punctuator: ASRPunctuator, wav_path: Path, text: str, expected: str, k: int
) -> None:
    """The #24 zh cases without a chunk cut hold for the ASR punctuator."""
    [result] = asr_punctuator.add_punctuation([text], [str(wav_path)], language="zh", k=k)

    assert result == expected


def test_each_text_gets_its_own_result_in_input_order(
    asr_punctuator: ASRPunctuator, wav_path: Path
) -> None:
    """Two texts come back in the order given."""
    results = asr_punctuator.add_punctuation(["鼓勵長輩", "中共2017年"], [str(wav_path)] * 2)

    assert results == ["鼓，勵，長，輩，", "中共，2017，年，"]


def test_build_prompt_fills_context_and_keeps_other_braces() -> None:
    """Only {text} is replaced, the pads match n_audio, and the tag becomes the model's name."""
    prompt = build_prompt("主題：{text}{x}", "今天", 2, "yue")

    assert prompt == (
        "<|im_start|>system\n主題：今天{x}<|im_end|>\n"
        "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_pad|><|audio_end|><|im_end|>\n"
        "<|im_start|>assistant\nlanguage Cantonese<asr_text>"
    )


@pytest.fixture
def tiny_asr_punctuator(asr_punctuator: ASRPunctuator) -> ASRPunctuator:
    """A randomly initialized real Qwen3ForCausalLM behind the ASR forward."""
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=len(asr_punctuator.tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
    )
    punctuator = ASRPunctuator.__new__(ASRPunctuator)
    punctuator.model = Qwen3ForCausalLM(config).eval()
    punctuator.device, punctuator.audio_token_id = punctuator.model.device, AUDIO_TOKEN_ID
    return punctuator


def forward_all(punctuator: ASRPunctuator, ids: list[int]) -> torch.Tensor:
    """Run the ASR forward on ids as one row with a fresh cache."""
    input_ids = torch.tensor([ids])
    positions = torch.arange(len(ids))
    return punctuator.forward(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        position_ids=positions[None],
        cache_position=positions,
        past_key_values=DynamicCache(),
    )


def test_forward_feeds_audio_vectors_at_audio_pads(tiny_asr_punctuator: ASRPunctuator) -> None:
    """The logits equal the real model fed the audio vectors in place of the pads."""
    audio = torch.randn(2, 32)
    tiny_asr_punctuator.pending_audio = audio
    embed = tiny_asr_punctuator.model.get_input_embeddings()
    before, after = torch.tensor([151644]), torch.tensor([151670, 151645])

    with torch.no_grad():
        logits = forward_all(
            tiny_asr_punctuator, [151644, AUDIO_TOKEN_ID, AUDIO_TOKEN_ID, 151670, 151645]
        )
        expected = tiny_asr_punctuator.model(
            inputs_embeds=torch.cat([embed(before), audio, embed(after)])[None]
        ).logits

    torch.testing.assert_close(logits, expected)


def test_forward_rejects_pad_count_that_differs_from_audio_vectors(
    tiny_asr_punctuator: ASRPunctuator,
) -> None:
    """Two pads with three vectors raises ValueError."""
    tiny_asr_punctuator.pending_audio = torch.randn(3, 32)

    with pytest.raises(ValueError, match="2 audio pads but 3 audio vectors"):
        forward_all(tiny_asr_punctuator, [151644, AUDIO_TOKEN_ID, AUDIO_TOKEN_ID, 151645])


MEL_BINS = 16
WINDOW_FRAMES = 800
WINDOW_VECTORS = 104


def tiny_audio_encoder(attn_implementation: str) -> Qwen3OmniMoeAudioEncoder:
    """A randomly initialized real audio encoder with Qwen3-ASR's chunk and window sizes."""
    torch.manual_seed(0)
    config = Qwen3OmniMoeAudioEncoderConfig(
        num_mel_bins=MEL_BINS,
        encoder_layers=2,
        encoder_attention_heads=2,
        encoder_ffn_dim=32,
        d_model=16,
        output_dim=16,
        downsample_hidden_size=4,
        n_window=50,
        n_window_infer=WINDOW_FRAMES,
    )
    config._attn_implementation = attn_implementation
    return Qwen3OmniMoeAudioEncoder(config).eval()


def encode(encoder: Qwen3OmniMoeAudioEncoder, features: torch.Tensor) -> torch.Tensor:
    """Encode one audio's mel features, shaped (MEL_BINS, frames)."""
    with torch.no_grad():
        return encoder(features, feature_lens=torch.tensor([features.shape[1]])).last_hidden_state


@pytest.mark.parametrize("attn_implementation", ["sdpa", "eager"])
def test_audio_after_the_first_window_does_not_change_its_vectors(
    attn_implementation: str,
) -> None:
    """Changing the 50-frame tail window leaves the first window's vectors as they were."""
    encoder = tiny_audio_encoder(attn_implementation)
    mask_attention_windows(encoder)
    features = torch.randn(MEL_BINS, WINDOW_FRAMES + 50)
    changed = features.clone()
    changed[:, WINDOW_FRAMES:] = torch.randn(MEL_BINS, 50)

    before, after = encode(encoder, features), encode(encoder, changed)

    assert torch.equal(after[:WINDOW_VECTORS], before[:WINDOW_VECTORS])


def test_audio_of_one_window_encodes_bit_identical_to_the_unmasked_encoder() -> None:
    """With a single window the mask hides nothing, so the output does not change at all."""
    encoder = tiny_audio_encoder("sdpa")
    unmasked = copy.deepcopy(encoder)
    mask_attention_windows(encoder)
    features = torch.randn(MEL_BINS, WINDOW_FRAMES)

    assert torch.equal(encode(encoder, features), encode(unmasked, features))


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        (
            {"texts": "今天"},
            TypeError,
            re.escape("texts must be a list of str, got str. Wrap a single text as [text]."),
        ),
        ({"k": 0}, ValueError, "k must be -1 or a positive integer, got 0"),
        ({"k": -2}, ValueError, "k must be -1 or a positive integer, got -2"),
        (
            {"texts": ["今天", "明天"]},
            ValueError,
            "texts and audios must have the same length, got 2 and 1",
        ),
        (
            {"language": "zh-TW"},
            ValueError,
            "Unsupported language 'zh-TW'. Supported: zh, en, yue, ",
        ),
        ({"language": "ko"}, ValueError, "No default punctuations for language 'ko'"),
        (
            {"audios": [(np.zeros((2, SAMPLE_RATE), dtype=np.float32), SAMPLE_RATE)]},
            ValueError,
            re.escape("audios[0] must be a 1-D array, got shape (2, 16000)"),
        ),
        (
            {"audios": ["/no/such.wav"]},
            FileNotFoundError,
            "audios\\[0\\]: no such file: /no/such.wav",
        ),
    ],
    ids=[
        "str_texts",
        "k_zero",
        "k_below_minus_one",
        "length_mismatch",
        "regional_tag",
        "no_default_marks",
        "2d_array",
        "missing_file",
    ],
)
def test_add_punctuation_rejects_bad_input(
    asr_punctuator: ASRPunctuator,
    wav_path: Path,
    kwargs: dict,
    error: type[Exception],
    message: str,
) -> None:
    """Each failure in #26 raises its error with a message that names the problem."""
    call = {"texts": ["今天"], "audios": [str(wav_path)], **kwargs}

    with pytest.raises(error, match=message):
        asr_punctuator.add_punctuation(**call)


def test_bad_array_is_reported_before_a_missing_file(
    asr_punctuator: ASRPunctuator,
) -> None:
    """Every array is checked for 1-D before any path is checked for existence."""
    audios = [
        "/no/such.wav",
        (np.zeros((2, SAMPLE_RATE), dtype=np.float32), SAMPLE_RATE),
    ]

    with pytest.raises(ValueError, match=re.escape("audios[1] must be a 1-D array")):
        asr_punctuator.add_punctuation(["今天", "明天"], audios)


def test_missing_second_file_raises_before_any_audio_is_encoded(
    asr_punctuator: ASRPunctuator, wav_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All paths are checked before the first text runs."""
    encoder = FakeAudioEncoder()
    monkeypatch.setattr(asr_punctuator, "audio_encoder", encoder)

    with pytest.raises(FileNotFoundError, match="audios\\[1\\]"):
        asr_punctuator.add_punctuation(["今天", "明天"], [str(wav_path), "/no/such.wav"])

    assert encoder.calls == 0


def test_unsupported_model_raises_before_loading() -> None:
    """A name outside ASR_MODELS raises ValueError listing the supported names, and loads nothing."""
    with (
        patch("llm_punctuator.asr_punctuator.load_model") as load_model,
        patch("llm_punctuator.asr_punctuator.AutoTokenizer") as tokenizer_cls,
        pytest.raises(ValueError, match="Supported: Qwen/Qwen3-ASR-1.7B, Qwen/Qwen3-ASR-0.6B"),
    ):
        ASRPunctuator("Qwen/Qwen3-1.7B")

    assert not load_model.called
    assert not tokenizer_cls.from_pretrained.called
