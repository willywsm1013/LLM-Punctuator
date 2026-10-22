"""Punctuate text against its audio with a Qwen3-ASR model."""

import itertools
import json
import math
from pathlib import Path

import numpy as np
import soundfile
import soxr
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import (
    AttentionInterface,
    AutoTokenizer,
    DynamicCache,
    Qwen3Config,
    Qwen3ForCausalLM,
    WhisperFeatureExtractor,
)
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoderConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeAudioEncoder

from llm_punctuator.logits_processor import DistanceBonusProcessor
from llm_punctuator.punctuator import ForcedPunctuator

SAMPLE_RATE = 16000

ASR_MODELS = ("Qwen/Qwen3-ASR-1.7B", "Qwen/Qwen3-ASR-0.6B")

QWEN3_ASR_LANGUAGES = {
    "zh": "Chinese",
    "en": "English",
    "yue": "Cantonese",
    "ar": "Arabic",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "id": "Indonesian",
    "it": "Italian",
    "ko": "Korean",
    "ru": "Russian",
    "th": "Thai",
    "vi": "Vietnamese",
    "ja": "Japanese",
    "tr": "Turkish",
    "hi": "Hindi",
    "ms": "Malay",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "pl": "Polish",
    "cs": "Czech",
    "fil": "Filipino",
    "fa": "Persian",
    "el": "Greek",
    "ro": "Romanian",
    "hu": "Hungarian",
    "mk": "Macedonian",
}


def load_audio(audio: str | tuple[np.ndarray, int]) -> np.ndarray:
    """Load audio as a mono 16 kHz float32 waveform in [-1, 1].

    The caller has checked that a path exists and an array is 1-D.

    Args:
        audio: A file path, or a ``(waveform, sample_rate)`` pair.

    Returns:
        A 1-D float32 array at ``SAMPLE_RATE``.
    """
    if isinstance(audio, str):
        waveform, sample_rate = soundfile.read(audio, dtype="float32")
    else:
        waveform, sample_rate = audio
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        waveform = soxr.resample(waveform, sample_rate, SAMPLE_RATE, quality="HQ")
    peak = float(np.abs(waveform).max())
    if peak > 1.0:
        waveform = waveform / peak
    return np.clip(waveform, -1.0, 1.0)


def build_prompt(context: str, text: str, n_audio: int, language: str) -> str:
    """Build the Qwen3-ASR prompt that ends where the transcript starts.

    The audio pad is expanded to one pad per audio vector.

    Args:
        context: System message template. Only ``{text}`` is replaced; other braces stay.
        text: The text to punctuate, filled into ``{text}``.
        n_audio: Number of audio vectors, one ``<|audio_pad|>`` each.
        language: Language tag, a key of ``QWEN3_ASR_LANGUAGES``.

    Returns:
        The prompt string.
    """
    return (
        f"<|im_start|>system\n{context.replace('{text}', text)}<|im_end|>\n"
        f"<|im_start|>user\n<|audio_start|>{'<|audio_pad|>' * n_audio}<|audio_end|><|im_end|>\n"
        f"<|im_start|>assistant\nlanguage {QWEN3_ASR_LANGUAGES[language]}<asr_text>"
    )


def load_model(model_name: str) -> tuple[Qwen3OmniMoeAudioEncoder, Qwen3ForCausalLM, int]:
    """Build a Qwen3-ASR checkpoint from transformers' audio encoder and Qwen3 LM.

    Args:
        model_name: HuggingFace repo of a Qwen3-ASR checkpoint.

    Returns:
        The audio encoder, the language model, and the ``<|audio_pad|>`` token ID.
    """
    path = Path(snapshot_download(model_name))
    config = json.loads((path / "config.json").read_text())["thinker_config"]
    weights = {}
    for file in sorted(path.glob("*.safetensors")):
        weights.update(load_file(file))

    audio_config = Qwen3OmniMoeAudioEncoderConfig(**_without_model_type(config["audio_config"]))
    text_config = Qwen3Config(**_without_model_type(config["text_config"]))
    audio_config._attn_implementation = "windowed_attention"
    text_config._attn_implementation = "sdpa"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, config["dtype"])
    with torch.device(device):
        audio_encoder = Qwen3OmniMoeAudioEncoder._from_config(audio_config, dtype=dtype).eval()
        model = Qwen3ForCausalLM._from_config(text_config, dtype=dtype).eval()

    audio_encoder.load_state_dict(_strip_prefix(weights, "thinker.audio_tower."), strict=True)
    lm_weights = {"model." + k: v for k, v in _strip_prefix(weights, "thinker.model.").items()}
    lm_weights["lm_head.weight"] = weights["thinker.lm_head.weight"]
    model.load_state_dict(lm_weights, strict=True)
    return audio_encoder, model, config["audio_token_id"]


def windowed_attention(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    scaling: float,
    cu_seq_lens_q: torch.Tensor,
    **kwargs: object,
) -> tuple[torch.Tensor, None]:
    """Attend within each window of cu_seq_lens_q; windows of one length go to sdpa as one batch.

    Args:
        module: The attention module that calls this.
        query: Shape (1, heads, sequence, head_dim). So are key and value.
        key: Keys.
        value: Values.
        attention_mask: Unused. The windows come from cu_seq_lens_q.
        scaling: Scale of the attention scores.
        cu_seq_lens_q: Cumulative window lengths, starting at 0.
        **kwargs: Other arguments of the attention interface, unused.

    Returns:
        The output, shaped (1, sequence, heads, head_dim), and None for the weights.
    """
    lengths = (cu_seq_lens_q[1:] - cu_seq_lens_q[:-1]).tolist()
    outputs, start = [], 0
    for length, run in itertools.groupby(lengths):
        count = len(list(run))
        end = start + count * length
        q, k, v = (
            t[0, :, start:end].unflatten(1, (count, length)).transpose(0, 1)
            for t in (query, key, value)
        )
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scaling)
        outputs.append(out.transpose(1, 2).flatten(0, 1))
        start = end
    return torch.cat(outputs)[None], None


AttentionInterface.register("windowed_attention", windowed_attention)


def _without_model_type(config: dict) -> dict:
    """Return the config without its model_type."""
    return {k: v for k, v in config.items() if k != "model_type"}


def _strip_prefix(weights: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    """Return the weights under prefix, keyed without it."""
    return {k.removeprefix(prefix): v for k, v in weights.items() if k.startswith(prefix)}


class ASRPunctuator(ForcedPunctuator):
    """Puts punctuation into a known text while a Qwen3-ASR model hears its audio."""

    audio_encoder: Qwen3OmniMoeAudioEncoder
    audio_token_id: int
    feature_extractor: WhisperFeatureExtractor
    pending_audio: torch.Tensor

    def __init__(self, model_name: str) -> None:
        """Load a Qwen3-ASR model.

        Args:
            model_name: One of ``ASR_MODELS``.

        Raises:
            ValueError: If model_name is not in ``ASR_MODELS``. Raised before any download.
        """
        if model_name not in ASR_MODELS:
            raise ValueError(
                f"Unsupported model {model_name!r}. Supported: {', '.join(ASR_MODELS)}"
            )
        self.audio_encoder, self.model, self.audio_token_id = load_model(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
        self.device = self.model.device
        self.pending_audio = torch.empty(0)

    def encode_audio(self, wav: np.ndarray) -> torch.Tensor:
        """Turn a waveform into audio vectors.

        Args:
            wav: Mono float32 waveform at ``SAMPLE_RATE``.

        Returns:
            Audio vectors, shaped (number of vectors, hidden size).
        """
        features = self.feature_extractor(
            wav,
            sampling_rate=SAMPLE_RATE,
            padding="longest",
            truncation=False,
            return_attention_mask=True,
            return_tensors="pt",
        )
        n_frames = int(features.attention_mask.sum())
        input_features = features.input_features[0, :, :n_frames].to(self.device, self.model.dtype)
        feature_lens = torch.tensor([n_frames], device=self.device)
        return self.audio_encoder(input_features, feature_lens=feature_lens).last_hidden_state

    def char_counts(self, tokens: list[int]) -> list[int]:
        """Count the non-whitespace characters the first p tokens decode to, for each p.

        A token that ends inside a character decodes it as one replacement character, as the
        whole prefix would. Decoding restarts after each token that does not, so the cost grows
        linearly with the text.

        Args:
            tokens: Token IDs of a text.

        Returns:
            The counts for p from 0 to len(tokens).
        """
        counts, start = [0], 0
        for end in range(1, len(tokens) + 1):
            piece = self.decode(tokens[start:end])
            counts.append(counts[start] + sum(not c.isspace() for c in piece))
            if not piece.endswith("\ufffd"):
                start = end
        return counts

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: DynamicCache,
    ) -> torch.Tensor:
        """Run the model once with the pending audio vectors at the audio pads.

        The first forward of a text uses up the audio vectors, so later forwards feed none.

        Args:
            input_ids: Token IDs to feed, shape (batch, width).
            attention_mask: 2D mask over the cache and the fed tokens.
            position_ids: Position of each fed token.
            cache_position: Cache slot of each fed token.
            past_key_values: Cache that this call extends.

        Returns:
            Logits of each fed token.

        Raises:
            ValueError: If the number of audio pads differs from the number of audio vectors.
        """
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        pads = input_ids == self.audio_token_id
        n_pads, n_vectors = int(pads.sum()), len(self.pending_audio)
        if n_pads != n_vectors:
            raise ValueError(f"{n_pads} audio pads but {n_vectors} audio vectors")
        inputs_embeds[pads] = self.pending_audio.to(inputs_embeds.dtype)
        self.pending_audio = self.pending_audio[:0]
        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=past_key_values,
            use_cache=True,
        ).logits

    def _check_inputs(
        self,
        texts: list[str],
        audios: list[str | tuple[np.ndarray, int]],
        language: str,
        punctuations: str | None,
        k: int,
        mark_bonus_per_char: float,
    ) -> None:
        """Raise on the first bad input. Reads no audio and runs no model."""
        if isinstance(texts, str):
            raise TypeError("texts must be a list of str, got str. Wrap a single text as [text].")
        if k == 0 or k < -1:
            raise ValueError(f"k must be -1 or a positive integer, got {k}")
        if not (math.isfinite(mark_bonus_per_char) and mark_bonus_per_char >= 0):
            raise ValueError(
                f"mark_bonus_per_char must be finite and >= 0, got {mark_bonus_per_char}"
            )
        if len(texts) != len(audios):
            raise ValueError(
                f"texts and audios must have the same length, got {len(texts)} and {len(audios)}"
            )
        if language not in QWEN3_ASR_LANGUAGES:
            raise ValueError(
                f"Unsupported language {language!r}. Supported: {', '.join(QWEN3_ASR_LANGUAGES)}"
            )
        if punctuations is None:
            try:
                self.default_punctuations(language)
            except ValueError as error:
                raise ValueError(
                    f"No default punctuations for language {language!r}. Pass punctuations."
                ) from error
        for i, audio in enumerate(audios):
            if not isinstance(audio, str) and np.ndim(audio[0]) != 1:
                raise ValueError(f"audios[{i}] must be a 1-D array, got shape {np.shape(audio[0])}")
        for i, audio in enumerate(audios):
            if isinstance(audio, str) and not Path(audio).is_file():
                raise FileNotFoundError(f"audios[{i}]: no such file: {audio}")

    @torch.no_grad()
    def add_punctuation(
        self,
        texts: list[str],
        audios: list[str | tuple[np.ndarray, int]],
        language: str = "zh",
        context: str = "{text}",
        punctuations: str | None = None,
        k: int = -1,
        mark_bonus_per_char: float = 0.0,
    ) -> list[str]:
        """Add punctuation to each text while the model hears its audio, one text at a time.

        Args:
            texts: Input texts without punctuation.
            audios: The audio of each text, as a file path or a ``(waveform, sample_rate)`` pair.
            language: Language tag, a key of ``QWEN3_ASR_LANGUAGES``. Default: "zh".
            context: System message template. ``{text}`` is replaced by the text.
                Default: "{text}".
            punctuations: String of allowed punctuation characters. If None, uses default for
                language.
            k: Most positions where a mark may go to decide per forward pass. -1 decides the
                whole text. Default: -1.
            mark_bonus_per_char: Where a mark may go, every mark's score gets this times the
                non-whitespace characters since the last mark, or since the start of the text.
                Default: 0.0.

        Returns:
            Each text with punctuation added, in the order of texts.

        Raises:
            TypeError: If texts is a str.
            ValueError: If k is 0 or below -1, mark_bonus_per_char is negative or not finite, texts
                and audios differ in length, language is not in ``QWEN3_ASR_LANGUAGES``,
                language has no default punctuations and punctuations is None, or an array is
                not 1-D.
            FileNotFoundError: If an audio path is not a file. Every input is checked before
                the first audio is read.
        """
        self._check_inputs(texts, audios, language, punctuations, k, mark_bonus_per_char)
        if punctuations is None:
            punctuations = self.default_punctuations(language)
        punctuation_tokens = self.punctuation_token_ids(punctuations)

        results = []
        for text, audio in zip(texts, audios, strict=True):
            self.pending_audio = self.encode_audio(load_audio(audio))
            prompt = build_prompt(context, text, len(self.pending_audio), language)
            prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            segments = self.split_segments(text, language)
            tokens = [token for segment in segments for token in segment]
            rule = DistanceBonusProcessor(
                tokens + [self.tokenizer.eos_token_id],
                punctuation_tokens,
                has_prev_input=False,
                unit_ends=set(itertools.accumulate(len(segment) for segment in segments)),
                char_counts=self.char_counts(tokens),
                mark_bonus_per_char=mark_bonus_per_char,
            )
            [generated] = self.greedy_search([(prompt_ids, rule)], k)
            results.append(self.decode(generated))
        return results
