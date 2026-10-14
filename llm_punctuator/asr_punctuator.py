"""Punctuate text against its audio with a Qwen3-ASR model."""

import numpy as np
import soundfile
import soxr

SAMPLE_RATE = 16000


def load_audio(audio: str | tuple[np.ndarray, int]) -> np.ndarray:
    """Load audio as a mono 16 kHz float32 waveform in [-1, 1].

    Mirrors qwen_asr's ``normalize_audio_input``, but reads files with soundfile and
    resamples with soxr. The caller has checked that a path exists and an array is 1-D.

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
