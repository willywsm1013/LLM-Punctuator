"""Tests for ASRPunctuator: audio input, prompt, and forced decoding."""

from pathlib import Path

import numpy as np
import soundfile

from llm_punctuator.asr_punctuator import SAMPLE_RATE, load_audio


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
