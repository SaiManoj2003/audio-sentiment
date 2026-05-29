"""
Audio loading and preprocessing.

Centralises all file I/O and resampling so every downstream module
receives audio in a guaranteed format: mono float32 numpy array at 16kHz.
This matters because:
  - Whisper expects 16kHz
  - wav2vec2 expects 16kHz mono
  - librosa works best with float32
Normalising here means no module needs to handle format edge cases.
"""

import logging
from pathlib import Path

import librosa
import numpy as np

from audio_sentiment.config import cfg

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".webm"}


def load_audio(file_path: str | Path) -> tuple[np.ndarray, int]:
    """
    Load an audio file and return a mono float32 array resampled to 16kHz.

    Args:
        file_path: Path to the audio file.

    Returns:
        Tuple of (waveform, sample_rate) where waveform is shape (N,).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is not supported.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    if path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format '{path.suffix}'. "
            f"Supported: {sorted(SUPPORTED_FORMATS)}"
        )

    logger.info("Loading audio: %s", path.name)

    # librosa handles resampling, mono conversion, and float32 normalisation
    # in one call — resampling to target rate immediately saves memory
    waveform, sr = librosa.load(
        path,
        sr=cfg.audio.sample_rate,   # resample to 16kHz on load
        mono=True,                  # mix down to mono
        dtype=np.float32,
    )

    duration = len(waveform) / sr
    logger.info(
        "Loaded: %.2fs audio | sr=%dHz | samples=%d",
        duration, sr, len(waveform),
    )

    return waveform, sr


def slice_waveform(
    waveform: np.ndarray,
    start_sec: float,
    end_sec: float,
    sample_rate: int,
) -> np.ndarray:
    """
    Extract a segment from a waveform by time boundaries.

    Args:
        waveform:    Full audio array (mono float32).
        start_sec:   Segment start in seconds.
        end_sec:     Segment end in seconds.
        sample_rate: Sample rate of the waveform.

    Returns:
        Sliced waveform segment as float32 array.
    """
    start_sample = int(start_sec * sample_rate)
    end_sample = int(end_sec * sample_rate)

    # Clamp to waveform bounds — Whisper timestamps can occasionally
    # exceed actual audio length by a few milliseconds
    start_sample = max(0, start_sample)
    end_sample = min(len(waveform), end_sample)

    return waveform[start_sample:end_sample].astype(np.float32)


def get_duration(file_path: str | Path) -> float:
    """Return audio duration in seconds without loading the full waveform."""
    return librosa.get_duration(path=file_path)
