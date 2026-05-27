"""
Acoustic feature extraction using librosa.

Why we extract these features:
    Text models read words. But the same words spoken calmly vs angrily
    produce completely different audio signals. These features capture
    that signal mathematically:

    MFCC (Mel Frequency Cepstral Coefficients) 
        Represents the shape of the vocal tract

    Pitch (Fundamental Frequency / F0)
        How high or low the voice is.

    RMS Energy
        Loudness over time. Anger and joy tend to be louder. Sadness quieter.

    Zero Crossing Rate (ZCR)
        How often the signal crosses zero — related to how "noisy" or
        "voiced" the sound is.
These four together form a feature vector that captures prosody — the
rhythm, stress, and intonation of speech — which is largely invisible
to text-based models.
"""

import logging

import librosa
import numpy as np

from audio_sentiment.config import cfg

logger = logging.getLogger(__name__)


def extract_acoustic_features(waveform: np.ndarray, sample_rate: int) -> dict[str, float]:
    """
    Extract a summary acoustic feature vector from a waveform segment.

    We use statistical summaries (mean, std) of each feature over the
    segment rather than raw frame-level sequences. This gives us a
    fixed-size vector regardless of segment length — important because
    sentences vary in duration.

    Args:
        waveform:    Mono float32 audio array at 16kHz.
        sample_rate: Sample rate of the waveform (expected: 16000).

    Returns:
        Dict of named scalar features. Keys:
            mfcc_mean_1..40   — mean of each MFCC coefficient
            mfcc_std_1..40    — std dev of each MFCC coefficient
            pitch_mean        — mean fundamental frequency (Hz)
            pitch_std         — std dev of pitch
            energy_mean       — mean RMS energy
            energy_std        — std dev of RMS energy
            zcr_mean          — mean zero crossing rate
            zcr_std           — std dev of zero crossing rate
    """
    if len(waveform) == 0:
        logger.warning("Empty waveform passed to extract_acoustic_features.")
        return _zero_features()

    features = {}

    # ── MFCCs ────────────────────────────────────────────────────────────────
    mfccs = librosa.feature.mfcc(
        y=waveform,
        sr=sample_rate,
        n_mfcc=cfg.audio.n_mfcc,
        hop_length=cfg.audio.hop_length,
        n_fft=cfg.audio.n_fft,
    )
    for i, (mean, std) in enumerate(zip(mfccs.mean(axis=1), mfccs.std(axis=1)), start=1):
        features[f"mfcc_mean_{i}"] = float(mean)
        features[f"mfcc_std_{i}"] = float(std)

    # ── Pitch (F0) ───────────────────────────────────────────────────────────
    # pyin is more accurate than yin for speech; fills unvoiced frames with NaN
    f0, voiced_flag, _ = librosa.pyin(
        waveform,
        fmin=librosa.note_to_hz("C2"),   # ~65 Hz — below normal speech range
        fmax=librosa.note_to_hz("C7"),   # ~2093 Hz — above normal speech range
        sr=sample_rate,
    )
    voiced_f0 = f0[voiced_flag] if voiced_flag is not None else np.array([])
    features["pitch_mean"] = float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0
    features["pitch_std"] = float(np.std(voiced_f0)) if len(voiced_f0) > 0 else 0.0

    # ── RMS Energy ───────────────────────────────────────────────────────────
    rms = librosa.feature.rms(
        y=waveform,
        hop_length=cfg.audio.hop_length,
    )[0]
    features["energy_mean"] = float(np.mean(rms))
    features["energy_std"] = float(np.std(rms))

    # ── Zero Crossing Rate ───────────────────────────────────────────────────
    zcr = librosa.feature.zero_crossing_rate(
        y=waveform,
        hop_length=cfg.audio.hop_length,
    )[0]
    features["zcr_mean"] = float(np.mean(zcr))
    features["zcr_std"] = float(np.std(zcr))

    return features


def _zero_features() -> dict[str, float]:
    """Return a zero-valued feature dict for empty/silent segments."""
    features = {}
    for i in range(1, cfg.audio.n_mfcc + 1):
        features[f"mfcc_mean_{i}"] = 0.0
        features[f"mfcc_std_{i}"] = 0.0
    features.update({
        "pitch_mean": 0.0,
        "pitch_std": 0.0,
        "energy_mean": 0.0,
        "energy_std": 0.0,
        "zcr_mean": 0.0,
        "zcr_std": 0.0,
    })
    return features


def features_to_array(features: dict[str, float]) -> np.ndarray:
    """
    Convert a feature dict to a sorted numpy array.

    Sorting by key ensures consistent ordering across calls —
    important when passing to a downstream model or classifier.

    Args:
        features: Output of extract_acoustic_features().

    Returns:
        1D float32 numpy array of feature values.
    """
    return np.array(
        [features[k] for k in sorted(features.keys())],
        dtype=np.float32,
    )
