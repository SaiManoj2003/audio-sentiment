"""
Audio-based emotion classification using wav2vec2.

Model: ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition
    - Fine-tuned on RAVDESS dataset
    - Takes raw waveform directly — no manual feature engineering needed
    - Predicts 8 emotion classes from audio signal alone

Why wav2vec2 on top of librosa features:
    librosa features (MFCCs, pitch, energy) are hand-crafted — humans
    decided these matter. wav2vec2 learned its own representations from
    thousands of hours of speech data. Using both gives us:
        - librosa: interpretable, fast, proven features
        - wav2vec2: deep learned representations the model discovered
    The fusion layer combines both with the text branch for final prediction.

Label mapping:
    The model was trained on RAVDESS which uses these 8 labels.
    We map them to our 7 canonical emotions for consistency with
    the text branch — 'calm' maps to 'neutral'.
"""

import logging
from functools import lru_cache

import numpy as np
import torch
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
)

from audio_sentiment.config import cfg

logger = logging.getLogger(__name__)

# RAVDESS labels from this model → our canonical emotion labels
_LABEL_MAP = {
    "angry":     "anger",
    "calm":      "neutral",
    "disgust":   "disgust",
    "fearful":   "fear",
    "happy":     "joy",
    "neutral":   "neutral",
    "sad":       "sadness",
    "surprised": "surprise",
}

EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]


@lru_cache(maxsize=1)
def _get_model_and_extractor():
    """
    Load and cache the wav2vec2 model and feature extractor.

    Loaded once at first call, reused for all subsequent calls.
    Model is ~1.2GB — per-call loading would be unusable in practice.
    """
    logger.info("Loading audio emotion model: %s", cfg.audio_emotion.model_name)

    extractor = AutoFeatureExtractor.from_pretrained(cfg.audio_emotion.model_name)
    model = AutoModelForAudioClassification.from_pretrained(cfg.audio_emotion.model_name)

    device = torch.device("cuda" if cfg.audio_emotion.device == "cuda"
                          and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    logger.info("Audio emotion model loaded on %s.", device)
    return model, extractor, device


def classify_audio_emotion(waveform: np.ndarray) -> dict[str, float]:
    """
    Classify emotion from a raw audio waveform.

    Args:
        waveform: Mono float32 audio array at 16kHz.
                  Should be a single sentence segment, not full call audio.

    Returns:
        Dict mapping each canonical emotion label to its probability.
        Probabilities sum to 1.0.

    Example:
        >>> classify_audio_emotion(sentence.waveform)
        {
            'anger': 0.65,
            'disgust': 0.08,
            'fear': 0.03,
            'joy': 0.02,
            'neutral': 0.12,
            'sadness': 0.07,
            'surprise': 0.03
        }
    """
    if len(waveform) < 160:
        # Segment too short — wav2vec2 needs at least 10ms of audio
        logger.warning("Waveform too short (%d samples) — returning neutral.", len(waveform))
        return {e: (1.0 if e == "neutral" else 0.0) for e in EMOTIONS}

    model, extractor, device = _get_model_and_extractor()

    # Feature extractor handles normalisation and padding
    inputs = extractor(
        waveform,
        sampling_rate=cfg.audio_emotion.sample_rate,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    # Softmax to convert logits → probabilities
    probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

    # Map model's label IDs to canonical emotion names
    id2label = model.config.id2label
    raw_probs = {
        _LABEL_MAP.get(id2label[i].lower(), id2label[i].lower()): float(probs[i])
        for i in range(len(probs))
    }

    # Merge probabilities for labels that map to the same canonical emotion
    # e.g. both 'calm' and 'neutral' → 'neutral'
    canonical_probs: dict[str, float] = {e: 0.0 for e in EMOTIONS}
    for label, prob in raw_probs.items():
        if label in canonical_probs:
            canonical_probs[label] += prob

    # Renormalise to sum to 1.0 after merging
    total = sum(canonical_probs.values())
    if total > 0:
        canonical_probs = {k: v / total for k, v in canonical_probs.items()}

    return canonical_probs
