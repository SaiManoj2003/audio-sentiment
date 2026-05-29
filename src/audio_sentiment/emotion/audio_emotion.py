"""
Audio-based emotion classification using wav2vec2.

Model: r-f/wav2vec-english-speech-emotion-recognition
    - Fine-tuned on RAVDESS + SAVEE + TESS
    - 7 emotion classes matching our canonical schema
    - Must be loaded with Wav2Vec2ForSequenceClassification explicitly
      (AutoModelForAudioClassification uses a different classifier head
       architecture causing weight mismatches and random predictions)
"""

import logging
from functools import lru_cache

import numpy as np
import torch
from transformers import (
    AutoFeatureExtractor,
    Wav2Vec2ForSequenceClassification,
)

from audio_sentiment.config import cfg

logger = logging.getLogger(__name__)

EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

_LABEL_MAP = {
    "angry":    "anger",
    "disgust":  "disgust",
    "fear":     "fear",
    "happy":    "joy",
    "neutral":  "neutral",
    "sad":      "sadness",
    "surprise": "surprise",
}


@lru_cache(maxsize=1)
def _get_model_and_extractor():
    """
    Load wav2vec2 using Wav2Vec2ForSequenceClassification — the exact class
    this checkpoint was saved with. Using AutoModelForAudioClassification
    causes classifier head weight mismatches and random predictions.
    """
    model_name = "r-f/wav2vec-english-speech-emotion-recognition"
    logger.info("Loading audio emotion model: %s", model_name)

    extractor = AutoFeatureExtractor.from_pretrained(model_name)

    # Explicit class load — no architecture mismatch
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)

    device = torch.device(
        "cuda" if cfg.audio_emotion.device == "cuda"
        and torch.cuda.is_available() else "cpu"
    )
    model = model.to(device)
    model.eval()

    logger.info("Audio emotion model loaded on %s.", device)
    return model, extractor, device


def classify_audio_emotion(waveform: np.ndarray) -> dict[str, float]:
    """
    Classify emotion from a raw audio waveform.

    Args:
        waveform: Mono float32 audio array at 16kHz.

    Returns:
        Dict mapping each canonical emotion label to its probability.
        Probabilities sum to 1.0.
    """
    if len(waveform) < 160:
        logger.warning(
            "Waveform too short (%d samples) — returning neutral.", len(waveform)
        )
        return {e: (1.0 if e == "neutral" else 0.0) for e in EMOTIONS}

    model, extractor, device = _get_model_and_extractor()

    inputs = extractor(
        waveform,
        sampling_rate=cfg.audio_emotion.sample_rate,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
    id2label = model.config.id2label

    canonical_probs: dict[str, float] = {e: 0.0 for e in EMOTIONS}
    for i, prob in enumerate(probs):
        raw_label = id2label[i].lower()
        canonical = _LABEL_MAP.get(raw_label, raw_label)
        if canonical in canonical_probs:
            canonical_probs[canonical] += float(prob)

    total = sum(canonical_probs.values())
    if total > 0:
        canonical_probs = {k: v / total for k, v in canonical_probs.items()}

    return canonical_probs
