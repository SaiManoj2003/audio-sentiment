"""
Text-based emotion classification.

Model: j-hartmann/emotion-english-distilroberta-base
- Fine-tuned on 6 datasets covering social media, dialogue, and news
- Predicts 7 emotion classes: anger, disgust, fear, joy, neutral, sadness, surprise
"""

import logging
from functools import lru_cache

import numpy as np
from transformers import pipeline as hf_pipeline

from audio_sentiment.config import cfg

logger = logging.getLogger(__name__)

# Canonical emotion labels from this model
EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

# Compound sentiment score weights — maps each emotion to [-1, 1] scale
# Used when we need a single scalar score for plotting timelines
EMOTION_VALENCE = {
    "anger":    -0.8,
    "disgust":  -0.7,
    "fear":     -0.5,
    "sadness":  -0.6,
    "neutral":   0.0,
    "surprise":  0.2,
    "joy":       0.9,
}


@lru_cache(maxsize=1)
def _get_pipeline():
    """
    Load and cache the text emotion pipeline.

    lru_cache ensures the model is loaded exactly once per process —
    the model is ~300MB and takes ~4s to load, so per-call loading
    would make the API unusable.
    """
    logger.info("Loading text emotion model: %s", cfg.text_emotion.model_name)
    pipe = hf_pipeline(
        task="text-classification",
        model=cfg.text_emotion.model_name,
        top_k=None,
        device=0 if cfg.text_emotion.device == "cuda" else -1,
        truncation=True,
        max_length=cfg.text_emotion.max_length,
    )
    logger.info("Text emotion model loaded.")
    return pipe


def classify_text_emotion(text: str) -> dict[str, float]:
    """
    Classify emotion from text and return a probability dict.

    Args:
        text: A single sentence or utterance.

    Returns:
        Dict mapping each emotion label to its probability.
        Probabilities sum to 1.0.
    """
    if not text or not text.strip():
        logger.warning("Empty text passed to classify_text_emotion — returning neutral.")
        return {e: (1.0 if e == "neutral" else 0.0) for e in EMOTIONS}

    pipe = _get_pipeline()
    raw = pipe(text.strip())[0]  # list of {label, score} dicts

    probs = {item["label"].lower(): item["score"] for item in raw}

    return {emotion: probs.get(emotion, 0.0) for emotion in EMOTIONS}


def emotion_to_valence(emotion_probs: dict[str, float]) -> float:
    """
    Convert emotion probability dict to a single compound valence score.

    Score is a weighted sum of each emotion's valence scaled by its
    probability. Result lies in approximately [-1, 1].

    Args:
        emotion_probs: Output of classify_text_emotion().

    Returns:
        Float in [-1, 1]. Negative = negative sentiment, positive = positive.
    """
    score = sum(
        EMOTION_VALENCE[emotion] * prob
        for emotion, prob in emotion_probs.items()
    )
    # Clamp to [-1, 1] to absorb any floating point edge cases
    return float(np.clip(score, -1.0, 1.0))


def dominant_emotion(emotion_probs: dict[str, float]) -> str:
    """Return the emotion label with the highest probability."""
    return max(emotion_probs, key=emotion_probs.get)
