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

EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

EMOTION_VALENCE = {
    "anger":    -0.8,
    "disgust":  -0.7,
    "fear":     -0.5,
    "sadness":  -0.6,
    "neutral":   0.0,
    "surprise":  0.2,
    "joy":       0.9,
}

_NEUTRAL_PROBS = {e: (1.0 if e == "neutral" else 0.0) for e in EMOTIONS}


@lru_cache(maxsize=1)
def _get_pipeline():
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


def _parse_raw(raw: list[dict]) -> dict[str, float]:
    """Normalise one pipeline output item into a canonical prob dict."""
    probs = {item["label"].strip().lower(): float(item["score"]) for item in raw}
    result = {e: probs.get(e, 0.0) for e in EMOTIONS}
    total = sum(result.values())
    if total > 0:
        return {k: v / total for k, v in result.items()}
    return dict(_NEUTRAL_PROBS)


def classify_text_emotion(text: str) -> dict[str, float]:
    """
    Classify emotion from a single sentence.

    Args:
        text: A single sentence or utterance.

    Returns:
        Dict mapping each emotion label to its probability. Sums to 1.0.
    """
    if not text or not text.strip():
        logger.warning("Empty text — returning neutral.")
        return dict(_NEUTRAL_PROBS)

    pipe = _get_pipeline()
    return _parse_raw(pipe(text.strip())[0])


def classify_text_emotion_batch(texts: list[str], batch_size: int = 16) -> list[dict[str, float]]:
    """
    Classify emotion for a list of sentences in batched GPU passes.

    Batching reduces GPU round-trips from O(n) to O(n/batch_size),
    typically 8-12x throughput improvement over per-sentence calls.

    Args:
        texts:      List of sentences to classify.
        batch_size: Number of sentences per GPU forward pass.

    Returns:
        List of emotion probability dicts, one per input sentence.
    """
    if not texts:
        return []

    pipe = _get_pipeline()

    results = []
    valid_indices = []
    valid_texts = []

    for i, text in enumerate(texts):
        if text and text.strip():
            valid_indices.append(i)
            valid_texts.append(text.strip())

    # Build placeholder results for empty inputs
    output: list[dict[str, float]] = [dict(_NEUTRAL_PROBS) for _ in texts]

    if not valid_texts:
        return output

    # Single batched forward pass — the pipeline handles chunking internally
    batch_outputs = pipe(valid_texts, batch_size=batch_size)

    for idx, raw in zip(valid_indices, batch_outputs):
        output[idx] = _parse_raw(raw)

    return output


def emotion_to_valence(emotion_probs: dict[str, float]) -> float:
    """
    Convert emotion probability dict to a compound valence score in [-1, 1].

    Args:
        emotion_probs: Output of classify_text_emotion().

    Returns:
        Float in [-1, 1]. Negative = negative sentiment, positive = positive.
    """
    score = sum(
        EMOTION_VALENCE[emotion] * prob
        for emotion, prob in emotion_probs.items()
    )
    return float(np.clip(score, -1.0, 1.0))


def dominant_emotion(emotion_probs: dict[str, float]) -> str:
    """Return the emotion label with the highest probability."""
    return max(emotion_probs, key=emotion_probs.get)
