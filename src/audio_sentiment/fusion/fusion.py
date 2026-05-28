"""
Late fusion layer.

Why late fusion vs early fusion:
    Early fusion: combine raw features (MFCCs + text embeddings) before
    classification. Requires a single jointly trained model — not feasible
    here since our text and audio models are pretrained separately.

    Late fusion: run each branch independently to get emotion probabilities,
    then combine the probability distributions. This is what we do.

    Late fusion advantages for this project:
        1. Works with any pretrained models without retraining
        2. Each branch can be swapped independently
        3. Weights are interpretable — we can say exactly how much
           each modality contributed to the final prediction
        4. Easy to ablate: set text_weight=1.0 to get text-only baseline,
           set audio_weight=1.0 for audio-only baseline

Fusion formula:
    final_prob[emotion] = (text_weight * text_prob[emotion])
                        + (audio_weight * audio_prob[emotion])

    With default weights 0.5/0.5 this is a simple average.
    Weights are configurable in config.py for experimentation.
"""

import logging

import numpy as np

from audio_sentiment.config import cfg
from audio_sentiment.emotion.text_emotion import (
    EMOTIONS,
    EMOTION_VALENCE,
    dominant_emotion,
)

logger = logging.getLogger(__name__)


def fuse_emotions(
    text_probs: dict[str, float],
    audio_probs: dict[str, float],
    text_weight: float | None = None,
    audio_weight: float | None = None,
) -> dict[str, float]:
    """
    Combine text and audio emotion probability dicts via weighted average.

    Args:
        text_probs:   Output of classify_text_emotion().
        audio_probs:  Output of classify_audio_emotion().
        text_weight:  Weight for text branch. Defaults to cfg.fusion.text_weight.
        audio_weight: Weight for audio branch. Defaults to cfg.fusion.audio_weight.

    Returns:
        Fused probability dict. Keys are canonical emotion labels.
        Values sum to 1.0.

    Raises:
        ValueError: If weights do not sum to 1.0.
    """
    tw = text_weight if text_weight is not None else cfg.fusion.text_weight
    aw = audio_weight if audio_weight is not None else cfg.fusion.audio_weight

    if abs(tw + aw - 1.0) > 1e-6:
        raise ValueError(
            f"text_weight ({tw}) + audio_weight ({aw}) must equal 1.0"
        )

    fused = {}
    for emotion in EMOTIONS:
        t_prob = text_probs.get(emotion, 0.0)
        a_prob = audio_probs.get(emotion, 0.0)
        fused[emotion] = (tw * t_prob) + (aw * a_prob)

    # Renormalise — floating point arithmetic can push sum slightly off 1.0
    total = sum(fused.values())
    if total > 0:
        fused = {k: v / total for k, v in fused.items()}

    return fused


def compute_valence(emotion_probs: dict[str, float]) -> float:
    """
    Compute a compound valence score in [-1, 1] from emotion probabilities.

    Uses psychologically grounded valence weights from EMOTION_VALENCE.
    This single scalar is used for timeline plotting.

    Args:
        emotion_probs: Any emotion probability dict — text, audio, or fused.

    Returns:
        Float in [-1, 1]. Negative = negative sentiment, positive = positive.
    """
    score = sum(
        EMOTION_VALENCE[emotion] * prob
        for emotion, prob in emotion_probs.items()
        if emotion in EMOTION_VALENCE
    )
    return float(np.clip(score, -1.0, 1.0))


def escalation_flag(
    valence_scores: list[float],
    threshold: float = -0.5,
    window: int = 3,
) -> bool:
    """
    Detect whether a speaker's sentiment is sustained negative.

    Business value: flags calls where a customer stays angry for multiple
    consecutive sentences — a signal that the agent may need supervisor
    support or that the call is at risk of ending badly.

    Logic: returns True if the rolling mean of the last `window` sentences
    stays below `threshold`.

    Args:
        valence_scores: Ordered list of valence scores for a speaker.
        threshold:      Valence below this is considered negative. Default -0.5.
        window:         Number of consecutive sentences to check. Default 3.

    Returns:
        True if escalation detected, False otherwise.
    """
    if len(valence_scores) < window:
        return False

    recent = valence_scores[-window:]
    return float(np.mean(recent)) < threshold


def summarise_speaker(
    valence_scores: list[float],
    emotion_probs_list: list[dict[str, float]],
    speaker_id: int,
) -> dict:
    """
    Produce a structured summary for one speaker across the full call.

    Args:
        valence_scores:     Ordered valence score per sentence.
        emotion_probs_list: Ordered emotion prob dict per sentence.
        speaker_id:         0 or 1.

    Returns:
        Dict with average_valence, dominant_emotion, escalation_detected,
        sentence_count, sentiment_label.
    """
    if not valence_scores:
        return {
            "speaker_id": speaker_id,
            "sentence_count": 0,
            "average_valence": 0.0,
            "dominant_emotion": "neutral",
            "sentiment_label": "Neutral",
            "escalation_detected": False,
        }

    avg_valence = float(np.mean(valence_scores))

    # Average emotion probabilities across all sentences for this speaker
    avg_probs: dict[str, float] = {e: 0.0 for e in EMOTIONS}
    for probs in emotion_probs_list:
        for emotion, prob in probs.items():
            avg_probs[emotion] += prob
    avg_probs = {k: v / len(emotion_probs_list) for k, v in avg_probs.items()}

    sentiment_label = (
        "Positive" if avg_valence > 0.05
        else "Negative" if avg_valence < -0.05
        else "Neutral"
    )

    return {
        "speaker_id": speaker_id,
        "sentence_count": len(valence_scores),
        "average_valence": round(avg_valence, 4),
        "dominant_emotion": dominant_emotion(avg_probs),
        "sentiment_label": sentiment_label,
        "escalation_detected": escalation_flag(valence_scores),
    }
