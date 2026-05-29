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

_NEUTRAL_PROBS = {e: (1.0 if e == "neutral" else 0.0) for e in EMOTIONS}


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
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)

    device = torch.device(
        "cuda" if cfg.audio_emotion.device == "cuda"
        and torch.cuda.is_available() else "cpu"
    )
    model = model.to(device)
    model.eval()

    logger.info("Audio emotion model loaded on %s.", device)
    return model, extractor, device


def _logits_to_probs(logits: torch.Tensor, id2label: dict) -> dict[str, float]:
    """Convert a single logits row to a canonical probability dict."""
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    canonical: dict[str, float] = {e: 0.0 for e in EMOTIONS}
    for i, prob in enumerate(probs):
        raw_label = id2label[i].lower()
        canonical_label = _LABEL_MAP.get(raw_label, raw_label)
        if canonical_label in canonical:
            canonical[canonical_label] += float(prob)
    total = sum(canonical.values())
    if total > 0:
        return {k: v / total for k, v in canonical.items()}
    return dict(_NEUTRAL_PROBS)


def classify_audio_emotion(waveform: np.ndarray) -> dict[str, float]:
    """
    Classify emotion from a raw audio waveform.

    Args:
        waveform: Mono float32 audio array at 16kHz.

    Returns:
        Dict mapping each canonical emotion label to its probability. Sums to 1.0.
    """
    if len(waveform) < 160:
        logger.warning("Waveform too short (%d samples) — returning neutral.", len(waveform))
        return dict(_NEUTRAL_PROBS)

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

    return _logits_to_probs(logits.squeeze(0), model.config.id2label)


def classify_audio_emotion_batch(
    waveforms: list[np.ndarray],
    batch_size: int = 8,
) -> list[dict[str, float]]:
    """
    Classify emotion for a list of waveforms in batched GPU passes.

    wav2vec2 processes variable-length audio — the extractor pads each
    batch to the longest item in that batch, so smaller batch sizes reduce
    padding waste on heterogeneous segment lengths.

    Args:
        waveforms:  List of mono float32 waveforms at 16kHz.
        batch_size: Waveforms per forward pass. Default 8 balances
                    GPU utilisation vs padding overhead.

    Returns:
        List of emotion probability dicts, one per input waveform.
    """
    if not waveforms:
        return []

    model, extractor, device = _get_model_and_extractor()
    id2label = model.config.id2label

    results: list[dict[str, float]] = []

    for batch_start in range(0, len(waveforms), batch_size):
        batch = waveforms[batch_start : batch_start + batch_size]

        valid = [(i, w) for i, w in enumerate(batch) if len(w) >= 160]
        output_batch: list[dict[str, float]] = [dict(_NEUTRAL_PROBS)] * len(batch)

        if not valid:
            results.extend(output_batch)
            continue

        valid_indices, valid_waveforms = zip(*valid)

        inputs = extractor(
            list(valid_waveforms),
            sampling_rate=cfg.audio_emotion.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits  # (batch, num_labels)

        for local_idx, logit_row in zip(valid_indices, logits):
            output_batch[local_idx] = _logits_to_probs(logit_row, id2label)

        results.extend(output_batch)

    return results
