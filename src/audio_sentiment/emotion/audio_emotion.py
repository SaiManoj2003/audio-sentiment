"""
Audio-based emotion classification using wav2vec2.

Model: ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition
    - Fine-tuned on RAVDESS (8 emotion classes)
    - Saved with a custom training script whose head layout is:
        classifier.dense   Linear(1024, 1024)   # not compatible with any stock HF class
        classifier.output  Linear(1024, 8)
    - No stock HuggingFace class matches this layout:
        ForAudioClassification  expects classifier.dense + classifier.out_proj  (different key)
        ForSequenceClassification expects projector(1024→256) + classifier(256→8)  (wrong dims)
    - Solution: load Wav2Vec2Model backbone cleanly, build the head manually,
      inject weights from the raw checkpoint.
"""

import logging
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoFeatureExtractor, Wav2Vec2Model

from audio_sentiment.config import cfg

logger = logging.getLogger(__name__)

EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

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

_NEUTRAL_PROBS = {e: (1.0 if e == "neutral" else 0.0) for e in EMOTIONS}


class _SERModel(nn.Module):
    """
    Minimal reimplementation matching the ehcalabres checkpoint exactly:
      Wav2Vec2Model → masked mean pool → Linear(1024,1024)+tanh → Linear(1024,8)

    This is necessary because no stock HuggingFace class has this exact head
    layout (classifier.dense / classifier.output at 1024→1024→8 dimensions).
    """

    def __init__(self, backbone: Wav2Vec2Model, dense: nn.Linear, output: nn.Linear, id2label: dict):
        super().__init__()
        self.wav2vec2 = backbone
        self.dense = dense
        self.output = output
        # Expose config.id2label so callers can use the same interface as stock models
        self.config = SimpleNamespace(id2label=id2label)

    def _mean_pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        """Mean-pool over time, respecting padding when attention_mask is provided."""
        if attention_mask is None:
            return hidden.mean(dim=1)
        mask = attention_mask.unsqueeze(-1).float()
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor | None = None):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        pooled = self._mean_pool(outputs.last_hidden_state, attention_mask)
        hidden = torch.tanh(self.dense(pooled))
        logits = self.output(hidden)
        return SimpleNamespace(logits=logits)


@lru_cache(maxsize=1)
def _get_model_and_extractor():
    model_name = cfg.audio_emotion.model_name
    logger.info("Loading audio emotion model: %s", model_name)

    extractor = AutoFeatureExtractor.from_pretrained(model_name)

    # ── Backbone ──────────────────────────────────────────────────────────
    # Wav2Vec2Model loads only wav2vec2.* keys — classifier.* are extra and ignored
    backbone = Wav2Vec2Model.from_pretrained(model_name)

    # ── Head: load from raw checkpoint and inject manually ────────────────
    ckpt_path = hf_hub_download(repo_id=model_name, filename="pytorch_model.bin")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    dense_w = ckpt["classifier.dense.weight"]    # [1024, 1024]
    dense_b = ckpt["classifier.dense.bias"]      # [1024]
    out_w   = ckpt["classifier.output.weight"]   # [8, 1024]
    out_b   = ckpt["classifier.output.bias"]     # [8]

    dense = nn.Linear(dense_w.shape[1], dense_w.shape[0])
    dense.weight.data.copy_(dense_w)
    dense.bias.data.copy_(dense_b)

    output = nn.Linear(out_w.shape[1], out_w.shape[0])
    output.weight.data.copy_(out_w)
    output.bias.data.copy_(out_b)

    id2label = AutoConfig.from_pretrained(model_name).id2label
    logger.info("Model labels: %s", {i: id2label[i] for i in sorted(id2label)})

    model = _SERModel(backbone, dense, output, id2label)

    device = torch.device(
        "cuda" if cfg.audio_emotion.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    model = model.to(device)
    model.eval()

    logger.info("Audio emotion model loaded on %s — no LOAD REPORT warnings expected.", device)
    return model, extractor, device


def _logits_to_probs(logits: torch.Tensor, id2label: dict) -> dict[str, float]:
    """Convert a single logits row to a canonical probability dict."""
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    canonical: dict[str, float] = {e: 0.0 for e in EMOTIONS}
    for i, prob in enumerate(probs):
        raw_label = id2label[i].lower()
        canonical_label = _LABEL_MAP.get(raw_label)
        if canonical_label is None:
            logger.warning("Label '%s' not in _LABEL_MAP — skipping.", raw_label)
            continue
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

    Args:
        waveforms:  List of mono float32 waveforms at 16kHz.
        batch_size: Waveforms per forward pass.

    Returns:
        List of emotion probability dicts, one per input waveform.
    """
    if not waveforms:
        return []

    model, extractor, device = _get_model_and_extractor()
    id2label = model.config.id2label
    results: list[dict[str, float]] = []

    for batch_start in range(0, len(waveforms), batch_size):
        batch = waveforms[batch_start: batch_start + batch_size]
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
            logits = model(**inputs).logits

        for local_idx, logit_row in zip(valid_indices, logits):
            output_batch[local_idx] = _logits_to_probs(logit_row, id2label)

        results.extend(output_batch)

    return results
