"""
Tests for the fusion layer.

No models loaded — all emotion probs are synthetic dicts.
Tests verify the math and business logic of fusion itself.
"""

import pytest
import numpy as np

from audio_sentiment.fusion.fusion import (
    fuse_emotions,
    compute_valence,
    escalation_flag,
    summarise_speaker,
)
from audio_sentiment.emotion.text_emotion import EMOTIONS

def uniform_probs() -> dict[str, float]:
    """Equal probability across all emotions."""
    p = 1.0 / len(EMOTIONS)
    return {e: p for e in EMOTIONS}


def pure_emotion(emotion: str) -> dict[str, float]:
    """100% probability on one emotion."""
    return {e: (1.0 if e == emotion else 0.0) for e in EMOTIONS}


class TestFuseEmotions:

    def test_output_sums_to_one(self):
        fused = fuse_emotions(uniform_probs(), uniform_probs())
        assert sum(fused.values()) == pytest.approx(1.0, abs=1e-6)

    def test_equal_weights_is_average(self):
        text = pure_emotion("joy")
        audio = pure_emotion("anger")
        fused = fuse_emotions(text, audio, text_weight=0.5, audio_weight=0.5)
        assert fused["joy"] == pytest.approx(0.5, abs=1e-6)
        assert fused["anger"] == pytest.approx(0.5, abs=1e-6)

    def test_full_text_weight_ignores_audio(self):
        text = pure_emotion("joy")
        audio = pure_emotion("anger")
        fused = fuse_emotions(text, audio, text_weight=1.0, audio_weight=0.0)
        assert fused["joy"] == pytest.approx(1.0, abs=1e-5)
        assert fused["anger"] == pytest.approx(0.0, abs=1e-5)

    def test_invalid_weights_raise(self):
        with pytest.raises(ValueError, match="must equal 1.0"):
            fuse_emotions(uniform_probs(), uniform_probs(),
                         text_weight=0.6, audio_weight=0.6)

    def test_all_emotions_present_in_output(self):
        fused = fuse_emotions(uniform_probs(), uniform_probs())
        assert set(fused.keys()) == set(EMOTIONS)

class TestComputeValence:

    def test_pure_joy_positive(self):
        assert compute_valence(pure_emotion("joy")) > 0

    def test_pure_anger_negative(self):
        assert compute_valence(pure_emotion("anger")) < 0

    def test_output_in_range(self):
        for _ in range(50):
            probs = uniform_probs()
            score = compute_valence(probs)
            assert -1.0 <= score <= 1.0

class TestEscalationFlag:

    def test_sustained_negative_triggers(self):
        scores = [-0.7, -0.8, -0.6, -0.9]
        assert escalation_flag(scores, threshold=-0.5, window=3) is True

    def test_recovering_sentiment_no_trigger(self):
        scores = [-0.8, -0.7, 0.1, 0.3]
        assert escalation_flag(scores, threshold=-0.5, window=3) is False

    def test_insufficient_history_no_trigger(self):
        scores = [-0.9, -0.8]
        assert escalation_flag(scores, threshold=-0.5, window=3) is False

    def test_mixed_recent_no_trigger(self):
        scores = [-0.9, -0.8, -0.9, 0.5]
        assert escalation_flag(scores, threshold=-0.5, window=3) is False


class TestSummariseSpeaker:

    def test_positive_call_summary(self):
        scores = [0.6, 0.7, 0.8]
        probs = [pure_emotion("joy")] * 3
        summary = summarise_speaker(scores, probs, speaker_id=0)
        assert summary["sentiment_label"] == "Positive"
        assert summary["dominant_emotion"] == "joy"
        assert summary["escalation_detected"] is False

    def test_negative_call_flags_escalation(self):
        scores = [-0.7, -0.8, -0.9]
        probs = [pure_emotion("anger")] * 3
        summary = summarise_speaker(scores, probs, speaker_id=1)
        assert summary["sentiment_label"] == "Negative"
        assert summary["escalation_detected"] is True

    def test_empty_speaker_returns_defaults(self):
        summary = summarise_speaker([], [], speaker_id=0)
        assert summary["sentence_count"] == 0
        assert summary["sentiment_label"] == "Neutral"
