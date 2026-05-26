"""
Tests for the text emotion classification module.

The HuggingFace pipeline is mocked throughout — these tests verify
the logic (valence calculation, dominant emotion, edge cases) without
downloading any model weights.
"""

from unittest.mock import MagicMock, patch

import pytest

from audio_sentiment.emotion.text_emotion import (
    EMOTIONS,
    EMOTION_VALENCE,
    classify_text_emotion,
    dominant_emotion,
    emotion_to_valence,
)

def _make_pipeline_output(dominant: str, dominant_score: float = 0.80):
    """Build a realistic pipeline output with one dominant emotion."""
    remaining = 1.0 - dominant_score
    per_other = remaining / (len(EMOTIONS) - 1)
    return [[
        {"label": e, "score": dominant_score if e == dominant else per_other}
        for e in EMOTIONS
    ]]

class TestEmotionToValence:

    def test_pure_joy_is_positive(self):
        probs = {e: (1.0 if e == "joy" else 0.0) for e in EMOTIONS}
        assert emotion_to_valence(probs) > 0

    def test_pure_anger_is_negative(self):
        probs = {e: (1.0 if e == "anger" else 0.0) for e in EMOTIONS}
        assert emotion_to_valence(probs) < 0

    def test_pure_neutral_is_zero(self):
        probs = {e: (1.0 if e == "neutral" else 0.0) for e in EMOTIONS}
        assert emotion_to_valence(probs) == pytest.approx(0.0)

    def test_output_clamped_to_range(self):
        """Score must always lie within [-1, 1]."""
        probs = {e: 1.0 / len(EMOTIONS) for e in EMOTIONS}
        score = emotion_to_valence(probs)
        assert -1.0 <= score <= 1.0

    def test_all_valences_defined(self):
        """Every emotion must have a valence weight — catches typos."""
        for emotion in EMOTIONS:
            assert emotion in EMOTION_VALENCE, f"Missing valence for: {emotion}"

class TestDominantEmotion:

    def test_returns_highest_prob_emotion(self):
        probs = {e: 0.1 for e in EMOTIONS}
        probs["joy"] = 0.8
        assert dominant_emotion(probs) == "joy"

    def test_works_with_uniform_probs(self):
        probs = {e: 1.0 / len(EMOTIONS) for e in EMOTIONS}
        result = dominant_emotion(probs)
        assert result in EMOTIONS

class TestClassifyTextEmotion:

    def test_returns_all_emotion_keys(self):
        mock_pipe = MagicMock(return_value=_make_pipeline_output("anger"))
        with patch("audio_sentiment.emotion.text_emotion._get_pipeline", return_value=mock_pipe):
            result = classify_text_emotion("This is terrible!")
        assert set(result.keys()) == set(EMOTIONS)

    def test_probabilities_sum_to_one(self):
        mock_pipe = MagicMock(return_value=_make_pipeline_output("joy"))
        with patch("audio_sentiment.emotion.text_emotion._get_pipeline", return_value=mock_pipe):
            result = classify_text_emotion("This is great!")
        assert sum(result.values()) == pytest.approx(1.0, abs=1e-5)

    def test_empty_string_returns_neutral(self):
        result = classify_text_emotion("")
        assert result["neutral"] == 1.0
        assert dominant_emotion(result) == "neutral"

    def test_whitespace_only_returns_neutral(self):
        result = classify_text_emotion("   ")
        assert result["neutral"] == 1.0
