"""
Smoke tests for the ASR layer.

These tests do NOT require a real audio file or a loaded Whisper model.
They verify the logic of our utility functions in isolation — fast,
offline, no GPU needed.
"""

import numpy as np
import pytest

from audio_sentiment.asr.audio_loader import slice_waveform
from audio_sentiment.asr.transcriber import Sentence


class TestSliceWaveform:

    def test_basic_slice(self):
        """Slicing at known boundaries returns correct length."""
        sr = 16000
        waveform = np.zeros(sr * 10, dtype=np.float32)  # 10 seconds
        segment = slice_waveform(waveform, start_sec=2.0, end_sec=5.0, sample_rate=sr)
        assert len(segment) == sr * 3  # 3 seconds worth of samples

    def test_clamps_to_waveform_length(self):
        """End time beyond audio length is clamped, not an error."""
        sr = 16000
        waveform = np.zeros(sr * 5, dtype=np.float32)  # 5 seconds
        segment = slice_waveform(waveform, start_sec=4.0, end_sec=9.0, sample_rate=sr)
        assert len(segment) == sr * 1  # only 1 second available

    def test_returns_float32(self):
        sr = 16000
        waveform = np.ones(sr * 2, dtype=np.float64)  # intentionally float64
        segment = slice_waveform(waveform, 0.0, 1.0, sr)
        assert segment.dtype == np.float32


class TestSentenceDataclass:

    def _make_sentence(self, start=0.0, end=3.5, speaker_id=0):
        return Sentence(
            speaker_id=speaker_id,
            text="Hello, how can I help you today?",
            start=start,
            end=end,
            waveform=np.zeros(int((end - start) * 16000), dtype=np.float32),
        )

    def test_duration_property(self):
        s = self._make_sentence(start=1.0, end=4.5)
        assert s.duration == pytest.approx(3.5)

    def test_repr_truncates_long_text(self):
        s = self._make_sentence()
        assert "Hello" in repr(s)
        assert "speaker=0" in repr(s)
