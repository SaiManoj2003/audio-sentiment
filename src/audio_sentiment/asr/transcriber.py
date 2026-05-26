"""
Whisper-based ASR with speaker diarization.

Uses faster-whisper (CTranslate2 backend) which is 2-4x faster than the
original OpenAI Whisper implementation at identical accuracy.

Speaker diarization note:
faster-whisper does not natively support diarization — that requires
pyannote.audio which needs a Hugging Face token and model agreement.
For this project we use a simpler approach: we detect speaker turns
via silence gaps between segments.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

from audio_sentiment.config import cfg
from audio_sentiment.asr.audio_loader import load_audio, slice_waveform

logger = logging.getLogger(__name__)


@dataclass
class Sentence:
    """
    A single transcribed sentence with metadata.

    Attributes:
        speaker_id:  0 = first speaker detected, 1 = second speaker, etc.
        text:        Transcribed text of the sentence.
        start:       Start time in seconds from audio beginning.
        end:         End time in seconds from audio beginning.
        waveform:    Raw audio segment (float32 array at 16kHz).
                     Used downstream for acoustic feature extraction.
    """
    speaker_id: int
    text: str
    start: float
    end: float
    waveform: np.ndarray

    @property
    def duration(self) -> float:
        return self.end - self.start

    def __repr__(self) -> str:
        return (
            f"Sentence(speaker={self.speaker_id}, "
            f"t={self.start:.1f}-{self.end:.1f}s, "
            f"text='{self.text[:50]}...')"
        )


class Transcriber:
    """
    Loads a Whisper model and transcribes audio files into Sentence objects.

    The model is loaded once on instantiation and reused across calls —
    loading it per-call would add ~3-5s cold start per request.
    """

    def __init__(self):
        logger.info(
            "Loading Whisper model: %s on %s (%s)",
            cfg.asr.model_size,
            cfg.asr.device,
            cfg.asr.compute_type,
        )
        self._model = WhisperModel(
            cfg.asr.model_size,
            device=cfg.asr.device,
            compute_type=cfg.asr.compute_type,
        )
        logger.info("Whisper model loaded.")

    def transcribe(self, file_path: str | Path) -> list[Sentence]:
        """
        Transcribe an audio file and return a list of Sentence objects.

        Args:
            file_path: Path to a supported audio file.

        Returns:
            List of Sentence objects ordered by start time.
        """
        path = Path(file_path)
        waveform, sr = load_audio(path)

        logger.info("Transcribing: %s", path.name)

        segments, info = self._model.transcribe(
            str(path),
            language=cfg.asr.language,
            beam_size=cfg.asr.beam_size,
            vad_filter=True,        # skip silent regions
            word_timestamps=True,   # needed for accurate segment boundaries
        )

        logger.info(
            "Detected language: %s (%.0f%% confidence)",
            info.language,
            info.language_probability * 100,
        )

        sentences = []
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue

            audio_segment = slice_waveform(
                waveform,
                start_sec=segment.start,
                end_sec=segment.end,
                sample_rate=sr,
            )

            # Speaker assignment via turn detection — see module docstring
            speaker_id = self._assign_speaker(segment.start, sentences)

            sentences.append(Sentence(
                speaker_id=speaker_id,
                text=text,
                start=segment.start,
                end=segment.end,
                waveform=audio_segment,
            ))

        logger.info("Transcribed %d sentences.", len(sentences))
        return sentences

    def _assign_speaker(
        self,
        start_time: float,
        previous_sentences: list[Sentence],
        gap_threshold: float = 1.0,
    ) -> int:
        """
        Assign a speaker ID based on silence gaps between segments.

        Logic: if the gap between the last sentence and this one is larger
        than gap_threshold seconds, we assume the speaker changed.
        Alternates between speaker 0 and speaker 1.

        This is a heuristic that works well for clean 2-speaker call
        recordings. For production use, replace with pyannote.audio.

        Args:
            start_time:         Start time of the current segment.
            previous_sentences: All previously processed sentences.
            gap_threshold:      Silence gap (seconds) that signals a turn.

        Returns:
            Speaker ID (0 or 1).
        """
        if not previous_sentences:
            return 0

        last = previous_sentences[-1]
        gap = start_time - last.end

        if gap >= gap_threshold:
            # Speaker changed — flip between 0 and 1
            return 1 - last.speaker_id

        # Same speaker continuing
        return last.speaker_id
