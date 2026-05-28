"""
Pipeline orchestrator.

This module is the single public entry point for the full analysis.
External code (API, evaluation script, notebook) only needs to call
analyse_call() — it never imports individual modules directly.

This matters for maintainability: if we swap wav2vec2 for a different
audio model tomorrow, only audio_emotion.py changes. The pipeline
interface stays identical.

Pipeline flow per sentence:
    1. Whisper  → text + timestamps + speaker_id + waveform slice
    2. Text branch  → emotion probs from words
    3. Audio branch → emotion probs from waveform
    4. Fusion       → combined emotion probs
    5. Valence      → single scalar for timeline
    6. Summarise    → per-speaker aggregated results
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from audio_sentiment.asr.transcriber import Sentence, Transcriber
from audio_sentiment.emotion.acoustic_features import extract_acoustic_features
from audio_sentiment.emotion.audio_emotion import classify_audio_emotion
from audio_sentiment.emotion.text_emotion import classify_text_emotion
from audio_sentiment.fusion.fusion import (
    compute_valence,
    fuse_emotions,
    summarise_speaker,
)
from audio_sentiment.config import cfg

logger = logging.getLogger(__name__)


@dataclass
class SentenceResult:
    """Full analysis result for a single sentence."""
    speaker_id: int
    text: str
    start: float
    end: float
    text_emotion_probs: dict[str, float]
    audio_emotion_probs: dict[str, float]
    fused_emotion_probs: dict[str, float]
    valence_score: float
    dominant_emotion: str


@dataclass
class CallResult:
    """Full analysis result for an entire call recording."""
    filename: str
    duration_seconds: float
    total_sentences: int
    sentences: list[SentenceResult]
    speaker_summaries: list[dict]

    def sentences_by_speaker(self, speaker_id: int) -> list[SentenceResult]:
        return [s for s in self.sentences if s.speaker_id == speaker_id]

    def valence_timeline(self, speaker_id: int) -> list[tuple[float, float]]:
        """Return (start_time, valence_score) pairs for a speaker."""
        return [
            (s.start, s.valence_score)
            for s in self.sentences_by_speaker(speaker_id)
        ]


class SentimentPipeline:
    """
    Full multimodal sentiment analysis pipeline.

    Instantiate once, call analyse() for each audio file.
    Models are loaded on first instantiation and reused — do not
    create a new SentimentPipeline per file.
    """

    def __init__(self):
        logger.info("Initialising sentiment pipeline...")
        self._transcriber = Transcriber()
        logger.info("Pipeline ready.")

    def analyse(self, file_path: str | Path) -> CallResult:
        """
        Run the full pipeline on an audio file.

        Args:
            file_path: Path to a supported audio file.

        Returns:
            CallResult containing per-sentence and per-speaker results.
        """
        path = Path(file_path)
        logger.info("Analysing: %s", path.name)

        # ── Step 1: Transcription ─────────────────────────────────────────
        sentences: list[Sentence] = self._transcriber.transcribe(path)

        if not sentences:
            logger.warning("No sentences transcribed from %s", path.name)
            return CallResult(
                filename=path.name,
                duration_seconds=0.0,
                total_sentences=0,
                sentences=[],
                speaker_summaries=[],
            )

        # ── Steps 2-5: Per-sentence analysis ─────────────────────────────
        results: list[SentenceResult] = []
        for i, sentence in enumerate(sentences):
            logger.debug(
                "Processing sentence %d/%d: '%s'",
                i + 1, len(sentences), sentence.text[:40],
            )

            # Text branch
            text_probs = classify_text_emotion(sentence.text)

            # Audio branch
            audio_probs = classify_audio_emotion(sentence.waveform)

            # Fusion
            fused_probs = fuse_emotions(text_probs, audio_probs)

            # Valence scalar
            valence = compute_valence(fused_probs)

            # Dominant emotion from fused probs
            dom_emotion = max(fused_probs, key=fused_probs.get)

            results.append(SentenceResult(
                speaker_id=sentence.speaker_id,
                text=sentence.text,
                start=sentence.start,
                end=sentence.end,
                text_emotion_probs=text_probs,
                audio_emotion_probs=audio_probs,
                fused_emotion_probs=fused_probs,
                valence_score=valence,
                dominant_emotion=dom_emotion,
            ))

        # ── Step 6: Per-speaker summaries ────────────────────────────────
        speaker_ids = sorted({r.speaker_id for r in results})
        summaries = []
        for speaker_id in speaker_ids:
            speaker_results = [r for r in results if r.speaker_id == speaker_id]
            summaries.append(summarise_speaker(
                valence_scores=[r.valence_score for r in speaker_results],
                emotion_probs_list=[r.fused_emotion_probs for r in speaker_results],
                speaker_id=speaker_id,
            ))

        duration = sentences[-1].end - sentences[0].start

        logger.info(
            "Analysis complete: %d sentences, %d speakers, %.1fs duration",
            len(results), len(summaries), duration,
        )

        return CallResult(
            filename=path.name,
            duration_seconds=round(duration, 2),
            total_sentences=len(results),
            sentences=results,
            speaker_summaries=summaries,
        )
