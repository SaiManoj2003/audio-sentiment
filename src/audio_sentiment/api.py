"""
FastAPI application for audio sentiment analysis.

Endpoints:
    POST /analyse   — upload an audio file, get full sentiment analysis
    GET  /health    — liveness check
"""

import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from audio_sentiment.asr.audio_loader import SUPPORTED_FORMATS
from audio_sentiment.config import cfg
from audio_sentiment.pipeline import CallResult, SentenceResult, SentimentPipeline

logger = logging.getLogger(__name__)

# Single pipeline instance — models loaded once on startup, reused per request
_pipeline: SentimentPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    logger.info("Loading sentiment pipeline on startup...")
    _pipeline = SentimentPipeline()
    logger.info("Pipeline ready.")
    yield
    _pipeline = None


app = FastAPI(
    title="Audio Sentiment Analysis API",
    description="Multimodal sentiment analysis for call recordings.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Response schemas ──────────────────────────────────────────────────────────

class SentenceOut(BaseModel):
    speaker_id: int
    text: str
    start: float
    end: float
    dominant_emotion: str
    low_confidence: bool
    valence_score: float
    fused_emotion_probs: dict[str, float]


class SpeakerSummaryOut(BaseModel):
    speaker_id: int
    sentence_count: int
    average_valence: float
    dominant_emotion: str
    sentiment_label: str
    escalation_detected: bool


class AnalysisResponse(BaseModel):
    filename: str
    duration_seconds: float
    total_sentences: int
    sentences: list[SentenceOut]
    speaker_summaries: list[SpeakerSummaryOut]


def _sentence_to_out(s: SentenceResult) -> SentenceOut:
    return SentenceOut(
        speaker_id=s.speaker_id,
        text=s.text,
        start=round(s.start, 3),
        end=round(s.end, 3),
        dominant_emotion=s.dominant_emotion,
        low_confidence=s.low_confidence,
        valence_score=round(s.valence_score, 4),
        fused_emotion_probs={k: round(v, 4) for k, v in s.fused_emotion_probs.items()},
    )


def _result_to_response(result: CallResult) -> AnalysisResponse:
    return AnalysisResponse(
        filename=result.filename,
        duration_seconds=result.duration_seconds,
        total_sentences=result.total_sentences,
        sentences=[_sentence_to_out(s) for s in result.sentences],
        speaker_summaries=[SpeakerSummaryOut(**s) for s in result.speaker_summaries],
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Liveness check."""
    return {"status": "ok", "pipeline_loaded": _pipeline is not None}


@app.post("/analyse", response_model=AnalysisResponse)
async def analyse(file: UploadFile = File(...)):
    """
    Analyse sentiment in an uploaded audio file.

    Accepts: .wav, .mp3, .m4a, .flac, .ogg, .opus, .webm

    Returns per-sentence emotion labels and per-speaker sentiment summaries.
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised.")

    suffix = Path(file.filename or "upload").suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported format '{suffix}'. Supported: {sorted(SUPPORTED_FORMATS)}",
        )

    content = await file.read()
    max_bytes = cfg.api.max_file_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {cfg.api.max_file_size_mb}MB limit.",
        )

    # Write to a temp file — audio libraries need a real path for decoding
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        logger.info("Analysing uploaded file: %s (%d bytes)", file.filename, len(content))
        result = _pipeline.analyse(tmp_path)
    except Exception as exc:
        logger.error("Analysis failed for %s: %s", file.filename, exc)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")
    finally:
        tmp_path.unlink(missing_ok=True)

    return _result_to_response(result)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )
    uvicorn.run(
        "audio_sentiment.api:app",
        host=cfg.api.host,
        port=cfg.api.port,
        reload=False,
    )
