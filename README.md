# Audio Call Sentiment Analysis

A multimodal pipeline for analysing sentiment in customer support call recordings.

## Architecture

Audio File
├──► Text Branch
│    └──► Whisper (faster-whisper)
│         Transcribes text & segments speaker diarization.
│         Output: Sentences with speaker IDs + timestamps.
│         │
│         ▼
│    └──► j-hartmann/emotion-english-distilroberta-base
│         Analyzes text emotion classification.
│         Output: Emotion probabilities from words.
│         │
│         ▼
├──► Audio Branch
│    └──► librosa
│         Extracts acoustic features (MFCC, pitch, energy, ZCR).
│         │
│         ▼
│    └──► wav2vec2-lg-xlsr-en-speech-emotion-recognition
│         Analyzes audio emotion classification.
│         Output: Emotion probabilities from tone.
│         │
│         ▼
└──► Late Fusion Layer
     Calculates weighted combination of both branches.
     Output 1: Final emotion label per sentence.
     Output 2: Per-speaker sentiment timeline.



## Why Multimodal?

Text-only sentiment misses sarcasm, tone, and paralinguistic cues.
Audio-only models miss semantic content. Fusing both achieves accuracy

## Setup

```bash
git clone <repo>
cd audio-sentiment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```