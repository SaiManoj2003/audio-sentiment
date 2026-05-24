# Audio Call Sentiment Analysis

A multimodal pipeline for analysing sentiment in customer support call recordings.

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