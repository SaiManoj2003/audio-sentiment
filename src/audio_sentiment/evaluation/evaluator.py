"""
Evaluation script — generates benchmark metrics on RAVDESS.

Runs three approaches on the same samples:
    1. Text-only   — classify_text_emotion on Whisper transcript
    2. Audio-only  — classify_audio_emotion on raw waveform
    3. Fusion      — fuse_emotions combining both branches

Reports accuracy and weighted F1 for each approach.

Usage:
    python -m audio_sentiment.evaluation.evaluator
    python -m audio_sentiment.evaluation.evaluator --max-samples 100
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)

from audio_sentiment.asr.audio_loader import load_audio
from audio_sentiment.asr.transcriber import Transcriber
from audio_sentiment.config import cfg
from audio_sentiment.emotion.audio_emotion import classify_audio_emotion
from audio_sentiment.emotion.text_emotion import (
    classify_text_emotion,
    dominant_emotion,
)
from audio_sentiment.evaluation.ravdess_loader import (
    RAVDESSample,
    dataset_summary,
    load_ravdess,
)
from audio_sentiment.fusion.fusion import fuse_emotions

logger = logging.getLogger(__name__)

EVAL_EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]


def evaluate_sample(
    sample: RAVDESSample,
    transcriber: Transcriber,
) -> dict | None:
    """
    Run all three approaches on a single RAVDESS sample.

    Returns:
        Dict with true label and predictions from each approach,
        or None if processing failed.
    """
    try:
        waveform, sr = load_audio(sample.file_path)

        sentences = transcriber.transcribe(sample.file_path, waveform=waveform, sr=sr)
        full_text = " ".join(s.text for s in sentences).strip()

        if not full_text:
            logger.warning("No transcription for %s — skipping.", sample.file_path.name)
            return None

        text_probs = classify_text_emotion(full_text)
        text_pred = dominant_emotion(text_probs)

        audio_probs = classify_audio_emotion(waveform)
        audio_pred = dominant_emotion(audio_probs)

        fused_probs = fuse_emotions(text_probs, audio_probs)
        fusion_pred = dominant_emotion(fused_probs)

        return {
            "file": sample.file_path.name,
            "true_label": sample.emotion,
            "text_pred": text_pred,
            "audio_pred": audio_pred,
            "fusion_pred": fusion_pred,
            "text_probs": text_probs,
            "audio_probs": audio_probs,
            "fused_probs": fused_probs,
        }

    except Exception as exc:
        logger.error("Failed on %s: %s", sample.file_path.name, exc)
        return None


def compute_metrics(
    true_labels: list[str],
    predicted_labels: list[str],
    approach_name: str,
) -> dict:
    """Compute and log accuracy + weighted F1 for one approach."""
    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average="weighted", zero_division=0)

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=EVAL_EMOTIONS,
        zero_division=0,
    )

    logger.info(
        "\n%s Results:\n  Accuracy:    %.4f (%.1f%%)\n  Weighted F1: %.4f\n\n%s",
        approach_name.upper(),
        accuracy, accuracy * 100,
        f1,
        report,
    )

    return {
        "approach": approach_name,
        "accuracy": round(accuracy, 4),
        "weighted_f1": round(f1, 4),
        "classification_report": report,
    }


def run_evaluation(
    max_samples: int | None = None,
    results_dir: Path | None = None,
) -> dict:
    """
    Run full evaluation on RAVDESS dataset.

    Args:
        max_samples: Cap number of samples — useful for quick testing.
        results_dir: Where to save JSON results.

    Returns:
        Dict containing metrics for all three approaches.
    """
    results_dir = Path(results_dir or cfg.eval.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading RAVDESS dataset...")
    samples = load_ravdess()

    summary = dataset_summary(samples)
    logger.info("Dataset summary: %s", json.dumps(summary, indent=2))

    if max_samples:
        rng = np.random.default_rng(cfg.eval.random_seed)
        indices = rng.choice(len(samples), size=min(max_samples, len(samples)), replace=False)
        samples = [samples[i] for i in sorted(indices)]
        logger.info("Evaluating on %d samples (max_samples=%d)", len(samples), max_samples)

    logger.info("Loading models...")
    transcriber = Transcriber()

    # Warm-load both models into cache before the evaluation loop
    from audio_sentiment.emotion.text_emotion import _get_pipeline
    from audio_sentiment.emotion.audio_emotion import _get_model_and_extractor
    _get_pipeline()
    _get_model_and_extractor()

    all_results = []
    start_time = time.perf_counter()

    for i, sample in enumerate(samples, start=1):
        if i % 10 == 0 or i == 1:
            elapsed = time.perf_counter() - start_time
            eta = (elapsed / i) * (len(samples) - i)
            logger.info(
                "Progress: %d/%d | Elapsed: %.0fs | ETA: %.0fs",
                i, len(samples), elapsed, eta,
            )

        result = evaluate_sample(sample, transcriber)
        if result:
            all_results.append(result)

    logger.info(
        "Processed %d/%d samples successfully.",
        len(all_results), len(samples),
    )

    true_labels = [r["true_label"] for r in all_results]
    text_preds = [r["text_pred"] for r in all_results]
    audio_preds = [r["audio_pred"] for r in all_results]
    fusion_preds = [r["fusion_pred"] for r in all_results]

    text_metrics = compute_metrics(true_labels, text_preds, "text_only")
    audio_metrics = compute_metrics(true_labels, audio_preds, "audio_only")
    fusion_metrics = compute_metrics(true_labels, fusion_preds, "fusion")

    output = {
        "dataset": "RAVDESS",
        "total_samples_evaluated": len(all_results),
        "results": {
            "text_only": text_metrics,
            "audio_only": audio_metrics,
            "fusion": fusion_metrics,
        },
        "improvement_over_text": round(
            fusion_metrics["accuracy"] - text_metrics["accuracy"], 4
        ),
        "improvement_over_audio": round(
            fusion_metrics["accuracy"] - audio_metrics["accuracy"], 4
        ),
    }

    out_path = results_dir / "ravdess_evaluation.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("Results saved to %s", out_path)

    print("\n" + "=" * 55)
    print("  RAVDESS EVALUATION SUMMARY")
    print("=" * 55)
    print(f"  {'Approach':<20} {'Accuracy':>10} {'Weighted F1':>12}")
    print("-" * 55)
    for metrics in [text_metrics, audio_metrics, fusion_metrics]:
        print(
            f"  {metrics['approach']:<20} "
            f"{metrics['accuracy']:>9.1%} "
            f"{metrics['weighted_f1']:>11.4f}"
        )
    print("=" * 55)
    print(f"  Fusion vs text-only:  {output['improvement_over_text']:+.1%}")
    print(f"  Fusion vs audio-only: {output['improvement_over_audio']:+.1%}")
    print("=" * 55 + "\n")

    return output


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Evaluate on RAVDESS dataset")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit evaluation to N samples (default: all 1440)",
    )
    args = parser.parse_args()

    run_evaluation(max_samples=args.max_samples)
