"""
RAVDESS dataset loader.

Parses emotion labels directly from filenames — no separate annotation
file needed. See filename convention:
    03-01-{emotion}-{intensity}-{statement}-{repetition}-{actor}.wav

We filter to speech-only files (modality=03) and exclude 'calm' as a
separate class — mapping it to neutral instead, giving us 7 classes
consistent with our text emotion model.

Reference: Livingstone & Russo (2018). The Ryerson Audio-Visual Database
of Emotional Speech and Song (RAVDESS).
https://doi.org/10.1371/journal.pone.0196391
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from audio_sentiment.config import cfg

logger = logging.getLogger(__name__)

# RAVDESS emotion code → canonical label
RAVDESS_EMOTION_MAP = {
    "01": "neutral",
    "02": "neutral",    # calm → neutral (consistent with our 7-class schema)
    "03": "joy",
    "04": "sadness",
    "05": "anger",
    "06": "fear",
    "07": "disgust",
    "08": "surprise",
}


@dataclass
class RAVDESSample:
    """One labelled audio sample from RAVDESS."""
    file_path: Path
    emotion: str          # canonical label e.g. "anger"
    intensity: str        # "normal" or "strong"
    actor_id: int
    gender: str           # "male" (odd actors) or "female" (even actors)

    def __repr__(self) -> str:
        return (
            f"RAVDESSample(emotion={self.emotion}, "
            f"intensity={self.intensity}, "
            f"actor={self.actor_id}, "
            f"gender={self.gender})"
        )


def load_ravdess(
    data_dir: Path | None = None,
    emotions: list[str] | None = None,
    intensity: str | None = None,
) -> list[RAVDESSample]:
    """
    Load and parse all RAVDESS speech samples from a directory.

    Args:
        data_dir:  Path to ravdess folder containing Actor_XX subdirs.
                   Defaults to cfg.eval.data_dir.
        emotions:  Optional list of canonical emotion labels to filter to.
                   e.g. ["anger", "joy", "sadness"]
        intensity: Optional filter — "normal" or "strong".

    Returns:
        List of RAVDESSample objects sorted by file path.

    Raises:
        FileNotFoundError: If data_dir does not exist.
        ValueError: If no samples found after filtering.
    """
    data_dir = Path(data_dir) if data_dir else cfg.eval.data_dir

    if not data_dir.exists():
        raise FileNotFoundError(
            f"RAVDESS data directory not found: {data_dir}\n"
            f"Download with: wget https://zenodo.org/record/1188976/files/"
            f"Audio_Speech_Actors_01-24.zip"
        )

    wav_files = sorted(data_dir.rglob("*.wav"))
    if not wav_files:
        raise ValueError(f"No .wav files found in {data_dir}")

    samples = []
    skipped = 0

    for path in wav_files:
        parts = path.stem.split("-")

        # Guard against unexpected filenames
        if len(parts) != 7:
            logger.warning("Skipping unexpected filename format: %s", path.name)
            skipped += 1
            continue

        modality = parts[0]

        # Skip song files (modality=02) — we only want speech (modality=03)
        if modality != "03":
            skipped += 1
            continue

        emotion_code = parts[2]
        intensity_code = parts[3]
        actor_id = int(parts[6])

        emotion = RAVDESS_EMOTION_MAP.get(emotion_code)
        if emotion is None:
            logger.warning("Unknown emotion code %s in %s", emotion_code, path.name)
            skipped += 1
            continue

        intensity_label = "normal" if intensity_code == "01" else "strong"
        gender = "female" if actor_id % 2 == 0 else "male"

        sample = RAVDESSample(
            file_path=path,
            emotion=emotion,
            intensity=intensity_label,
            actor_id=actor_id,
            gender=gender,
        )

        # Apply optional filters
        if emotions and emotion not in emotions:
            continue
        if intensity and intensity_label != intensity:
            continue

        samples.append(sample)

    logger.info(
        "Loaded %d RAVDESS samples (%d skipped) from %s",
        len(samples), skipped, data_dir,
    )

    if not samples:
        raise ValueError(
            f"No samples found after filtering. "
            f"emotions={emotions}, intensity={intensity}"
        )

    return samples


def dataset_summary(samples: list[RAVDESSample]) -> dict:
    """
    Print a breakdown of the loaded dataset.
    Useful for verifying class balance before evaluation.
    """
    from collections import Counter
    emotion_counts = Counter(s.emotion for s in samples)
    gender_counts = Counter(s.gender for s in samples)

    return {
        "total_samples": len(samples),
        "emotion_distribution": dict(emotion_counts),
        "gender_distribution": dict(gender_counts),
        "unique_actors": len({s.actor_id for s in samples}),
    }
