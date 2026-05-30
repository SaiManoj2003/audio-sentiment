"""
Central configuration — all tuneable values live here.
Nothing is hardcoded anywhere else in the codebase.
"""

from dataclasses import dataclass, field
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"


@dataclass
class AudioConfig:
    """Acoustic feature extraction settings."""
    sample_rate: int = 16000
    n_mfcc: int = 40
    hop_length: int = 512
    n_fft: int = 2048


@dataclass
class ASRConfig:
    """Whisper transcription settings."""
    model_size: str = "medium"        # tiny | base | small | medium | large-v3
    device: str = "cuda"              # cuda | cpu
    compute_type: str = "float16"     # float16 for GPU, int8 for CPU
    language: str = "en"
    beam_size: int = 5


@dataclass
class TextEmotionConfig:
    """Text emotion model settings."""
    model_name: str = "j-hartmann/emotion-english-distilroberta-base"
    device: str = "cuda"
    max_length: int = 512
    batch_size: int = 16


@dataclass
class AudioEmotionConfig:
    """Audio emotion model settings."""
    model_name: str = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    device: str = "cuda"
    sample_rate: int = 16000
    batch_size: int = 8


@dataclass
class FusionConfig:
    """Late-fusion weighting between text and audio branches."""
    text_weight: float = 0.3
    audio_weight: float = 0.7
    # Audio weighted higher because RAVDESS emotion is carried by voice
    # tone, not word choice. For real call analysis where language matters
    # more, adjust toward 0.5/0.5.
    low_confidence_threshold: float = 0.35
    # If the dominant emotion probability is below this threshold,
    # dominant_emotion is reported as "uncertain" in the output.

    def __post_init__(self):
        assert abs(self.text_weight + self.audio_weight - 1.0) < 1e-6, \
            "text_weight + audio_weight must equal 1.0"


@dataclass
class APIConfig:
    """FastAPI server settings."""
    host: str = "0.0.0.0"
    port: int = 8000
    max_file_size_mb: int = 100
    request_timeout_seconds: int = 300


@dataclass
class EvalConfig:
    """Evaluation settings."""
    dataset: str = "RAVDESS"
    data_dir: Path = DATA_DIR / "ravdess"
    results_dir: Path = ROOT_DIR / "results"
    random_seed: int = 42


@dataclass
class Config:
    asr: ASRConfig = field(default_factory=ASRConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    text_emotion: TextEmotionConfig = field(default_factory=TextEmotionConfig)
    audio_emotion: AudioEmotionConfig = field(default_factory=AudioEmotionConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    api: APIConfig = field(default_factory=APIConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


cfg = Config()
