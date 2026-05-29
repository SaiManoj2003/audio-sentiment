"""
Central configuration — all tuneable values live here.
Nothing is hardcoded anywhere else in the codebase.
"""

from dataclasses import dataclass, field
from pathlib import Path

# Project root — reliable regardless of where the script is called from
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"

@dataclass
class AudioConfig:
    """Acoustic feature extraction settings."""
    sample_rate: int = 16000          # Hz — wav2vec2 expects 16kHz
    n_mfcc: int = 40                  # number of MFCC coefficients
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

@dataclass
class AudioEmotionConfig:
    """Audio emotion model settings."""
    model_name: str = "r-f/wav2vec-english-speech-emotion-recognition"
    device: str = "cuda"
    sample_rate: int = 16000

@dataclass
class FusionConfig:
    """Late-fusion weighting between text and audio branches."""
    text_weight: float = 0.3
    audio_weight: float = 0.7        # must sum to 1.0
    # Audio weighted higher because RAVDESS emotion is carried
    # by voice tone, not word choice. For real call analysis
    # where language matters more, adjust toward 0.5/0.5.

    def __post_init__(self):
        assert abs(self.text_weight + self.audio_weight - 1.0) < 1e-6, \
            "text_weight + audio_weight must equal 1.0"

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
    eval: EvalConfig = field(default_factory=EvalConfig)


# Single importable instance
cfg = Config()