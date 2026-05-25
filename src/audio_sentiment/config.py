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
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)

# Single importable instance
cfg = Config()