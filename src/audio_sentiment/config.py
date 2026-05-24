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

