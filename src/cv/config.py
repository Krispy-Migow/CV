"""Carga de configuración y rutas del proyecto."""
from pathlib import Path

import yaml

# src/cv/config.py -> raíz del repo
ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(relative: str) -> Path:
    """Convierte una ruta relativa de la config en una ruta absoluta."""
    return ROOT / relative
