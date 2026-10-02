"""Configuración central del proyecto, cargada desde variables de entorno.

La configuración se lee del archivo ``.env`` (si existe) mediante ``python-dotenv``.
Todas las rutas se anclan a la raíz del proyecto para que los scripts funcionen igual
lanzados desde cualquier directorio. En particular, una URI de tracking SQLite relativa
(``sqlite:///mlflow.db``) se convierte a una ruta absoluta anclada en la raíz, de modo
que el entrenamiento y la API apunten siempre a la misma base de datos.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Raíz del proyecto: shared/ -> raíz.
ROOT_DIR: Path = Path(__file__).resolve().parents[1]

# Carga el .env si existe. No sobrescribe variables ya presentes en el entorno.
load_dotenv(ROOT_DIR / ".env")


def _resolve_tracking_uri(uri: str) -> str:
    """Convierte una URI SQLite relativa en una ruta absoluta anclada a la raíz."""
    prefix = "sqlite:///"
    if uri.startswith(prefix) and not uri.startswith("sqlite:////"):
        relative = uri[len(prefix):]
        absolute = (ROOT_DIR / relative).resolve()
        return f"sqlite:///{absolute}"
    return uri


def _as_int(value: str | None, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


def _as_list(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI: str = _resolve_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
)
MLFLOW_EXPERIMENT_NAME: str = os.getenv(
    "MLFLOW_EXPERIMENT_NAME", "HousePricePrediction"
)
MLFLOW_MODEL_NAME: str = os.getenv("MLFLOW_MODEL_NAME", "HousePriceModel")
MLFLOW_MODEL_ALIAS: str = os.getenv("MLFLOW_MODEL_ALIAS", "champion")

# Ruta absoluta donde MLflow guarda los artifacts.
MLRUNS_DIR: Path = ROOT_DIR / "mlruns"

# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------
BACKEND_HOST: str = os.getenv("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT: int = _as_int(os.getenv("BACKEND_PORT"), 8000)
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# ---------------------------------------------------------------------------
# Frontend / CORS
# ---------------------------------------------------------------------------
FRONTEND_URL: list[str] = _as_list(
    os.getenv("FRONTEND_URL"),
    ["http://localhost:3000", "http://127.0.0.1:3000"],
)

__all__ = [
    "ROOT_DIR",
    "MLFLOW_TRACKING_URI",
    "MLFLOW_EXPERIMENT_NAME",
    "MLFLOW_MODEL_NAME",
    "MLFLOW_MODEL_ALIAS",
    "MLRUNS_DIR",
    "BACKEND_HOST",
    "BACKEND_PORT",
    "LOG_LEVEL",
    "FRONTEND_URL",
]
