"""Carga del modelo desde el Model Registry y resolución de sus metadatos.

El modelo se carga una sola vez al arrancar (desde el alias ``champion``) y sus
metadatos se leen del Registry y del run asociado, guardándose en memoria. La API
no consulta MLflow en cada petición y no depende de él después del arranque.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import mlflow

from backend.app.config import (
    MLFLOW_MODEL_ALIAS,
    MLFLOW_MODEL_NAME,
    MLFLOW_TRACKING_URI,
)
from ml.registry import get_model_version_by_alias, get_run

logger = logging.getLogger(__name__)

# Claves de métricas que se exponen en la API.
METRIC_KEYS: list[str] = [
    "mae",
    "mse",
    "rmse",
    "r2",
    "cv_r2_mean",
    "cv_rmse_mean",
    "cv_rmse_std",
]


def _iso_utc_from_epoch_ms(epoch_ms: int) -> str:
    return datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc).isoformat()


def load_model_and_metadata() -> tuple[Any, dict[str, Any]]:
    """Carga el modelo campeón y devuelve ``(modelo, metadatos)``.

    Lanza una excepción si no hay un modelo con el alias configurado; la API lo
    captura y entra en estado degradado.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model_version = get_model_version_by_alias(MLFLOW_MODEL_NAME, MLFLOW_MODEL_ALIAS)
    model_uri = f"models:/{MLFLOW_MODEL_NAME}@{MLFLOW_MODEL_ALIAS}"
    model = mlflow.sklearn.load_model(model_uri)

    run = get_run(model_version.run_id)
    tags: dict[str, str] = dict(model_version.tags or {})
    params: dict[str, str] = dict(run.data.params or {})

    algorithm = tags.get("algorithm") or params.get("algorithm", "desconocido")
    algorithm_name = tags.get("model_name") or params.get("model_name", algorithm)
    trained_at = tags.get("trained_at") or _iso_utc_from_epoch_ms(run.info.start_time)

    metrics = {
        key: float(run.data.metrics[key])
        for key in METRIC_KEYS
        if key in run.data.metrics
    }

    metadata: dict[str, Any] = {
        "name": MLFLOW_MODEL_NAME,
        "version": str(model_version.version),
        "alias": MLFLOW_MODEL_ALIAS,
        "algorithm": algorithm,
        "algorithm_name": algorithm_name,
        "run_id": model_version.run_id,
        "trained_at": trained_at,
        "metrics": metrics,
    }
    logger.info(
        "Modelo cargado: %s@%s (algoritmo=%s, run_id=%s)",
        MLFLOW_MODEL_NAME,
        MLFLOW_MODEL_ALIAS,
        algorithm,
        model_version.run_id,
    )
    return model, metadata


__all__ = ["load_model_and_metadata"]
