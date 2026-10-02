"""Evaluación de modelos: métricas, validación cruzada y regla de selección.

La regla de selección usa la validación cruzada del conjunto de entrenamiento para
no sesgar la elección con el conjunto de prueba (que queda como estimación final e
imparcial del modelo elegido).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from shared.schema import EPSILON, SEED
from ml.models import MODEL_ORDER

CV_SPLITS: int = 5


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calcula las métricas obligatorias y ``mape`` sobre predicciones y reales."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "r2": r2_score(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
    }


def cross_validate(pipeline: Pipeline, x_train, y_train) -> dict[str, float]:
    """Validación cruzada de 5 particiones sobre el conjunto de entrenamiento.

    Se clona el pipeline en cada partición para evitar fuga de datos.
    """
    kf = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=SEED)
    r2_scores: list[float] = []
    rmse_scores: list[float] = []
    for train_idx, val_idx in kf.split(x_train):
        model = clone(pipeline)
        model.fit(x_train.iloc[train_idx], y_train.iloc[train_idx])
        pred = model.predict(x_train.iloc[val_idx])
        r2_scores.append(r2_score(y_train.iloc[val_idx], pred))
        rmse_scores.append(math.sqrt(mean_squared_error(y_train.iloc[val_idx], pred)))
    return {
        "cv_r2_mean": float(np.mean(r2_scores)),
        "cv_rmse_mean": float(np.mean(rmse_scores)),
        "cv_rmse_std": float(np.std(rmse_scores)),
    }


def select_best_model(
    results: list[dict[str, Any]], order: list[str] | None = None
) -> str:
    """Selecciona el mejor modelo de forma determinista y pura.

    Ordena por ``cv_r2_mean`` descendente; en caso de empate (diferencia menor que
    ``EPSILON``), por ``cv_rmse_mean`` ascendente; si persiste, por el orden fijo de
    :func:`ml.models.build_models`. El resultado no depende del orden de entrada.
    """
    model_order = order or MODEL_ORDER
    rank = {name: i for i, name in enumerate(model_order)}

    # Se recorre en el orden fijo, de modo que el desempate final no dependa
    # del orden en que llegan los resultados.
    ordered = sorted(
        results, key=lambda r: rank.get(str(r["model_name"]), len(model_order))
    )
    best: dict[str, Any] | None = None
    for record in ordered:
        if best is None:
            best = record
            continue
        if record["cv_r2_mean"] - best["cv_r2_mean"] > EPSILON:
            best = record
        elif abs(record["cv_r2_mean"] - best["cv_r2_mean"]) <= EPSILON:
            if best["cv_rmse_mean"] - record["cv_rmse_mean"] > EPSILON:
                best = record
    assert best is not None
    return str(best["model_name"])


__all__ = ["compute_metrics", "cross_validate", "select_best_model"]
