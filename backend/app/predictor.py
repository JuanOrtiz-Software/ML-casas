"""Conversión de la petición validada a datos del contrato y ejecución de la predicción.

El predictor construye un ``DataFrame`` con exactamente las columnas y tipos del
contrato (``astype`` según ``shared/schema.py``), ejecuta ``predict`` y calcula el
rango referencial como ``estimación ± RMSE``. La API nunca entrena ni llama a ``fit``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from backend.app.schemas import HousePredictionRequest
from shared.schema import COLUMN_DTYPES, INPUT_COLUMNS, MONEDA, TRAINING_RANGES


def _to_dataframe(payload: HousePredictionRequest) -> pd.DataFrame:
    """Convierte la petición en un DataFrame con las columnas y tipos del contrato."""
    row = {col: getattr(payload, col) for col in INPUT_COLUMNS}
    df = pd.DataFrame([row], columns=INPUT_COLUMNS)
    for col in INPUT_COLUMNS:
        df[col] = df[col].astype(COLUMN_DTYPES[col])
    return df


def _build_warnings(payload: HousePredictionRequest) -> list[str]:
    """Genera avisos en español cuando un valor cae fuera del rango de entrenamiento."""
    warnings: list[str] = []
    for col, (minimo, maximo) in TRAINING_RANGES.items():
        value = getattr(payload, col)
        if value < minimo or value > maximo:
            warnings.append(
                f"'{col}' ({value:g}) está fuera del rango de entrenamiento "
                f"[{minimo:g}, {maximo:g}]; la estimación puede ser poco fiable."
            )
    return warnings


def predict(
    payload: HousePredictionRequest,
    model: Any,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Ejecuta la predicción y devuelve la respuesta ya estructurada."""
    df = _to_dataframe(payload)
    warnings = _build_warnings(payload)

    estimacion = float(model.predict(df)[0])
    rmse = float(metadata.get("metrics", {}).get("rmse", 0.0))

    if estimacion < 0:
        warnings.append(
            "La estimación fue negativa y se ajustó a 0 (posible extrapolación "
            "fuera del rango de entrenamiento)."
        )
        estimacion = 0.0

    precio_estimado = round(estimacion, 2)
    rango_referencial = {
        "minimo": round(max(0.0, precio_estimado - rmse), 2),
        "maximo": round(precio_estimado + rmse, 2),
    }

    return {
        "success": True,
        "prediction": {
            "precio_estimado": precio_estimado,
            "moneda": MONEDA,
            "rango_referencial": rango_referencial,
        },
        "model": {
            "name": metadata["name"],
            "version": str(metadata["version"]),
            "algorithm": metadata["algorithm"],
        },
        "warnings": warnings,
    }


__all__ = ["predict"]
