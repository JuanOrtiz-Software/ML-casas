"""Preprocesamiento y división de datos.

Define la separación de ``X``/``y``, la división única en entrenamiento/prueba
y el ``ColumnTransformer`` que forma parte del ``Pipeline`` completo de cada modelo.
El pipeline completo (preprocesamiento + modelo) es lo que se entrena, evalúa,
registra en MLflow y sirve desde la API, para que no haya diferencias entre
entrenamiento y predicción.
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from shared.schema import (
    BINARY_COLUMNS,
    CATEGORICAL_COLUMN,
    INPUT_COLUMNS,
    SCALED_COLUMNS,
    SEED,
    TARGET,
    TEST_SIZE,
    ZONAS,
)


def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separa ``X``/``y`` y divide en entrenamiento (80 %) y prueba (20 %).

    La división se hace una sola vez y de forma compartida para que todos los
    modelos vean exactamente los mismos datos.
    """
    x = df[INPUT_COLUMNS]
    y = df[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=SEED
    )
    return x_train, x_test, y_train, y_test


def build_preprocessor() -> ColumnTransformer:
    """Construye el preprocesador con el contrato de columnas.

    * numéricas (continuas y de conteo): ``StandardScaler``;
    * binarias ``garage``/``jardin``: passthrough;
    * ``zona``: ``OneHotEncoder`` con categorías fijas y ``handle_unknown="error"``.
    """
    return ColumnTransformer(
        transformers=[
            ("numericas", StandardScaler(), SCALED_COLUMNS),
            ("binarias", "passthrough", BINARY_COLUMNS),
            (
                "zona",
                OneHotEncoder(
                    categories=[ZONAS],
                    handle_unknown="error",
                    sparse_output=False,
                ),
                [CATEGORICAL_COLUMN],
            ),
        ],
        verbose_feature_names_out=False,
    )


def build_pipeline(model: BaseEstimator) -> Pipeline:
    """Envuelve el preprocesador y el modelo en un único ``Pipeline``."""
    return Pipeline(
        steps=[
            ("preprocesamiento", build_preprocessor()),
            ("modelo", model),
        ]
    )


__all__ = ["split_data", "build_preprocessor", "build_pipeline"]
