"""Los cinco modelos del proyecto.

Define exactamente cinco modelos de scikit-learn, cada uno con su nombre legible,
su clase y sus hiperparámetros de partida (sin búsqueda de hiperparámetros).
El orden del diccionario devuelto por :func:`build_models` es fijo y se usa para
los desempates de la regla de selección y para imprimir los resultados.
"""

from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# Orden fijo (clave de desempate de la regla de selección).
MODEL_ORDER: list[str] = [
    "Linear Regression",
    "Decision Tree",
    "Random Forest",
    "Gradient Boosting",
    "Extra Trees",
]

# Hiperparámetros que se registran en MLflow para cada clase de modelo.
PARAM_WHITELIST: dict[str, list[str]] = {
    "LinearRegression": ["fit_intercept"],
    "DecisionTreeRegressor": [
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "random_state",
    ],
    "RandomForestRegressor": [
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "random_state",
    ],
    "GradientBoostingRegressor": [
        "n_estimators",
        "learning_rate",
        "max_depth",
        "random_state",
    ],
    "ExtraTreesRegressor": [
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "random_state",
    ],
}


def build_models(reduced: bool = False) -> dict[str, BaseEstimator]:
    """Devuelve un diccionario ordenado con los cinco modelos (nombre → instancia).

    Con ``reduced=True`` se usan hiperparámetros reducidos para que el entrenamiento
    sea rápido (solo para las pruebas; el flujo real usa ``reduced=False``).
    """
    if reduced:
        return {
            "Linear Regression": LinearRegression(fit_intercept=True),
            "Decision Tree": DecisionTreeRegressor(
                max_depth=4, min_samples_split=5, min_samples_leaf=2, random_state=42
            ),
            "Random Forest": RandomForestRegressor(
                n_estimators=20,
                max_depth=6,
                min_samples_split=4,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=42,
            ),
            "Gradient Boosting": GradientBoostingRegressor(
                n_estimators=20, learning_rate=0.1, max_depth=2, random_state=42
            ),
            "Extra Trees": ExtraTreesRegressor(
                n_estimators=20,
                max_depth=8,
                min_samples_split=4,
                n_jobs=-1,
                random_state=42,
            ),
        }
    return {
        "Linear Regression": LinearRegression(fit_intercept=True),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=8, min_samples_split=10, min_samples_leaf=4, random_state=42
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=14,
            min_samples_split=4,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=300,
            max_depth=16,
            min_samples_split=4,
            n_jobs=-1,
            random_state=42,
        ),
    }


def get_loggable_params(model: BaseEstimator) -> dict[str, str]:
    """Filtra ``get_params()`` con la lista explícita por clase, con ``None`` como texto."""
    algorithm = type(model).__name__
    whitelist = PARAM_WHITELIST.get(algorithm, [])
    params = model.get_params()
    return {
        key: ("None" if params[key] is None else str(params[key]))
        for key in whitelist
        if key in params
    }


__all__ = ["build_models", "get_loggable_params", "MODEL_ORDER", "PARAM_WHITELIST"]
