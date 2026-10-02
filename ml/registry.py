"""Lógica aislada del Model Registry de MLflow.

Solo se registra el modelo ganador. La política de promoción es simple y
determinista: cada sesión registra una nueva versión y el alias ``champion``
apunta a la versión de la última sesión. Las versiones anteriores se conservan.
"""

from __future__ import annotations

from typing import Any

import mlflow
from mlflow.entities import Run
from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient


def register_model(model_uri: str, name: str) -> ModelVersion:
    """Registra el modelo en la URI indicada bajo ``name`` y devuelve la versión."""
    return mlflow.register_model(model_uri=model_uri, name=name)


def set_model_alias(name: str, alias: str, version: str) -> None:
    """Asigna el alias a una versión concreta del modelo registrado."""
    MlflowClient().set_registered_model_alias(name=name, alias=alias, version=version)


def set_version_tags(name: str, version: str, tags: dict[str, str]) -> None:
    """Añade etiquetas a una versión del modelo registrado."""
    client = MlflowClient()
    for key, value in tags.items():
        client.set_model_version_tag(
            name=name, version=version, key=key, value=str(value)
        )


def set_version_description(name: str, version: str, description: str) -> None:
    """Actualiza la descripción de una versión del modelo registrado."""
    MlflowClient().update_model_version(
        name=name, version=version, description=description
    )


def get_model_version_by_alias(name: str, alias: str) -> ModelVersion:
    """Devuelve la versión asociada al alias, o lanza si no existe."""
    return MlflowClient().get_model_version_by_alias(name=name, alias=alias)


def get_run(run_id: str) -> Run:
    """Devuelve el objeto ``Run`` con sus parámetros, métricas y etiquetas."""
    return MlflowClient().get_run(run_id=run_id)


def get_registered_model(name: str) -> Any:
    """Devuelve el modelo registrado (incluye sus aliases)."""
    return MlflowClient().get_registered_model(name=name)


__all__ = [
    "register_model",
    "set_model_alias",
    "set_version_tags",
    "set_version_description",
    "get_model_version_by_alias",
    "get_run",
    "get_registered_model",
]
