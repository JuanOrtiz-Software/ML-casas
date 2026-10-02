"""Configuración compartida de pytest.

Las pruebas **no tocan** el ``mlflow.db`` real: antes de importar los módulos de la
aplicación se sobrescribe la configuración por variables de entorno para apuntar a un
almacén SQLite temporal. Una fixture de ámbito de sesión genera un dataset pequeño,
entrena los cinco modelos con hiperparámetros reducidos y registra el campeón en ese
almacén temporal.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Redirigir MLflow a un almacén temporal ANTES de importar la aplicación.
# ---------------------------------------------------------------------------
_TMP_STORE = tempfile.mkdtemp(prefix="houseprice_test_")
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{_TMP_STORE}/mlflow.db"
os.environ["MLFLOW_EXPERIMENT_NAME"] = "HousePricePredictionTest"
os.environ["MLFLOW_MODEL_NAME"] = "HousePriceModelTest"
os.environ["MLFLOW_MODEL_ALIAS"] = "champion"

import mlflow  # noqa: E402

from data.generate_dataset import generate_dataset  # noqa: E402
from ml.models import build_models  # noqa: E402
from ml.train import run_training  # noqa: E402
from shared.settings import (  # noqa: E402
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_MODEL_ALIAS,
    MLFLOW_MODEL_NAME,
    MLFLOW_TRACKING_URI,
)

TEST_TRACKING_URI = MLFLOW_TRACKING_URI


def pytest_configure(config) -> None:
    """Registra el marcador ``integration`` para evitar avisos de pytest."""
    config.addinivalue_line(
        "markers",
        "integration: prueba de integración que requiere el champion real entrenado",
    )


@pytest.fixture(autouse=True)
def _reset_tracking_uri() -> None:
    """Garantiza que cada prueba arranque apuntando al almacén temporal."""
    mlflow.set_tracking_uri(TEST_TRACKING_URI)
    yield


@pytest.fixture(scope="session")
def trained_store(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Entrena los cinco modelos (hiperparámetros reducidos) en el almacén temporal."""
    data_dir = tmp_path_factory.mktemp("dataset")
    df = generate_dataset(n=300, seed=123)
    dataset_path = data_dir / "small_dataset.csv"
    df.to_csv(dataset_path, index=False, encoding="utf-8")
    sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    info = run_training(
        df=df,
        models=build_models(reduced=True),
        dataset_name=dataset_path.name,
        dataset_sha256=sha256,
        log_importance=False,
    )

    return {
        "results": info["results"],
        "winner_name": info["winner_name"],
        "winner": info["winner"],
        "version": info["version"],
        "run_id": info["run_id"],
        "experiment_name": MLFLOW_EXPERIMENT_NAME,
        "model_name": MLFLOW_MODEL_NAME,
        "alias": MLFLOW_MODEL_ALIAS,
        "tracking_uri": MLFLOW_TRACKING_URI,
        "x_sample": df.drop(columns=["precio"]).head(3),
    }


@pytest.fixture()
def client(trained_store: dict):
    """Cliente de pruebas contra la API con el modelo del almacén temporal cargado."""
    from fastapi.testclient import TestClient

    from backend.app.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def degraded_client(monkeypatch: pytest.MonkeyPatch):
    """Cliente de pruebas con la carga del modelo forzada a fallar (estado degradado)."""
    from fastapi.testclient import TestClient

    import backend.app.model_loader as model_loader
    from backend.app.main import app

    def _fallar(*args, **kwargs):
        raise RuntimeError("Modelo no disponible en la prueba.")

    monkeypatch.setattr(model_loader, "load_model_and_metadata", _fallar)
    with TestClient(app) as test_client:
        yield test_client
