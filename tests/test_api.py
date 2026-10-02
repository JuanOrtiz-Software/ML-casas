"""Pruebas de la API FastAPI (contra el almacén temporal, sin tocar el real)."""

from __future__ import annotations

import copy

import mlflow
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from shared.schema import INPUT_COLUMNS, ZONAS
from shared.settings import ROOT_DIR

VALID_PAYLOAD: dict = {
    "metros_construidos": 120,
    "metros_terreno": 200,
    "habitaciones": 3,
    "banos": 2,
    "antiguedad": 5,
    "distancia_centro": 4.2,
    "garage": 1,
    "jardin": 1,
    "pisos": 2,
    "zona": "residencial",
}


def test_health_normal(client) -> None:
    """/health devuelve 200 con el modelo cargado."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_health_degradado(degraded_client) -> None:
    """/health devuelve 503 degradado cuando no hay modelo."""
    response = degraded_client.get("/health")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["model_loaded"] is False


def test_predict_valido(client, trained_store: dict) -> None:
    """/predict con entrada válida devuelve una predicción con la forma esperada."""
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    prediction = body["prediction"]
    assert isinstance(prediction["precio_estimado"], (int, float))
    assert prediction["moneda"] == "MXN"
    minimo = prediction["rango_referencial"]["minimo"]
    maximo = prediction["rango_referencial"]["maximo"]
    assert minimo <= prediction["precio_estimado"] <= maximo
    assert body["model"]["name"] == trained_store["model_name"]
    assert body["model"]["version"] == trained_store["version"]
    assert body["model"]["algorithm"] != ""
    assert body["warnings"] == []


@pytest.mark.parametrize(
    "payload, campo",
    [
        ({**VALID_PAYLOAD, "zona": "lujo"}, "zona"),
        ({**VALID_PAYLOAD, "metros_construidos": -5}, "metros_construidos"),
        ({k: v for k, v in VALID_PAYLOAD.items() if k != "zona"}, "zona"),
        ({**VALID_PAYLOAD, "garage": 2}, "garage"),
        ({**VALID_PAYLOAD, "metros_terreno": 50}, "coherencia"),
    ],
)
def test_predict_invalido_formato_error(client, payload: dict, campo: str) -> None:
    """Entradas inválidas devuelven 422 con el formato de error de la sección 12.4."""
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] == "VALIDATION_ERROR"
    campos = [detail["campo"] for detail in body["error"]["details"]]
    assert campo in campos


def test_model_info(client, trained_store: dict) -> None:
    """/model-info devuelve métricas, algoritmo y descripción de variables."""
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == trained_store["model_name"]
    assert body["version"] == trained_store["version"]
    assert body["alias"] == trained_store["alias"]
    assert body["algorithm"] in {
        "LinearRegression",
        "DecisionTreeRegressor",
        "RandomForestRegressor",
        "GradientBoostingRegressor",
        "ExtraTreesRegressor",
    }
    for key in ["mae", "mse", "rmse", "r2"]:
        assert key in body["metrics"]
    assert body["currency"] == "MXN"
    assert len(body["variables"]) == 10


def test_predict_sin_modelo_devuelve_503(degraded_client) -> None:
    """/predict devuelve 503 (no 500) cuando no hay modelo."""
    response = degraded_client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 503
    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] == "MODEL_UNAVAILABLE"


@pytest.mark.integration
def test_orden_de_precio_por_zona_champion_real() -> None:
    """Integración: el champion real ordena el precio económica < media < residencial < premium."""
    real_uri = f"sqlite:///{ROOT_DIR / 'mlflow.db'}"
    mlflow.set_tracking_uri(real_uri)
    try:
        model = mlflow.sklearn.load_model("models:/HousePriceModel@champion")
    except Exception:  # noqa: BLE001 — se omite si no hay un champion real.
        pytest.skip("No hay un champion real entrenado (ejecuta python ml/train.py).")

    precios: list[float] = []
    for zona in ZONAS:
        fila = copy.deepcopy(VALID_PAYLOAD)
        fila["zona"] = zona
        df = pd.DataFrame([fila], columns=INPUT_COLUMNS)
        precios.append(float(model.predict(df)[0]))

    assert precios == sorted(precios), f"Orden incorrecto de precios: {precios}"
