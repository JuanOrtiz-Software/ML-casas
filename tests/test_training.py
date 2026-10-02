"""Pruebas del entrenamiento, MLflow y la regla de selección."""

from __future__ import annotations

import mlflow
from mlflow.tracking import MlflowClient

from ml.evaluate import select_best_model
from ml.models import MODEL_ORDER, build_models
from ml.registry import get_model_version_by_alias

CLASES_ESPERADAS = {
    "LinearRegression",
    "DecisionTreeRegressor",
    "RandomForestRegressor",
    "GradientBoostingRegressor",
    "ExtraTreesRegressor",
}


def test_cinco_modelos_de_clases_distintas() -> None:
    """Los cinco modelos son instancias de cinco clases distintas."""
    models = build_models()
    assert len(models) == 5
    assert {type(m).__name__ for m in models.values()} == CLASES_ESPERADAS
    assert list(models.keys()) == MODEL_ORDER


def test_se_crean_cinco_runs_con_parametros_metricas_etiquetas(
    trained_store: dict,
) -> None:
    """El entrenamiento crea cinco runs con parámetros, métricas y etiquetas."""
    client = MlflowClient()
    run_ids = [r["run_id"] for r in trained_store["results"]]
    assert len(run_ids) == 5

    for run_id in run_ids:
        run = client.get_run(run_id)
        data = run.data
        # Parámetros obligatorios.
        for key in [
            "algorithm",
            "model_name",
            "random_state",
            "test_size",
            "train_size",
            "n_train",
            "n_test",
            "n_features",
        ]:
            assert key in data.params, f"Falta el parámetro {key} en {run_id}"
        # Métricas obligatorias.
        for key in ["mae", "mse", "rmse", "r2"]:
            assert key in data.metrics, f"Falta la métrica {key} en {run_id}"
        # Etiquetas.
        for key in ["project", "problem_type", "dataset", "dataset_sha256",
                    "training_session", "sklearn_version", "is_champion"]:
            assert key in data.tags, f"Falta la etiqueta {key} en {run_id}"


def test_existen_las_cuatro_metricas_obligatorias(trained_store: dict) -> None:
    """Cada resultado incluye las cuatro métricas obligatorias."""
    for record in trained_store["results"]:
        for key in ["mae", "mse", "rmse", "r2"]:
            assert key in record and isinstance(record[key], (int, float))


def test_select_best_model_determinista_y_sin_depender_del_orden() -> None:
    """La regla de selección es determinista e independiente del orden de entrada."""
    a = {"model_name": "Random Forest", "cv_r2_mean": 0.91, "cv_rmse_mean": 100.0}
    b = {"model_name": "Linear Regression", "cv_r2_mean": 0.91, "cv_rmse_mean": 100.0}
    c = {"model_name": "Decision Tree", "cv_r2_mean": 0.80, "cv_rmse_mean": 200.0}

    assert select_best_model([a, b, c]) == "Linear Regression"
    assert select_best_model([c, b, a]) == "Linear Regression"
    assert select_best_model([b, a, c]) == "Linear Regression"


def test_select_best_model_empate_en_r2_desempata_por_rmse() -> None:
    """Ante empate en cv_r2, gana el de menor cv_rmse."""
    alto_rmse = {"model_name": "Random Forest", "cv_r2_mean": 0.90, "cv_rmse_mean": 150.0}
    bajo_rmse = {"model_name": "Extra Trees", "cv_r2_mean": 0.90, "cv_rmse_mean": 120.0}
    assert select_best_model([alto_rmse, bajo_rmse]) == "Extra Trees"


def test_select_best_model_gana_mayor_r2() -> None:
    """Gana el modelo con mayor cv_r2 independientemente del orden fijo."""
    peor = {"model_name": "Linear Regression", "cv_r2_mean": 0.70, "cv_rmse_mean": 10.0}
    mejor = {"model_name": "Extra Trees", "cv_r2_mean": 0.95, "cv_rmse_mean": 999.0}
    assert select_best_model([peor, mejor]) == "Extra Trees"


def test_version_registrada_con_alias_champion(trained_store: dict) -> None:
    """Existe una versión registrada con el alias champion."""
    mlflow.set_tracking_uri(trained_store["tracking_uri"])
    mv = get_model_version_by_alias(trained_store["model_name"], trained_store["alias"])
    assert str(mv.version) == trained_store["version"]
    assert "algorithm" in mv.tags


def test_modelo_cargado_acepta_datos_crudos(trained_store: dict) -> None:
    """El modelo cargado desde el Registry acepta datos crudos y devuelve un número."""
    mlflow.set_tracking_uri(trained_store["tracking_uri"])
    model = mlflow.sklearn.load_model(
        f"models:/{trained_store['model_name']}@{trained_store['alias']}"
    )
    x_sample = trained_store["x_sample"]
    assert list(x_sample.columns) == [
        "metros_construidos",
        "metros_terreno",
        "habitaciones",
        "banos",
        "antiguedad",
        "distancia_centro",
        "garage",
        "jardin",
        "pisos",
        "zona",
    ]
    prediction = model.predict(x_sample)
    assert len(prediction) == len(x_sample)
    assert all(isinstance(value, float) for value in prediction)
