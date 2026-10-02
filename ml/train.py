"""Entrenamiento, evaluación, tracking y registro del modelo campeón.

Ejecuta el flujo completo: carga y valida el dataset, separa los datos, construye los
cinco pipelines, entrena cada uno en su propio run de MLflow, calcula métricas y
validación cruzada, imprime la tabla comparativa, selecciona el ganador con la regla
determinista y lo registra en el Model Registry con el alias ``champion``.

Se puede ejecutar como ``python ml/train.py`` o ``python -m ml.train``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Redirigir la caché de matplotlib a un directorio escribible antes de que se use.
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import mlflow  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from data.validate_dataset import (  # noqa: E402
    DATASET_PATH,
    load_dataset,
    validate_dataframe,
)
from ml.evaluate import compute_metrics, cross_validate, select_best_model  # noqa: E402
from ml.models import build_models, get_loggable_params  # noqa: E402
from ml.preprocessing import build_pipeline, split_data  # noqa: E402
from ml import registry  # noqa: E402
from shared.schema import INPUT_COLUMNS, SEED, TEST_SIZE, ZONAS  # noqa: E402
from shared.settings import (  # noqa: E402
    LOG_LEVEL,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_MODEL_ALIAS,
    MLFLOW_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    MLRUNS_DIR,
)

# Formato de serialización: cloudpickle funciona para los cinco modelos (skops,
# el formato por defecto de MLflow 3.x, rechaza los modelos basados en árboles
# por tipos no confiables). Decisión documentada en el README.
SERIALIZATION_FORMAT = "cloudpickle"
MODEL_ARTIFACT_NAME = "model"

logger = logging.getLogger("houseprice.ml")


def _configure_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _sha256(path: Path) -> str:
    """Huella SHA-256 del archivo (identifica el dataset usado en el entrenamiento)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _get_or_create_experiment() -> str:
    """Devuelve el id del experimento, creándolo con artifacts anclados si no existe."""
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        artifact_uri = MLRUNS_DIR.resolve().as_uri()
        logger.info(
            "Creando experimento '%s' con artifacts en %s",
            MLFLOW_EXPERIMENT_NAME,
            artifact_uri,
        )
        return mlflow.create_experiment(
            MLFLOW_EXPERIMENT_NAME, artifact_location=artifact_uri
        )
    return experiment.experiment_id


def _log_feature_importance(
    pipeline: Any, x_test: pd.DataFrame, y_test: pd.Series
) -> None:
    """Calcula y registra la importancia por permutación de las diez variables.

    Los gráficos nunca son necesarios para que el sistema funcione: si fallan se
    registra un aviso y el entrenamiento continúa.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
        import matplotlib.pyplot as plt
        from sklearn.inspection import permutation_importance

        resultado = permutation_importance(
            pipeline,
            x_test,
            y_test,
            n_repeats=5,
            random_state=SEED,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        importancias = resultado.importances_mean
        orden = np.argsort(importancias)[::-1]

        with tempfile.TemporaryDirectory() as tmpdir:
            png_path = Path(tmpdir) / "feature_importance.png"
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh([INPUT_COLUMNS[i] for i in orden], importancias[orden])
            ax.set_xlabel("Importancia (descenso de MSE por permutación)")
            ax.set_title("Importancia de variables (permutación)")
            fig.tight_layout()
            fig.savefig(png_path, bbox_inches="tight")
            plt.close(fig)
            mlflow.log_artifact(str(png_path))
    except Exception as exc:  # noqa: BLE001 — no debe detener el entrenamiento.
        logger.warning("No se pudo generar feature_importance.png: %s", exc)


def _log_session_artifacts(
    run_id: str,
    results: list[dict[str, Any]],
    df: pd.DataFrame,
) -> None:
    """Registra los artifacts de la sesión en el run ganador."""
    comparison_df = pd.DataFrame(results)
    resumen = {
        "filas": int(len(df)),
        "precio_min": float(df["precio"].min()),
        "precio_p25": float(df["precio"].quantile(0.25)),
        "precio_p50": float(df["precio"].median()),
        "precio_p75": float(df["precio"].quantile(0.75)),
        "precio_max": float(df["precio"].max()),
        "precio_medio_por_zona": {
            zona: float(df.loc[df["zona"] == zona, "precio"].mean()) for zona in ZONAS
        },
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        comparison_path = tmp / "model_comparison.csv"
        metrics_path = tmp / "metrics.json"
        summary_path = tmp / "dataset_summary.json"

        comparison_df.to_csv(comparison_path, index=False)
        metrics_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        summary_path.write_text(
            json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifact(str(comparison_path), "session")
            mlflow.log_artifact(str(metrics_path), "session")
            mlflow.log_artifact(str(summary_path), "session")


def _format_table(results: list[dict[str, Any]]) -> str:
    """Construye la tabla comparativa legible para la terminal."""
    header = (
        f"{'MODELO':<20} {'MAE':>11} {'MSE':>16} {'RMSE':>11} "
        f"{'R2':>8} {'CV_R2':>9}"
    )
    lines = [header, "-" * len(header)]
    for record in results:
        lines.append(
            f"{record['model_name']:<20} {record['mae']:>11,.0f} "
            f"{record['mse']:>16,.0f} {record['rmse']:>11,.0f} "
            f"{record['r2']:>8.4f} {record['cv_r2_mean']:>9.4f}"
        )
    return "\n".join(lines)


def run_training(
    df: pd.DataFrame,
    models: dict[str, Any],
    dataset_name: str,
    dataset_sha256: str,
    log_importance: bool = True,
) -> dict[str, Any]:
    """Ejecuta el flujo completo de entrenamiento, tracking y registro.

    Devuelve un diccionario con los resultados de los cinco modelos, el ganador,
    la versión registrada y el ``run_id`` del ganador.
    """
    x_train, x_test, y_train, y_test = split_data(df)
    training_session = datetime.now(timezone.utc).isoformat()

    experiment_id = _get_or_create_experiment()
    mlflow.set_experiment(experiment_id=experiment_id)

    client = MlflowClient()
    results: list[dict[str, Any]] = []

    # 1. Entrenar los cinco modelos, cada uno en su run.
    for model_name, model in models.items():
        algorithm = type(model).__name__
        pipeline = build_pipeline(model)

        with mlflow.start_run(run_name=model_name) as run:
            run_id = run.info.run_id

            mlflow.set_tags(
                {
                    "project": "HousePrice ML",
                    "problem_type": "regression",
                    "dataset": dataset_name,
                    "dataset_sha256": dataset_sha256,
                    "training_session": training_session,
                    "sklearn_version": __import__("sklearn").__version__,
                    "is_champion": "false",
                }
            )
            mlflow.log_params(
                {
                    "algorithm": algorithm,
                    "model_name": model_name,
                    "random_state": str(SEED),
                    "test_size": f"{TEST_SIZE:g}",
                    "train_size": f"{1 - TEST_SIZE:g}",
                    "n_train": str(len(x_train)),
                    "n_test": str(len(x_test)),
                    "n_features": str(len(INPUT_COLUMNS)),
                }
            )
            mlflow.log_params(get_loggable_params(model))

            start = time.perf_counter()
            pipeline.fit(x_train, y_train)
            fit_seconds = time.perf_counter() - start

            pred_train = pipeline.predict(x_train)
            pred_test = pipeline.predict(x_test)

            metrics = compute_metrics(y_test.values, pred_test)
            metrics["r2_train"] = compute_metrics(y_train.values, pred_train)["r2"]
            metrics["fit_seconds"] = fit_seconds
            cv = cross_validate(pipeline, x_train, y_train)
            metrics.update(cv)

            mlflow.log_metrics(metrics)

            signature = mlflow.models.infer_signature(x_train, pipeline.predict(x_train))
            model_info = mlflow.sklearn.log_model(
                pipeline,
                name=MODEL_ARTIFACT_NAME,
                signature=signature,
                input_example=x_train.head(1),
                serialization_format=SERIALIZATION_FORMAT,
            )

            if log_importance:
                _log_feature_importance(pipeline, x_test, y_test)

            record: dict[str, Any] = {
                "model_name": model_name,
                "algorithm": algorithm,
                "run_id": run_id,
                "model_uri": model_info.model_uri,
            }
            record.update(metrics)
            results.append(record)

            logger.info(
                "Run %s (%s): r2=%.4f cv_r2=%.4f rmse=%.0f",
                run_id,
                model_name,
                metrics["r2"],
                metrics["cv_r2_mean"],
                metrics["rmse"],
            )

    # 2. Tabla comparativa.
    print("-" * 60)
    print(_format_table(results))
    print("-" * 60)

    # 3. Selección del ganador.
    winner_name = select_best_model(results)
    winner = next(r for r in results if r["model_name"] == winner_name)

    best_test_r2 = max(r["r2"] for r in results)
    if abs(winner["r2"] - best_test_r2) > 1e-9:
        logger.warning(
            "El ganador por validación cruzada (%s) no coincide con el mejor R² "
            "de prueba (%.4f). Se respeta la regla de selección.",
            winner_name,
            best_test_r2,
        )

    # 4. Registro en el Model Registry.
    model_version = registry.register_model(winner["model_uri"], MLFLOW_MODEL_NAME)
    version = str(model_version.version)
    registry.set_model_alias(MLFLOW_MODEL_NAME, MLFLOW_MODEL_ALIAS, version)

    trained_at = datetime.fromtimestamp(
        client.get_run(winner["run_id"]).info.start_time / 1000, tz=timezone.utc
    ).isoformat()
    registry.set_version_tags(
        MLFLOW_MODEL_NAME,
        version,
        {
            "algorithm": winner["algorithm"],
            "model_name": winner["model_name"],
            "cv_r2_mean": f"{winner['cv_r2_mean']:.6f}",
            "r2": f"{winner['r2']:.6f}",
            "rmse": f"{winner['rmse']:.6f}",
            "dataset_sha256": dataset_sha256,
            "run_id": winner["run_id"],
            "trained_at": trained_at,
        },
    )
    registry.set_version_description(
        MLFLOW_MODEL_NAME,
        version,
        (
            f"Modelo campeón: {winner['model_name']} ({winner['algorithm']}). "
            f"cv_r2_mean={winner['cv_r2_mean']:.4f}, r2={winner['r2']:.4f}, "
            f"rmse={winner['rmse']:.0f}."
        ),
    )

    # 5. Actualizar la etiqueta is_champion en todos los runs.
    for record in results:
        client.set_tag(
            record["run_id"],
            "is_champion",
            "true" if record["run_id"] == winner["run_id"] else "false",
        )

    # 6. Artifacts de la sesión en el run ganador.
    _log_session_artifacts(winner["run_id"], results, df)

    print(f"MEJOR MODELO: {winner_name}   (criterio: CV R² ↓, CV RMSE ↑)")
    print(
        f"Modelo registrado: {MLFLOW_MODEL_NAME}  versión {version}  "
        f"alias {MLFLOW_MODEL_ALIAS}"
    )
    print("=" * 60)

    return {
        "results": results,
        "winner_name": winner_name,
        "winner": winner,
        "version": version,
        "run_id": winner["run_id"],
    }


def main() -> None:
    """Punto de entrada del entrenamiento completo."""
    _configure_logging()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # 1. Comprobar y validar el dataset.
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset no encontrado en {DATASET_PATH}. Ejecuta "
            "`python data/generate_dataset.py` primero."
        )
    df = load_dataset(DATASET_PATH)
    validation = validate_dataframe(df)
    if not validation.is_valid:
        raise ValueError(
            "El dataset no es válido:\n  " + "\n  ".join(validation.errors)
        )

    dataset_sha256 = _sha256(DATASET_PATH)

    print("=" * 60)
    print("HOUSEPRICE ML — ENTRENAMIENTO")
    print("=" * 60)
    print(
        f"Dataset: {DATASET_PATH.name}   Registros: {len(df)}   "
        f"Train: {int(len(df) * (1 - TEST_SIZE))}   Test: {int(len(df) * TEST_SIZE)}"
    )

    run_training(df, build_models(), DATASET_PATH.name, dataset_sha256)


if __name__ == "__main__":
    main()
