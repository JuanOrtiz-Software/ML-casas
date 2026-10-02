"""Punto de entrada de la API FastAPI.

``main.py`` solo crea la aplicación, el middleware CORS, los manejadores de error y
las rutas; no contiene lógica de negocio. El modelo se carga una sola vez al iniciar
mediante el *lifespan* y queda en ``app.state``. La API nunca entrena.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app import model_loader, predictor, schemas
from backend.app.config import FRONTEND_URL
from shared.schema import MONEDA

logger = logging.getLogger("houseprice.api")

SERVICE_NAME = "houseprice-api"


class APIError(Exception):
    """Error de dominio con código y detalles ya estructurados."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: list[dict[str, str]] | None = None,
    ) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or []
        super().__init__(message)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo una sola vez al arrancar; si falla, entra en estado degradado."""
    try:
        model, metadata = model_loader.load_model_and_metadata()
        app.state.model = model
        app.state.metadata = metadata
        logger.info("HOUSEPRICE ML API lista con modelo cargado.")
    except Exception as exc:  # noqa: BLE001 — la API debe arrancar igualmente.
        app.state.model = None
        app.state.metadata = None
        logger.error("No se pudo cargar el modelo desde el Registry: %s", exc)
        logger.warning("La API arranca en estado degradado (sin modelo).")
    yield


app = FastAPI(
    title="HOUSEPRICE ML API",
    description=(
        "API de predicción de precios de vivienda. Carga el modelo campeón desde "
        "MLflow Model Registry y sirve predicciones sobre datos crudos."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_URL,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
    allow_credentials=False,
)


# ---------------------------------------------------------------------------
# Manejadores de error (respuestas uniformes)
# ---------------------------------------------------------------------------
def _error_response(
    status_code: int,
    code: str,
    message: str,
    details: list[dict[str, str]] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "error": {
                "code": code,
                "message": message,
                "details": details or [],
            },
        },
    )


@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    return _error_response(exc.status_code, exc.code, exc.message, exc.details)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    details: list[dict[str, str]] = []
    for err in exc.errors():
        loc: tuple[Any, ...] = err.get("loc", ())
        # Errores a nivel de modelo (p. ej. coherencia) llegan con loc ("body",).
        if len(loc) <= 1:
            campo = "coherencia"
        else:
            campo = str(loc[-1])
        details.append({"campo": campo, "mensaje": err.get("msg", "Entrada no válida.")})
    return _error_response(
        422,
        "VALIDATION_ERROR",
        "La petición contiene datos no válidos. Revisa los detalles.",
        details,
    )


@app.exception_handler(Exception)
async def internal_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Error interno no controlado: %s", exc)
    return _error_response(500, "INTERNAL_ERROR", "Error interno del servidor.")


# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
@app.get("/health", response_model=schemas.HealthResponse)
def health(request: Request) -> Any:
    """Estado del servicio. Distingue API en línea de API en línea sin modelo."""
    loaded = request.app.state.model is not None
    if loaded:
        return {"status": "ok", "service": SERVICE_NAME, "model_loaded": True}
    return JSONResponse(
        status_code=503,
        content={"status": "degraded", "service": SERVICE_NAME, "model_loaded": False},
    )


@app.post("/predict", response_model=schemas.PredictionResponse)
def predict_payload(
    payload: schemas.HousePredictionRequest, request: Request
) -> dict[str, Any]:
    """Predice el precio de una vivienda a partir de datos crudos."""
    model = request.app.state.model
    metadata = request.app.state.metadata
    if model is None or metadata is None:
        raise APIError(
            503,
            "MODEL_UNAVAILABLE",
            "El modelo no está disponible. Entrena el modelo antes de consultar la API.",
        )
    return predictor.predict(payload, model, metadata)


@app.get("/model-info", response_model=schemas.ModelInfoResponse)
def model_info(request: Request) -> dict[str, Any]:
    """Metadatos del modelo campeón y descripción de las variables."""
    metadata = request.app.state.metadata
    if metadata is None:
        raise APIError(
            503,
            "MODEL_UNAVAILABLE",
            "El modelo no está disponible. Entrena el modelo antes de consultar la API.",
        )
    return {
        "name": metadata["name"],
        "version": metadata["version"],
        "alias": metadata["alias"],
        "algorithm": metadata["algorithm"],
        "algorithm_name": metadata["algorithm_name"],
        "run_id": metadata["run_id"],
        "trained_at": metadata["trained_at"],
        "metrics": metadata["metrics"],
        "currency": MONEDA,
        "variables": schemas.build_variables_info(),
    }


__all__ = ["app", "APIError"]
