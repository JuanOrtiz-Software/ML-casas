"""Esquemas Pydantic de la API.

Define el contrato de entrada (``HousePredictionRequest``) y las respuestas.
La validación tiene dos niveles:

* límites físicos duros (rechazan con 422);
* rangos de entrenamiento, más estrechos, que no rechazan sino que generan avisos.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from shared.schema import (
    COLUMN_DTYPES,
    TRAINING_RANGES,
    ZONA_LABELS,
    ZONAS,
)

# ---------------------------------------------------------------------------
# Límites físicos duros por campo (mínimo, máximo, mínimo_inclusivo)
# ---------------------------------------------------------------------------
LIMITES_FISICOS: dict[str, tuple[float, float, bool]] = {
    "metros_construidos": (0.0, 2000.0, False),
    "metros_terreno": (0.0, 10000.0, False),
    "habitaciones": (1.0, 20.0, True),
    "banos": (1.0, 15.0, True),
    "antiguedad": (0.0, 150.0, True),
    "distancia_centro": (0.0, 200.0, True),
    "garage": (0.0, 1.0, True),
    "jardin": (0.0, 1.0, True),
    "pisos": (1.0, 10.0, True),
}

CAMPOS_NUMERICOS = tuple(LIMITES_FISICOS.keys())


class HousePredictionRequest(BaseModel):
    """Petición de predicción con las diez variables de entrada (datos crudos)."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
        }
    )

    metros_construidos: float
    metros_terreno: float
    habitaciones: int
    banos: int
    antiguedad: int
    distancia_centro: float
    garage: int
    jardin: int
    pisos: int
    zona: str

    @field_validator(*CAMPOS_NUMERICOS, mode="before")
    @classmethod
    def _rechazar_no_numerico(cls, value: Any) -> Any:
        """Rechaza texto o booleanos donde se espera un número (tipos estrictos)."""
        if isinstance(value, (str, bool)):
            raise ValueError("se espera un valor numérico")
        return value

    @field_validator(*CAMPOS_NUMERICOS)
    @classmethod
    def _validar_limites(cls, value: float, info: Any) -> float:
        minimo, maximo, minimo_inclusivo = LIMITES_FISICOS[info.field_name]
        por_debajo = value < minimo if minimo_inclusivo else value <= minimo
        if por_debajo or value > maximo:
            comparador = ">=" if minimo_inclusivo else ">"
            raise ValueError(f"debe cumplir {comparador} {minimo:g} y <= {maximo:g}")
        return value

    @field_validator("zona", mode="before")
    @classmethod
    def _normalizar_zona(cls, value: Any) -> Any:
        if not isinstance(value, str):
            raise ValueError("se espera un texto")
        normalizada = value.strip().lower()
        if normalizada not in ZONAS:
            raise ValueError(f"debe ser una de: {', '.join(ZONAS)}")
        return normalizada

    @model_validator(mode="after")
    def _validar_coherencia(self) -> "HousePredictionRequest":
        if self.metros_construidos / self.pisos > self.metros_terreno:
            raise ValueError(
                "metros_construidos dividido entre pisos no puede superar metros_terreno"
            )
        return self


# ---------------------------------------------------------------------------
# Respuestas
# ---------------------------------------------------------------------------
class RangoReferencial(BaseModel):
    minimo: float
    maximo: float


class Prediction(BaseModel):
    precio_estimado: float
    moneda: str
    rango_referencial: RangoReferencial


class ModelMeta(BaseModel):
    name: str
    version: str
    algorithm: str


class PredictionResponse(BaseModel):
    success: bool
    prediction: Prediction
    model: ModelMeta
    warnings: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    service: str
    model_loaded: bool


class CategoriaInfo(BaseModel):
    valor: str
    etiqueta: str


class VariableInfo(BaseModel):
    nombre: str
    etiqueta: str
    tipo: str
    rango: list[float] | None = None
    categorias: list[CategoriaInfo] | None = None


class ModelInfoResponse(BaseModel):
    name: str
    version: str
    alias: str
    algorithm: str
    algorithm_name: str
    run_id: str
    trained_at: str
    metrics: dict[str, float]
    currency: str
    variables: list[VariableInfo]


class ErrorDetail(BaseModel):
    campo: str
    mensaje: str


class ErrorBody(BaseModel):
    code: str
    message: str
    details: list[ErrorDetail]


class ErrorResponse(BaseModel):
    success: bool
    error: ErrorBody


def build_variables_info() -> list[VariableInfo]:
    """Describe las variables de entrada a partir del contrato (rangos y zonas)."""
    etiquetas: dict[str, str] = {
        "metros_construidos": "Metros construidos",
        "metros_terreno": "Metros de terreno",
        "habitaciones": "Habitaciones",
        "banos": "Baños",
        "antiguedad": "Antigüedad",
        "distancia_centro": "Distancia al centro",
        "garage": "Garage",
        "jardin": "Jardín",
        "pisos": "Pisos",
        "zona": "Zona",
    }
    variables: list[VariableInfo] = []
    for nombre in [
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
    ]:
        if nombre == "zona":
            variables.append(
                VariableInfo(
                    nombre=nombre,
                    etiqueta=etiquetas[nombre],
                    tipo=COLUMN_DTYPES[nombre],
                    categorias=[
                        CategoriaInfo(valor=zona, etiqueta=ZONA_LABELS[zona])
                        for zona in ZONAS
                    ],
                )
            )
        else:
            minimo, maximo = TRAINING_RANGES[nombre]
            variables.append(
                VariableInfo(
                    nombre=nombre,
                    etiqueta=etiquetas[nombre],
                    tipo=COLUMN_DTYPES[nombre],
                    rango=[minimo, maximo],
                )
            )
    return variables


__all__ = [
    "HousePredictionRequest",
    "PredictionResponse",
    "HealthResponse",
    "ModelInfoResponse",
    "ErrorResponse",
    "build_variables_info",
    "ZONA_LABELS",
]
