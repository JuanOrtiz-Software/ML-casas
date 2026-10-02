"""Contrato compartido del dataset HOUSEPRICE ML.

Este módulo es la fuente única de verdad de:

* el nombre de la variable objetivo y de las diez variables de entrada (en orden);
* qué columnas son numéricas, cuáles binarias y cuál es la categórica;
* las categorías válidas de ``zona`` y su etiqueta legible para la interfaz;
* el tipo de dato de cada columna;
* los rangos de entrenamiento de cada variable;
* la semilla global, el tamaño del conjunto de prueba y la moneda.

Cualquier lista de columnas, rango o categoría se define aquí **una sola vez** y el
resto del proyecto (generador, validador, entrenamiento, API y pruebas) la importa.
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Variable objetivo
# ---------------------------------------------------------------------------
TARGET: Final[str] = "precio"

# ---------------------------------------------------------------------------
# Variables de entrada (ordenadas, tal como aparecen en el CSV y en el formulario)
# ---------------------------------------------------------------------------
INPUT_COLUMNS: Final[list[str]] = [
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

# Todas las columnas del CSV (entradas + objetivo).
ALL_COLUMNS: Final[list[str]] = INPUT_COLUMNS + [TARGET]

# ---------------------------------------------------------------------------
# Clasificación por tipo
# ---------------------------------------------------------------------------
FLOAT_COLUMNS: Final[list[str]] = [
    "metros_construidos",
    "metros_terreno",
    "distancia_centro",
]
INT_COLUMNS: Final[list[str]] = [
    "habitaciones",
    "banos",
    "antiguedad",
    "garage",
    "jardin",
    "pisos",
]

# Columnas numéricas (continuas y de conteo). `zona` es la única categórica.
NUMERIC_COLUMNS: Final[list[str]] = FLOAT_COLUMNS + INT_COLUMNS

# Columnas que pasan por StandardScaler en el preprocesamiento.
SCALED_COLUMNS: Final[list[str]] = [
    "metros_construidos",
    "metros_terreno",
    "habitaciones",
    "banos",
    "antiguedad",
    "distancia_centro",
    "pisos",
]

# Columnas binarias {0, 1} que pasan sin transformar.
BINARY_COLUMNS: Final[list[str]] = ["garage", "jardin"]

# Columna categórica.
CATEGORICAL_COLUMN: Final[str] = "zona"

# ---------------------------------------------------------------------------
# Categorías de `zona` (ordenadas de menor a mayor poder adquisitivo)
# ---------------------------------------------------------------------------
ZONAS: Final[list[str]] = ["economica", "media", "residencial", "premium"]

ZONA_LABELS: Final[dict[str, str]] = {
    "economica": "Económica",
    "media": "Media",
    "residencial": "Residencial",
    "premium": "Premium",
}

# ---------------------------------------------------------------------------
# Tipos de dato (numpy) de cada columna
# ---------------------------------------------------------------------------
COLUMN_DTYPES: Final[dict[str, str]] = {
    "metros_construidos": "float64",
    "metros_terreno": "float64",
    "distancia_centro": "float64",
    "habitaciones": "int64",
    "banos": "int64",
    "antiguedad": "int64",
    "garage": "int64",
    "jardin": "int64",
    "pisos": "int64",
    "zona": "object",
    "precio": "float64",
}

# ---------------------------------------------------------------------------
# Rangos de entrenamiento (mínimo, máximo) de cada variable de entrada
# ---------------------------------------------------------------------------
TRAINING_RANGES: Final[dict[str, tuple[float, float]]] = {
    "metros_construidos": (40.0, 500.0),
    "metros_terreno": (60.0, 1000.0),
    "habitaciones": (1.0, 8.0),
    "banos": (1.0, 6.0),
    "antiguedad": (0.0, 50.0),
    "distancia_centro": (0.5, 30.0),
    "garage": (0.0, 1.0),
    "jardin": (0.0, 1.0),
    "pisos": (1.0, 3.0),
}

# ---------------------------------------------------------------------------
# Constantes globales
# ---------------------------------------------------------------------------
SEED: Final[int] = 42
TEST_SIZE: Final[float] = 0.20
MONEDA: Final[str] = "MXN"

# Umbral usado para definir "empate" en la regla de selección.
EPSILON: Final[float] = 1e-9

__all__ = [
    "TARGET",
    "INPUT_COLUMNS",
    "ALL_COLUMNS",
    "FLOAT_COLUMNS",
    "INT_COLUMNS",
    "NUMERIC_COLUMNS",
    "SCALED_COLUMNS",
    "BINARY_COLUMNS",
    "CATEGORICAL_COLUMN",
    "ZONAS",
    "ZONA_LABELS",
    "COLUMN_DTYPES",
    "TRAINING_RANGES",
    "SEED",
    "TEST_SIZE",
    "MONEDA",
    "EPSILON",
]
