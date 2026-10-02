"""Generador del dataset sintético HOUSEPRICE ML.

Genera ``data/house_prices.csv`` con al menos 2000 registros (configurable con
``--n``) y un precio calculado con una fórmula determinista más ruido reproducible.

La generación es totalmente reproducible: se usa ``numpy.random.default_rng(SEED)``
y dos ejecuciones con la misma semilla producen archivos idénticos byte a byte.

Se puede ejecutar como ``python data/generate_dataset.py`` o como
``python -m data.generate_dataset`` desde cualquier directorio.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Asegura que la raíz del proyecto esté en sys.path para poder importar `shared`.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from shared.schema import (  # noqa: E402
    ALL_COLUMNS,
    INPUT_COLUMNS,
    MONEDA,
    SEED,
    TARGET,
    ZONAS,
)

# ---------------------------------------------------------------------------
# Coeficientes de la fórmula del precio (MXN)
# ---------------------------------------------------------------------------
PRECIO_BASE: float = 350_000.0
PRECIO_POR_HABITACION: float = 45_000.0
PRECIO_POR_BANO: float = 60_000.0
PRECIO_GARAGE: float = 120_000.0
PRECIO_JARDIN: float = 90_000.0
PRECIO_POR_PISO_ADICIONAL: float = 70_000.0

# Precio de construcción por m², en orden de ZONAS.
PRECIO_CONSTRUCCION_POR_ZONA: dict[str, float] = {
    "economica": 6_500.0,
    "media": 9_500.0,
    "residencial": 13_500.0,
    "premium": 21_000.0,
}

# Precio de terreno por m², en orden de ZONAS.
PRECIO_TERRENO_POR_ZONA: dict[str, float] = {
    "economica": 700.0,
    "media": 1_100.0,
    "residencial": 1_800.0,
    "premium": 3_200.0,
}

# Factores multiplicativos.
DEPRECIACION_ANTIGUEDAD: float = 0.012
PENALIZACION_DISTANCIA: float = 0.015

# Ruido log-normal.
RUIDO_SIGMA: float = 0.07

# Parámetros de las distribuciones.
ZONA_PROBABILIDADES: list[float] = [0.30, 0.35, 0.22, 0.13]
PISOS_PROBABILIDADES: list[float] = [0.55, 0.38, 0.07]
METROS_MEDIANA: float = 115.0
METROS_SIGMA: float = 0.35
METROS_FACTOR_ZONA: dict[str, float] = {
    "economica": 0.90,
    "media": 1.00,
    "residencial": 1.12,
    "premium": 1.28,
}
DISTANCIA_SHAPE: float = 1.8
DISTANCIA_SCALE: float = 6.0
DISTANCIA_FACTOR_ZONA: dict[str, float] = {
    "economica": 1.25,
    "media": 1.05,
    "residencial": 0.85,
    "premium": 0.70,
}
GARAGE_PROB_ZONA: dict[str, float] = {
    "economica": 0.25,
    "media": 0.45,
    "residencial": 0.70,
    "premium": 0.90,
}
JARDIN_PROB_ZONA: dict[str, float] = {
    "economica": 0.20,
    "media": 0.40,
    "residencial": 0.65,
    "premium": 0.85,
}


def _vector_por_zona(zona: np.ndarray, valores: dict[str, float]) -> np.ndarray:
    """Devuelve un vector con el valor del diccionario correspondiente a cada zona."""
    return np.array([valores[str(z)] for z in zona], dtype=float)


def generate_dataset(n: int = 2000, seed: int = SEED) -> pd.DataFrame:
    """Genera el dataset sintético con ``n`` registros, reproducible con ``seed``."""
    rng = np.random.default_rng(seed)

    # --- Zona -------------------------------------------------------------
    zona = rng.choice(ZONAS, size=n, p=ZONA_PROBABILIDADES)

    # --- Metros construidos (log-normal, mayores en zonas altas) -----------
    metros_construidos = rng.lognormal(
        mean=np.log(METROS_MEDIANA), sigma=METROS_SIGMA, size=n
    ) * _vector_por_zona(zona, METROS_FACTOR_ZONA)
    metros_construidos = np.clip(metros_construidos, 40.0, 500.0)

    # --- Pisos (55 % una planta, 38 % dos, 7 % tres) -----------------------
    pisos = rng.choice([1, 2, 3], size=n, p=PISOS_PROBABILIDADES).astype(int)

    # --- Habitaciones (≈ un dormitorio por cada 35 m², ±1) -----------------
    habitaciones = np.round(metros_construidos / 35.0).astype(int)
    habitaciones += rng.integers(-1, 2, size=n)
    habitaciones = np.clip(habitaciones, 1, 8).astype(int)

    # --- Baños (derivados de habitaciones, entre 1 y habitaciones + 1) -----
    banos = habitaciones + rng.integers(0, 2, size=n)
    banos = np.clip(banos, 1, 6).astype(int)

    # --- Metros de terreno (huella ≤ terreno, factor 1.1–2.5) ---------------
    huella = metros_construidos / np.maximum(pisos, 1)
    factor_terreno = rng.uniform(1.1, 2.5, size=n)
    metros_terreno = huella * factor_terreno
    metros_terreno = np.clip(metros_terreno, 60.0, 1000.0)

    # --- Antigüedad (entero 0–50) ------------------------------------------
    antiguedad = rng.integers(0, 51, size=n).astype(int)

    # --- Distancia al centro (sesgada a la derecha, depende de la zona) ----
    distancia_centro = rng.gamma(shape=DISTANCIA_SHAPE, scale=DISTANCIA_SCALE, size=n)
    distancia_centro = distancia_centro * _vector_por_zona(zona, DISTANCIA_FACTOR_ZONA) + 0.5
    distancia_centro = np.clip(distancia_centro, 0.5, 30.0)

    # --- Garage y jardín (probabilidad creciente con la zona) ---------------
    garage = rng.binomial(1, _vector_por_zona(zona, GARAGE_PROB_ZONA)).astype(int)
    jardin = rng.binomial(1, _vector_por_zona(zona, JARDIN_PROB_ZONA)).astype(int)

    # --- Fórmula del precio -------------------------------------------------
    precio = _calcular_precio(
        metros_construidos=metros_construidos,
        metros_terreno=metros_terreno,
        habitaciones=habitaciones,
        banos=banos,
        antiguedad=antiguedad,
        distancia_centro=distancia_centro,
        garage=garage,
        jardin=jardin,
        pisos=pisos,
        zona=zona,
        rng=rng,
    )

    data: dict[str, np.ndarray] = {
        "metros_construidos": metros_construidos,
        "metros_terreno": metros_terreno,
        "habitaciones": habitaciones,
        "banos": banos,
        "antiguedad": antiguedad,
        "distancia_centro": distancia_centro,
        "garage": garage,
        "jardin": jardin,
        "pisos": pisos,
        "zona": zona,
        TARGET: precio,
    }
    df = pd.DataFrame(data, columns=ALL_COLUMNS)
    return _fijar_tipos(df)


def _calcular_precio(
    *,
    metros_construidos: np.ndarray,
    metros_terreno: np.ndarray,
    habitaciones: np.ndarray,
    banos: np.ndarray,
    antiguedad: np.ndarray,
    distancia_centro: np.ndarray,
    garage: np.ndarray,
    jardin: np.ndarray,
    pisos: np.ndarray,
    zona: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Calcula el precio (MXN) a partir de las variables, con ruido reproducible."""
    precio_m2_construccion = _vector_por_zona(zona, PRECIO_CONSTRUCCION_POR_ZONA)
    precio_m2_terreno = _vector_por_zona(zona, PRECIO_TERRENO_POR_ZONA)

    subtotal = (
        PRECIO_BASE
        + metros_construidos * precio_m2_construccion
        + metros_terreno * precio_m2_terreno
        + habitaciones * PRECIO_POR_HABITACION
        + banos * PRECIO_POR_BANO
        + garage * PRECIO_GARAGE
        + jardin * PRECIO_JARDIN
        + (pisos - 1) * PRECIO_POR_PISO_ADICIONAL
    )

    depreciacion = np.exp(-DEPRECIACION_ANTIGUEDAD * antiguedad)
    penalizacion = np.exp(-PENALIZACION_DISTANCIA * distancia_centro)
    ruido = rng.lognormal(mean=0.0, sigma=RUIDO_SIGMA, size=len(subtotal))

    precio = subtotal * depreciacion * penalizacion * ruido
    # Redondeo a la centena.
    precio = np.round(precio / 100.0) * 100.0
    return np.maximum(precio, 100.0)


def _fijar_tipos(df: pd.DataFrame) -> pd.DataFrame:
    """Fija los tipos de dato exactos del contrato sobre el DataFrame."""
    from shared.schema import COLUMN_DTYPES

    for col, dtype in COLUMN_DTYPES.items():
        df[col] = df[col].astype(dtype)
    return df


def _imprimir_resumen(df: pd.DataFrame) -> None:
    """Imprime un resumen legible de la distribución del precio."""
    print("========================================")
    print("RESUMEN DISTRIBUCIÓN DE PRECIO")
    print("========================================")
    print(f"Registros generados: {len(df)}")
    print(f"Moneda: {MONEDA}")
    print("----------------------------------------")
    print(
        f"{'Zona':<12} {'Mín':>12} {'P25':>12} {'Mediana':>12} "
        f"{'P75':>12} {'Máx':>12} {'Media':>14}"
    )
    print("-" * 82)
    for zona in ZONAS:
        sub = df.loc[df["zona"] == zona, TARGET]
        print(
            f"{zona:<12} {sub.min():>12,.0f} {sub.quantile(0.25):>12,.0f} "
            f"{sub.median():>12,.0f} {sub.quantile(0.75):>12,.0f} "
            f"{sub.max():>12,.0f} {sub.mean():>14,.0f}"
        )
    print("========================================")


def main() -> None:
    """Punto de entrada: genera el CSV y lo escribe en ``data/house_prices.csv``."""
    parser = argparse.ArgumentParser(
        description="Genera el dataset sintético HOUSEPRICE ML."
    )
    parser.add_argument("--n", type=int, default=2000, help="Número de registros.")
    parser.add_argument("--seed", type=int, default=SEED, help="Semilla aleatoria.")
    args = parser.parse_args()

    if args.n < 1:
        parser.error("--n debe ser al menos 1.")

    df = generate_dataset(n=args.n, seed=args.seed)

    output_path = ROOT_DIR / "data" / "house_prices.csv"
    df.to_csv(output_path, index=False, encoding="utf-8", sep=",")
    print(f"Dataset escrito en: {output_path}")
    _imprimir_resumen(df)


if __name__ == "__main__":
    main()
