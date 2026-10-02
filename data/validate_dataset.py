"""Validador del dataset HOUSEPRICE ML.

Comprueba que ``data/house_prices.csv`` cumple el contrato definido en
``shared/schema.py``: columnas y orden, tipos, ausencia de nulos/duplicados/negativos,
rangos, categorías de ``zona``, precio positivo, reglas de coherencia y la jerarquía
del precio medio por zona.

La lógica de validación vive en funciones reutilizables (``validate_dataframe`` y
``validate_dataset``) para que las pruebas y el entrenamiento puedan llamarla sin
duplicarla. Se puede ejecutar como ``python data/validate_dataset.py`` o como
``python -m data.validate_dataset``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from shared.schema import (  # noqa: E402
    ALL_COLUMNS,
    COLUMN_DTYPES,
    NUMERIC_COLUMNS,
    TARGET,
    TRAINING_RANGES,
    ZONAS,
)

MIN_ROWS: int = 2000
DATASET_PATH: Path = ROOT_DIR / "data" / "house_prices.csv"


@dataclass
class ValidationResult:
    """Resultado de una validación, con errores críticos y estadísticos del resumen."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    n_rows: int = 0
    n_columns: int = 0
    n_nulls: int = 0
    n_duplicates: int = 0
    n_invalid_rows: int = 0


def validate_dataframe(df: pd.DataFrame, min_rows: int = MIN_ROWS) -> ValidationResult:
    """Valida un DataFrame contra el contrato y devuelve el resultado estructurado."""
    errors: list[str] = []
    bad_mask = pd.Series(False, index=df.index)

    n_rows = len(df)
    n_columns = len(df.columns)

    # 1. Número de filas.
    if n_rows < min_rows:
        errors.append(
            f"El dataset tiene {n_rows} registros; se esperan al menos {min_rows}."
        )

    # 2. Columnas exactas y en el orden esperado.
    if list(df.columns) != ALL_COLUMNS:
        errors.append(
            f"Columnas incorrectas. Esperadas {ALL_COLUMNS}; obtenidas {list(df.columns)}."
        )

    # 3. Tipos de dato. `zona` es categórica y acepta cualquier dtype de cadena
    # (pandas 3.x lee las columnas de texto como `str`, no como `object`).
    for col, expected_dtype in COLUMN_DTYPES.items():
        if col not in df.columns:
            continue
        if col == "zona":
            if not pd.api.types.is_string_dtype(df[col].dtype) and not pd.api.types.is_object_dtype(
                df[col].dtype
            ):
                errors.append(
                    f"Columna 'zona' tiene tipo {df[col].dtype}; se espera texto/categórica."
                )
        elif str(df[col].dtype) != expected_dtype:
            errors.append(
                f"Columna '{col}' tiene tipo {df[col].dtype}; se espera {expected_dtype}."
            )

    # 4. Valores nulos.
    n_nulls = int(df.isna().sum().sum())
    if n_nulls:
        cols_nulls = df.columns[df.isna().any()].tolist()
        errors.append(f"Hay {n_nulls} valores nulos en las columnas {cols_nulls}.")
        bad_mask |= df.isna().any(axis=1)

    # 5. Duplicados.
    n_duplicates = int(df.duplicated().sum())
    if n_duplicates:
        errors.append(f"Hay {n_duplicates} registros duplicados.")

    # 6. Valores negativos en las variables numéricas.
    for col in NUMERIC_COLUMNS:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            neg = df[col] < 0
            if neg.any():
                errors.append(
                    f"La columna '{col}' tiene {int(neg.sum())} valores negativos."
                )
                bad_mask |= neg

    # 7. Valores dentro de rango.
    for col, (lo, hi) in TRAINING_RANGES.items():
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            fuera = (df[col] < lo) | (df[col] > hi)
            if fuera.any():
                errors.append(
                    f"La columna '{col}' tiene {int(fuera.sum())} valores fuera del "
                    f"rango [{lo}, {hi}]."
                )
                bad_mask |= fuera

    # 8. Categorías válidas de zona.
    if "zona" in df.columns:
        zona_invalida = ~df["zona"].isin(ZONAS)
        if zona_invalida.any():
            errors.append(
                f"La columna 'zona' tiene {int(zona_invalida.sum())} categorías "
                f"inválidas: {sorted(set(df.loc[zona_invalida, 'zona'].astype(str)))}."
            )
            bad_mask |= zona_invalida

    # 9. Precio positivo.
    if TARGET in df.columns and pd.api.types.is_numeric_dtype(df[TARGET]):
        precio_no_positivo = df[TARGET] <= 0
        if precio_no_positivo.any():
            errors.append(
                f"La columna '{TARGET}' tiene {int(precio_no_positivo.sum())} valores "
                "no positivos."
            )
            bad_mask |= precio_no_positivo

    # 10. Coherencia: huella de la casa no excede el terreno.
    if {"metros_terreno", "metros_construidos", "pisos"} <= set(df.columns):
        incoherente_terreno = df["metros_terreno"] < (df["metros_construidos"] / df["pisos"])
        if incoherente_terreno.any():
            errors.append(
                f"Hay {int(incoherente_terreno.sum())} registros donde "
                "metros_terreno < metros_construidos / pisos."
            )
            bad_mask |= incoherente_terreno

    # 11. Coherencia: baños no exceden habitaciones + 1.
    if {"banos", "habitaciones"} <= set(df.columns):
        incoherente_banos = df["banos"] > (df["habitaciones"] + 1)
        if incoherente_banos.any():
            errors.append(
                f"Hay {int(incoherente_banos.sum())} registros donde "
                "banos > habitaciones + 1."
            )
            bad_mask |= incoherente_banos

    # 12. Jerarquía del precio medio por zona.
    if {"zona", TARGET} <= set(df.columns):
        means = df.groupby("zona")[TARGET].mean()
        medios = [means[z] for z in ZONAS if z in means.index]
        if len(medios) == len(ZONAS) and not all(
            medios[i] < medios[i + 1] for i in range(len(medios) - 1)
        ):
            errors.append(
                "El precio medio por zona no crece en el orden "
                "económica < media < residencial < premium."
            )

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        n_rows=n_rows,
        n_columns=n_columns,
        n_nulls=n_nulls,
        n_duplicates=n_duplicates,
        n_invalid_rows=int(bad_mask.sum()),
    )


def load_dataset(path: Path) -> pd.DataFrame:
    """Lee el CSV y devuelve el DataFrame, o lanza ``FileNotFoundError``."""
    return pd.read_csv(path, encoding="utf-8", sep=",")


def validate_dataset(path: Path = DATASET_PATH) -> ValidationResult:
    """Valida el CSV en ``path``. Devuelve el resultado aunque el archivo no exista."""
    if not path.exists():
        return ValidationResult(
            is_valid=False,
            errors=[f"El archivo {path} no existe."],
        )
    df = load_dataset(path)
    return validate_dataframe(df)


def _print_summary(result: ValidationResult) -> None:
    """Imprime el resumen legible de la validación."""
    print("========================================")
    print("VALIDACIÓN DATASET")
    print("========================================")
    print(f"Registros: {result.n_rows}")
    print(f"Columnas: {result.n_columns}")
    print(f"Valores nulos: {result.n_nulls}")
    print(f"Duplicados: {result.n_duplicates}")
    print(f"Registros inválidos: {result.n_invalid_rows}")
    print(f"Dataset válido: {'SÍ' if result.is_valid else 'NO'}")
    print("========================================")
    if result.errors:
        print("\nErrores críticos:")
        for i, error in enumerate(result.errors, start=1):
            print(f"  {i}. {error}")
        print("========================================")


def main() -> int:
    """Punto de entrada. Devuelve 0 si el dataset es válido y distinto de cero si no."""
    result = validate_dataset(DATASET_PATH)
    _print_summary(result)
    return 0 if result.is_valid else 1


if __name__ == "__main__":
    sys.exit(main())
