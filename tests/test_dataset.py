"""Pruebas del dataset: generación reproducible, contrato y validación."""

from __future__ import annotations

import pandas as pd
import pytest

from data.generate_dataset import generate_dataset
from data.validate_dataset import validate_dataframe, validate_dataset
from shared.schema import ALL_COLUMNS, COLUMN_DTYPES, ZONAS


def test_generador_es_reproducible() -> None:
    """Dos ejecuciones con la misma semilla producen exactamente los mismos datos."""
    df1 = generate_dataset(n=100, seed=42)
    df2 = generate_dataset(n=100, seed=42)
    assert df1.to_csv(index=False) == df2.to_csv(index=False)


def test_csv_tiene_columnas_y_tipos_esperados() -> None:
    """El dataset generado cumple el contrato de columnas y tipos."""
    df = generate_dataset(n=300, seed=7)
    assert list(df.columns) == ALL_COLUMNS
    for col, dtype in COLUMN_DTYPES.items():
        if col == "zona":
            assert pd.api.types.is_string_dtype(df[col].dtype) or pd.api.types.is_object_dtype(
                df[col].dtype
            )
        else:
            assert str(df[col].dtype) == dtype, f"{col}: {df[col].dtype} != {dtype}"


def test_reglas_de_coherencia() -> None:
    """Se cumplen las reglas de coherencia terreno/huella y baños/habitaciones."""
    df = generate_dataset(n=500, seed=11)
    assert (df["metros_terreno"] >= df["metros_construidos"] / df["pisos"]).all()
    assert (df["banos"] <= df["habitaciones"] + 1).all()
    assert df["zona"].isin(ZONAS).all()


def test_validador_acepta_dataset_valido() -> None:
    """El validador marca como válido un dataset generado correctamente."""
    df = generate_dataset(n=300, seed=3)
    result = validate_dataframe(df, min_rows=300)
    assert result.is_valid is True
    assert result.errors == []


def _dataframe_corrupto() -> pd.DataFrame:
    df = generate_dataset(n=300, seed=5)
    return df


def test_validador_detecta_nulos(tmp_path) -> None:
    df = _dataframe_corrupto()
    df.loc[0, "metros_construidos"] = None
    result = validate_dataframe(df, min_rows=300)
    assert result.is_valid is False
    assert any("nulos" in error for error in result.errors)


def test_validador_detecta_negativos(tmp_path) -> None:
    df = _dataframe_corrupto()
    df.loc[0, "antiguedad"] = -1
    result = validate_dataframe(df, min_rows=300)
    assert result.is_valid is False
    assert any("negativos" in error for error in result.errors)


def test_validador_detecta_zona_invalida(tmp_path) -> None:
    df = _dataframe_corrupto()
    df.loc[0, "zona"] = "lujo"
    result = validate_dataframe(df, min_rows=300)
    assert result.is_valid is False
    assert any("zona" in error for error in result.errors)


def test_validador_archivo_corrupto_devuelve_invalido(tmp_path) -> None:
    """Con un CSV corrupto, validate_dataset devuelve un resultado no válido."""
    df = generate_dataset(n=300, seed=5)
    df.loc[0, "precio"] = -500.0
    path = tmp_path / "corrupto.csv"
    df.to_csv(path, index=False)
    result = validate_dataset(path)
    assert result.is_valid is False
    assert result.errors != []


def test_validador_archivo_inexistente(tmp_path) -> None:
    """Un archivo inexistente se reporta como error."""
    result = validate_dataset(tmp_path / "no_existe.csv")
    assert result.is_valid is False
    assert any("no existe" in error for error in result.errors)
