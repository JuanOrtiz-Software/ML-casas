"""Capa fina de configuración sobre ``shared/settings.py``.

La API no redefine la configuración: solo reexporta los valores centralizados para
que el resto del backend los importe desde un único lugar.
"""

from shared.settings import (  # noqa: F401
    BACKEND_HOST,
    BACKEND_PORT,
    FRONTEND_URL,
    LOG_LEVEL,
    MLFLOW_MODEL_ALIAS,
    MLFLOW_MODEL_NAME,
    MLFLOW_TRACKING_URI,
)
