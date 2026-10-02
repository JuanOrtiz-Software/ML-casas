# HOUSEPRICE ML

Aplicación MLOps completa que predice el precio de una vivienda (regresión
supervisada) en pesos mexicanos (MXN). El flujo completo es:

**Dataset sintético → CSV → validación → preprocesamiento → entrenamiento de 5
modelos → MLflow Tracking → comparación de métricas → selección automática →
Model Registry → FastAPI → frontend Glassmorphism → predicción.**

---

## 1. Descripción y objetivo

HOUSEPRICE ML demuestra un flujo MLOps de extremo a extremo, no un simple
notebook de Machine Learning. Genera un dataset sintético reproducible, valida el
contrato de datos, entrena cinco modelos de regresión, registra cada uno como un
*run* de MLflow, selecciona automáticamente el campeón con una regla determinista,
lo publica en el Model Registry bajo el alias `champion` y lo sirve a través de una
API FastAPI consumida por un frontend Next.js con diseño Glassmorphism.

La arquitectura de ejecución final es:

```
Usuario → Next.js → FastAPI → modelo cargado desde MLflow Registry → predicción → FastAPI → Next.js → Usuario
```

---

## 2. Arquitectura

```
                          ┌─────────────────────────────┐
                          │         Usuario             │
                          └──────────────┬──────────────┘
                                         │
                                         ▼
                    ┌─────────────────────────────────────┐
                    │        Next.js (Glassmorphism)      │
                    │  formulario · resultado · estado API │
                    └───────────┬──────────────┬──────────┘
                     GET /health │              │ POST /predict
                     GET /model-info             │
                                 ▼              ▼
                    ┌─────────────────────────────────────┐
                    │           FastAPI (backend)          │
                    │  schemas → predictor → model_loader  │
                    └──────────────────┬──────────────────┘
                                       │ carga al arrancar
                                       ▼
                    ┌─────────────────────────────────────┐
                    │   MLflow Model Registry (champion)  │
                    │   Pipeline sklearn (preprocesamiento │
                    │   + modelo) con firma y ejemplo      │
                    └─────────────────────────────────────┘
```

Separación en capas:

- **`shared/`** — fuente única de verdad: contrato de datos (`schema.py`) y
  configuración (`settings.py`).
- **`data/`** — generación y validación del dataset sintético.
- **`ml/`** — preprocesamiento, modelos, evaluación, registro y entrenamiento.
- **`backend/`** — API FastAPI.
- **`frontend/`** — aplicación Next.js.
- **`tests/`** — pruebas con pytest que no tocan el `mlflow.db` real.

---

## 3. Dataset y variables

El dataset es sintético (no se descarga nada externo) y reproducible con la semilla
`SEED = 42`. Se genera en `data/house_prices.csv` con 2000 registros por defecto.

| Variable | Tipo | Rango de entrenamiento | Descripción |
|---|---|---|---|
| `metros_construidos` | float64 | 40 – 500 | Superficie construida (m²) |
| `metros_terreno` | float64 | 60 – 1000 | Superficie del terreno (m²) |
| `habitaciones` | int64 | 1 – 8 | Número de habitaciones |
| `banos` | int64 | 1 – 6 | Número de baños |
| `antiguedad` | int64 | 0 – 50 | Años de antigüedad |
| `distancia_centro` | float64 | 0.5 – 30 | Distancia al centro (km) |
| `garage` | int64 | {0, 1} | ¿Tiene garage? |
| `jardin` | int64 | {0, 1} | ¿Tiene jardín? |
| `pisos` | int64 | 1 – 3 | Número de plantas |
| `zona` | categórica | económica, media, residencial, premium | Zona de la vivienda |
| `precio` (objetivo) | float64 | — | Precio en MXN |

Las variables no se muestrean de forma independiente: la generación respeta reglas
de coherencia (por ejemplo `metros_terreno ≥ metros_construidos / pisos` y
`banos ≤ habitaciones + 1`), distribuciones realistas (metros construidos log-normal,
distancia sesgada a la derecha, probabilidad de garage/jardín creciente con la zona) y
una fórmula de precio determinista más ruido log-normal moderado. El precio medio por
zona crece en el orden económica < media < residencial < premium.

---

## 4. Generación y validación

```bash
python data/generate_dataset.py      # genera data/house_prices.csv
python data/validate_dataset.py      # valida el contrato
```

El validador comprueba: existencia del archivo, número de filas, columnas y orden,
tipos, ausencia de nulos/duplicados/negativos, rangos, categorías de `zona`, precio
positivo, reglas de coherencia y jerarquía del precio medio por zona. Termina con
código 0 si el dataset es válido y distinto de cero en caso contrario.

Salida real de la validación:

```
========================================
VALIDACIÓN DATASET
========================================
Registros: 2000
Columnas: 11
Valores nulos: 0
Duplicados: 0
Registros inválidos: 0
Dataset válido: SÍ
========================================
```

---

## 5. Modelos

Se entrenan exactamente cinco modelos de scikit-learn, cada uno en su propio *run*
de MLflow, con un único `Pipeline` (`preprocesamiento` + `modelo`):

| Nombre legible | Clase | Hiperparámetros de partida |
|---|---|---|
| Linear Regression | `LinearRegression` | `fit_intercept=True` |
| Decision Tree | `DecisionTreeRegressor` | `max_depth=8, min_samples_split=10, min_samples_leaf=4` |
| Random Forest | `RandomForestRegressor` | `n_estimators=300, max_depth=14, ...` |
| Gradient Boosting | `GradientBoostingRegressor` | `n_estimators=300, learning_rate=0.05, max_depth=3` |
| Extra Trees | `ExtraTreesRegressor` | `n_estimators=300, max_depth=16, ...` |

Todos con `random_state=42` salvo la regresión lineal. No se hace búsqueda de
hiperparámetros: el objetivo del proyecto es el flujo MLOps.

El preprocesador es un `ColumnTransformer`: `StandardScaler` para las numéricas,
`passthrough` para `garage`/`jardin` y `OneHotEncoder(categories=[ZONAS],
handle_unknown="error")` para `zona`.

---

## 6. Métricas (tabla real de la última ejecución)

Métricas sobre el conjunto de prueba (400 registros), más la validación cruzada
(5 particiones sobre el entrenamiento):

```
MODELO                       MAE           MSE        RMSE        R2     CV_R2
-----------------------------------------------------------------------------
Linear Regression        213,442  96,366,851,570     310,430   0.9012    0.8952
Decision Tree            214,849 102,658,876,695     320,404   0.8947    0.8920
Random Forest            140,555  44,284,233,264     210,438   0.9546    0.9415
Gradient Boosting        122,739  33,036,529,746     181,760   0.9661    0.9584
Extra Trees              135,883  37,693,265,841     194,148   0.9613    0.9475
```

**MEJOR MODELO: Gradient Boosting** (criterio: CV R² descendente, CV RMSE
ascendente). Modelo registrado: `HousePriceModel`, versión 2, alias `champion`.

Los modelos de árboles superan claramente a la regresión lineal (la relación no es
lineal), sin que ninguno sea trivialmente perfecto.

### Regla de selección

La selección usa la **validación cruzada del conjunto de entrenamiento** (no el
conjunto de prueba) para no sesgar la elección. Se ordena por `cv_r2_mean`
descendente; en caso de empate (diferencia < 1e-9), por `cv_rmse_mean` ascendente;
si persiste, por el orden fijo de `build_models()`. El conjunto de prueba queda como
estimación final e imparcial del modelo elegido.

---

## 7. MLflow y Model Registry

- **Experimento:** `HousePricePrediction`.
- Cada ejecución de `ml/train.py` crea **exactamente cinco runs** (uno por modelo),
  con `run_name` legible, y comparte la etiqueta `training_session` (marca UTC).
- Por run se registran: parámetros (`algorithm`, `model_name`, `random_state`,
  `test_size`, `train_size`, `n_train`, `n_test`, `n_features` y los hiperparámetros
  propios), métricas (`mae`, `mse`, `rmse`, `r2`, `r2_train`, `mape`, `fit_seconds`,
  `cv_r2_mean`, `cv_rmse_mean`, `cv_rmse_std`), etiquetas (`project`, `problem_type`,
  `dataset`, `dataset_sha256`, `training_session`, `sklearn_version`,
  `is_champion`), el `Pipeline` completo con **firma** y **ejemplo de entrada**, y el
  artifact `feature_importance.png` (importancia por permutación).
- Solo se registra el **ganador** en el Registry con el alias `champion`. Cada sesión
  añade una versión nueva y mueve el alias; las versiones anteriores se conservan.
- En el run ganador se guardan además `model_comparison.csv`, `metrics.json` y
  `dataset_summary.json`.

### Abrir la interfaz de MLflow

Con una base de datos SQLite **no** basta `mlflow ui` a secas (leería `./mlruns` y
mostraría un panel vacío). El comando correcto es:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

(o `export MLFLOW_TRACKING_URI=sqlite:///mlflow.db` antes de `mlflow ui`). Desde la
interfaz se puede abrir el experimento `HousePricePrediction`, comparar los cinco
runs, ver parámetros/métricas/artifacts y localizar el modelo `HousePriceModel` con
el alias `champion`.

---

## 8. Backend (FastAPI) y endpoints

La API carga el modelo **una sola vez al arrancar** (desde
`models:/HousePriceModel@champion`) y no entrena nunca. Si el modelo no puede
cargarse, arranca en estado degradado (no se cae).

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/health` | Estado del servicio (200 con modelo, 503 degradado) |
| POST | `/predict` | Predicción de precio a partir de datos crudos |
| GET | `/model-info` | Metadatos del campeón y descripción de variables |
| GET | `/docs` | Swagger UI automático |

### Ejemplos con cURL

```bash
curl http://localhost:8000/health

curl http://localhost:8000/model-info

curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "metros_construidos": 120, "metros_terreno": 200, "habitaciones": 3,
    "banos": 2, "antiguedad": 5, "distancia_centro": 4.2,
    "garage": 1, "jardin": 1, "pisos": 2, "zona": "residencial"
  }'
```

Respuesta real (el precio lo produce el modelo registrado en MLflow):

```json
{
  "success": true,
  "prediction": {
    "precio_estimado": 2327132.22,
    "moneda": "MXN",
    "rango_referencial": { "minimo": 2145372.68, "maximo": 2508891.76 }
  },
  "model": { "name": "HousePriceModel", "version": "2", "algorithm": "GradientBoostingRegressor" },
  "warnings": []
}
```

### Validación

- **Límites físicos duros** (422): `metros_construidos ∈ (0, 2000]`,
  `metros_terreno ∈ (0, 10000]`, `habitaciones ∈ [1, 20]`, `banos ∈ [1, 15]`,
  `antiguedad ∈ [0, 150]`, `distancia_centro ∈ [0, 200]`, `garage`/`jardin ∈ {0,1}`,
  `pisos ∈ [1, 10]`, `zona` válida (normalizada a minúsculas) y la coherencia
  `metros_construidos / pisos ≤ metros_terreno`.
- **Rangos de entrenamiento** (más estrechos): no rechazan; generan `warnings`.
- Tipos estrictos: un texto donde se espera un número es un error.
- Errores uniformes: `{ "success": false, "error": { "code", "message", "details" } }`
  con códigos `VALIDATION_ERROR` (422), `MODEL_UNAVAILABLE` (503) e `INTERNAL_ERROR` (500).

---

## 9. Frontend y Glassmorphism

Next.js (App Router) + React + TypeScript estricto + Tailwind CSS 4 + Lucide React +
Framer Motion, en `frontend/`. El frontend **no** contiene el modelo, no entrena y no
calcula precios: todo lo muestra desde la API.

Componentes: `Header`, `GlassCard`, `PredictionForm`, `PredictionResult`,
`ModelInfo` y `ApiStatus`. La capa `lib/api.ts` centraliza el único cliente `fetch`
(con `AbortController` y traducción de errores a español). Las zonas y los rangos de
los campos salen de `GET /model-info`.

Estados del flujo de predicción (unión discriminada): `idle`, `loading`, `success` y
`error`. El indicador de estado consulta `/health` periódicamente y muestra
**● API ONLINE**, **● API ONLINE · SIN MODELO** o **● API OFFLINE**.

Diseño Glassmorphism accesible: fondo oscuro con degradado, luces difusas animadas,
tarjetas translúcidas (`backdrop-blur` + borde semitransparente + brillo interior),
alternativa opaca con `@supports` cuando no hay `backdrop-filter`, respeto de
`prefers-reduced-motion`, foco visible, etiquetas asociadas, `aria-live` y diseño
adaptativo.

`NEXT_PUBLIC_API_URL` se incorpora en **tiempo de compilación** (ver `.env.example`).

---

## 10. Instalación y ejecución

### Requisitos

- Python 3.12 (proyecto probado con 3.12.3)
- Node.js (proyecto probado con v24.19.0) y npm

### Instalación

```bash
cp .env.example .env        # opcional: los valores por defecto ya funcionan
./scripts/setup.sh          # crea .venv e instala dependencias de Python y Node
```

O, paso a paso:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cd frontend && npm install && cd ..
```

### Orden de ejecución

```bash
python data/generate_dataset.py
python data/validate_dataset.py
python ml/train.py
mlflow ui --backend-store-uri sqlite:///mlflow.db
uvicorn backend.app.main:app --reload
cd frontend && npm run dev
```

> **Nota:** usa `.venv/bin/python` en lugar de `python` si quieres usar el entorno
> virtual del proyecto.

Abre `http://localhost:3000` (frontend) y `http://localhost:8000/docs` (Swagger).

### Pruebas

```bash
.venv/bin/python -m pytest tests/ -q
```

Las pruebas usan un almacén SQLite temporal y **no** modifican el `mlflow.db` real.

### Scripts

| Script | Acción |
|---|---|
| `scripts/setup.sh` | Crea `.venv` e instala dependencias de Python y Node |
| `scripts/generate_data.sh` | Genera y valida el dataset |
| `scripts/train.sh` | Ejecuta el entrenamiento completo |
| `scripts/test_api.sh` | Comprueba los endpoints con `curl` (códigos HTTP) |

### Comandos equivalentes en Windows (PowerShell)

```powershell
python data\generate_dataset.py
python data\validate_dataset.py
python ml\train.py
mlflow ui --backend-store-uri sqlite:///mlflow.db
uvicorn backend.app.main:app --reload
cd frontend; npm install; npm run dev
$env:API_URL="http://localhost:8000"; .\scripts\test_api.sh   # o usa curl manualmente
```

---

## 11. Estructura de carpetas

```
houseprice-ml/
├── shared/                  # contrato de datos y configuración
│   ├── schema.py
│   └── settings.py
├── data/
│   ├── generate_dataset.py
│   ├── validate_dataset.py
│   └── house_prices.csv
├── ml/
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluate.py
│   ├── registry.py
│   └── train.py
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── schemas.py
│   │   ├── predictor.py
│   │   ├── model_loader.py
│   │   └── config.py
│   └── requirements.txt
├── frontend/
│   ├── app/                 # page.tsx, layout.tsx, globals.css
│   ├── components/
│   ├── lib/                 # api.ts, types.ts, format.ts
│   └── public/
├── scripts/
├── tests/
├── mlruns/                  # artifacts de MLflow (regenerable)
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 12. Flujo completo

1. Generar el dataset sintético reproducible.
2. Validar el contrato de datos.
3. Entrenar los cinco modelos y registrarlos en MLflow.
4. Comparar métricas y seleccionar el campeón con la regla determinista.
5. Registrar el campeón en el Model Registry y asignar el alias `champion`.
6. Levantar FastAPI, que carga el campeón desde el Registry.
7. Levantar Next.js, que consulta `/health` y `/model-info` y envía `/predict`.

---

## 13. Solución de problemas

| Síntoma | Causa probable | Solución |
|---|---|---|
| La API carga otro modelo o no encuentra runs | `MLFLOW_TRACKING_URI` apunta a otra base | Revisa `.env`; usa `sqlite:///mlflow.db` (anclada a la raíz) |
| `mlflow ui` muestra un panel vacío | Falta `--backend-store-uri` | `mlflow ui --backend-store-uri sqlite:///mlflow.db` |
| El frontend no conecta (CORS) | `FRONTEND_URL` no incluye el origen | Añade el origen en `FRONTEND_URL` (separado por coma) |
| Error al cargar el modelo en la API | scikit-learn distinto entre entrenamiento y API | Usa el mismo entorno (mismo `requirements.txt`) |
| `uvicorn` falla al arrancar | Puerto 8000 ocupado | Cambia `BACKEND_PORT` |
| `/predict` devuelve 503 | El modelo aún no se ha entrenado | Ejecuta `python ml/train.py` primero |

---

## 14. Decisiones y supuestos

- **Moneda:** MXN, definida como constante en el contrato (`shared/schema.py`).
- **Regla de selección:** por validación cruzada del entrenamiento (no por el
  conjunto de prueba) para evitar el sesgo de selección.
- **Serialización del modelo:** se usa `serialization_format="cloudpickle"` en lugar
  del `skops` por defecto de MLflow 3.x, porque `skops` rechaza los modelos basados
  en árboles (`sklearn.tree._tree.Tree` no es un tipo confiable). Es una decisión
  técnica documentada; los cinco modelos usan el mismo formato.
- **Versiones detectadas:** Python 3.12.3, MLflow 3.16.1, scikit-learn 1.9.1,
  pandas 3.0.6, numpy 2.5.3, FastAPI 0.142.2, Pydantic 2.13.5, Node v24.19.0,
  Next.js 16.3.8, Tailwind 4.3.3, React 19.3.0.
- **MLflow 3.x:** se usa `name=` en `log_model` y **aliases** en lugar de los *stages*
  (deprecados). La API de MLflow 2.x (con `artifact_path=` y stages) no se usa.
- **`zona` en pandas 3.x:** pandas 3 lee las columnas de texto como dtype `str` (no
  `object`); el validador acepta ambos para la columna categórica.
- **Validación del cliente:** el formulario valida contra los rangos de entrenamiento
  que devuelve `/model-info` (los límites físicos duros, más amplios, los aplica el
  servidor).
- **Entorno limpio:** no existía `mlflow.db` ni `mlflow_example.py` previos, por lo
  que no fue necesaria la copia `mlflow.db.bak` (sección 11.1 de la especificación).

## 15. Limitaciones

- **Datos sintéticos:** el modelo aprende la fórmula que generó los datos; su R²
  describe esa fórmula, no el mercado inmobiliario real.
- **Evaluación final:** la selección usa validación cruzada, pero la estimación final
  se hace con un único conjunto de prueba de 400 registros.

## 16. Trabajo futuro

Fuera de alcance (no implementado): Docker, autenticación, CI/CD, despliegue en la
nube, almacenamiento de predicciones, recarga en caliente del modelo y
reentrenamiento programado.

---

## 17. Guía de demostración (~5 minutos)

1. **Datos:** muestra `data/house_prices.csv` y ejecuta
   `python data/validate_dataset.py` (resumen "Dataset válido: SÍ").
2. **Entrenamiento:** ejecuta `python ml/train.py` y muestra la tabla comparativa y
   el ganador.
3. **MLflow:** abre `mlflow ui --backend-store-uri sqlite:///mlflow.db`, entra al
   experimento `HousePricePrediction`, compara los cinco runs (parámetros, métricas,
   artifacts) y muestra `feature_importance.png`.
4. **Registry:** en la pestaña "Models" muestra `HousePriceModel` con el alias
   `champion` apuntando a la última versión.
5. **API:** `curl http://localhost:8000/health`, `curl http://localhost:8000/model-info`
   y `curl -X POST .../predict` (o desde `http://localhost:8000/docs`).
6. **Frontend:** abre `http://localhost:3000`, pulsa **PREDECIR PRECIO** con los
   valores del ejemplo y destaca que el algoritmo y las métricas mostrados provienen
   del Registry (no están escritos a mano en el frontend).
