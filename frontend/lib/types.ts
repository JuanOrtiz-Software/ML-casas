/** Tipos de las peticiones y respuestas de la API (espejo del backend). */

export interface HousePredictionRequest {
  metros_construidos: number;
  metros_terreno: number;
  habitaciones: number;
  banos: number;
  antiguedad: number;
  distancia_centro: number;
  garage: number;
  jardin: number;
  pisos: number;
  zona: string;
}

export interface RangoReferencial {
  minimo: number;
  maximo: number;
}

export interface Prediction {
  precio_estimado: number;
  moneda: string;
  rango_referencial: RangoReferencial;
}

export interface ModelMeta {
  name: string;
  version: string;
  algorithm: string;
}

export interface PredictionResponse {
  success: boolean;
  prediction: Prediction;
  model: ModelMeta;
  warnings: string[];
}

export interface HealthResponse {
  status: string;
  service: string;
  model_loaded: boolean;
}

export interface CategoriaInfo {
  valor: string;
  etiqueta: string;
}

export interface VariableInfo {
  nombre: string;
  etiqueta: string;
  tipo: string;
  rango: number[] | null;
  categorias: CategoriaInfo[] | null;
}

export interface ModelInfoResponse {
  name: string;
  version: string;
  alias: string;
  algorithm: string;
  algorithm_name: string;
  run_id: string;
  trained_at: string;
  metrics: Record<string, number>;
  currency: string;
  variables: VariableInfo[];
}

export interface ApiErrorDetail {
  campo: string;
  mensaje: string;
}

export interface ApiErrorBody {
  code: string;
  message: string;
  details: ApiErrorDetail[];
}

export interface ApiErrorResponse {
  success: false;
  error: ApiErrorBody;
}

/** Unión discriminada de los cuatro estados del flujo de predicción. */
export type PredictionState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "success"; result: PredictionResponse }
  | { status: "error"; message: string };
