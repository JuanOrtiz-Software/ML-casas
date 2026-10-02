/** Capa única de acceso a la API: centraliza el cliente de fetch y la traducción de errores. */

import type {
  ApiErrorResponse,
  HealthResponse,
  HousePredictionRequest,
  ModelInfoResponse,
  PredictionResponse,
} from "./types";

// NEXT_PUBLIC_API_URL se incorpora en tiempo de compilación (ver README).
const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const TIMEOUT_MS = 10_000;

/** Error de la capa de acceso con código y mensaje en español. */
export class ApiError extends Error {
  code: string;

  constructor(code: string, message: string) {
    super(message);
    this.name = "ApiError";
    this.code = code;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), TIMEOUT_MS);

  try {
    let response: Response;
    try {
      response = await fetch(`${API_URL}${path}`, {
        ...init,
        signal: controller.signal,
        headers: {
          "Content-Type": "application/json",
          ...(init?.headers ?? {}),
        },
      });
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        throw new ApiError(
          "TIMEOUT",
          "La petición tardó demasiado en responder. Inténtalo de nuevo.",
        );
      }
      throw new ApiError(
        "NETWORK",
        "No se pudo conectar con el servidor. Comprueba que la API está en ejecución.",
      );
    }

    if (!response.ok) {
      const body = (await response.json().catch(() => null)) as ApiErrorResponse | null;
      if (body && body.error) {
        const detalle =
          body.error.details.length > 0
            ? body.error.details.map((d) => d.mensaje).join(" · ")
            : body.error.message;
        throw new ApiError(body.error.code, detalle);
      }
      throw new ApiError(
        "HTTP",
        `El servidor respondió con un error (HTTP ${response.status}).`,
      );
    }

    return (await response.json()) as T;
  } finally {
    clearTimeout(timeout);
  }
}

export function predictHouse(
  payload: HousePredictionRequest,
): Promise<PredictionResponse> {
  return request<PredictionResponse>("/predict", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/health");
}

export function getModelInfo(): Promise<ModelInfoResponse> {
  return request<ModelInfoResponse>("/model-info");
}
