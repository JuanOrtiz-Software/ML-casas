"use client";

import { useCallback, useEffect, useState } from "react";
import { ApiError, getModelInfo, predictHouse } from "@/lib/api";
import type {
  HousePredictionRequest,
  ModelInfoResponse,
  PredictionState,
} from "@/lib/types";
import { GlassCard } from "@/components/GlassCard";
import { ModelInfo } from "@/components/ModelInfo";
import { PredictionForm } from "@/components/PredictionForm";
import { PredictionResult } from "@/components/PredictionResult";

export default function HomePage() {
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [infoError, setInfoError] = useState<string | null>(null);
  const [state, setState] = useState<PredictionState>({ status: "idle" });

  useEffect(() => {
    let activo = true;
    getModelInfo()
      .then((info) => {
        if (activo) setModelInfo(info);
      })
      .catch((error) => {
        if (activo) {
          setInfoError(
            error instanceof ApiError
              ? error.message
              : "No se pudo cargar la información del modelo.",
          );
        }
      });
    return () => {
      activo = false;
    };
  }, []);

  const handlePredict = useCallback(async (payload: HousePredictionRequest) => {
    setState({ status: "loading" });
    try {
      const result = await predictHouse(payload);
      setState({ status: "success", result });
    } catch (error) {
      const message =
        error instanceof ApiError
          ? error.message
          : "Ocurrió un error inesperado al predecir.";
      setState({ status: "error", message });
    }
  }, []);

  return (
    <div className="space-y-8">
      <section className="text-center">
        <h2 className="text-3xl font-bold tracking-tight text-white sm:text-4xl">
          ¿Cuánto vale tu casa?
        </h2>
        <p className="mx-auto mt-3 max-w-2xl text-slate-300">
          Introduce las características de una propiedad y nuestro modelo de
          Machine Learning generará una estimación.
        </p>
      </section>

      <GlassCard className="p-6">
        {infoError ? (
          <p className="text-sm text-rose-300">{infoError}</p>
        ) : modelInfo ? (
          <PredictionForm
            modelInfo={modelInfo}
            onPredict={handlePredict}
            isLoading={state.status === "loading"}
          />
        ) : (
          <p className="text-sm text-slate-300">
            Cargando configuración del modelo…
          </p>
        )}
      </GlassCard>

      <PredictionResult state={state} modelInfo={modelInfo} />

      <ModelInfo info={modelInfo} />
    </div>
  );
}
