"use client";

import { motion } from "framer-motion";
import { AlertTriangle, CheckCircle2 } from "lucide-react";
import { GlassCard } from "./GlassCard";
import { formatCurrency } from "@/lib/format";
import type { ModelInfoResponse, PredictionState } from "@/lib/types";

export function PredictionResult({
  state,
  modelInfo,
}: {
  state: PredictionState;
  modelInfo: ModelInfoResponse | null;
}) {
  if (state.status === "idle" || state.status === "loading") {
    return null;
  }

  if (state.status === "error") {
    return (
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        aria-live="polite"
      >
        <GlassCard className="border-rose-500/30 p-6">
          <div className="flex items-start gap-3">
            <AlertTriangle aria-hidden size={22} className="mt-0.5 shrink-0 text-rose-300" />
            <div>
              <p className="font-semibold text-white">No se pudo obtener la estimación</p>
              <p className="mt-1 text-sm text-slate-300">{state.message}</p>
              <p className="mt-2 text-sm text-slate-400">
                Revisa los valores e inténtalo de nuevo.
              </p>
            </div>
          </div>
        </GlassCard>
      </motion.div>
    );
  }

  const { prediction, model, warnings } = state.result;
  const algoritmoAmigable =
    modelInfo && modelInfo.algorithm === model.algorithm
      ? modelInfo.algorithm_name
      : model.algorithm;
  const r2 = modelInfo?.metrics.r2 != null ? modelInfo.metrics.r2.toFixed(4) : null;
  const rmse =
    modelInfo?.metrics.rmse != null ? formatCurrency(modelInfo.metrics.rmse) : null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      aria-live="polite"
    >
      <GlassCard className="border-indigo-400/30 p-6 sm:p-8">
        <div className="flex items-center gap-2 text-indigo-300">
          <CheckCircle2 aria-hidden size={20} />
          <span className="text-xs font-semibold uppercase tracking-widest text-slate-300">
            Precio estimado
          </span>
        </div>

        <p className="mt-3 text-4xl font-bold tracking-tight text-white sm:text-5xl">
          {formatCurrency(prediction.precio_estimado)}
        </p>

        <p className="mt-3 text-sm text-slate-300">
          Rango referencial:{" "}
          <span className="font-medium text-white">
            {formatCurrency(prediction.rango_referencial.minimo)} –{" "}
            {formatCurrency(prediction.rango_referencial.maximo)}
          </span>
          <span className="ml-1 text-slate-500">(± RMSE, no es un intervalo de confianza)</span>
        </p>

        {warnings.length > 0 && (
          <div className="mt-4 rounded-lg border border-amber-400/30 bg-amber-400/10 p-3">
            <ul className="space-y-1 text-sm text-amber-200">
              {warnings.map((warning) => (
                <li key={warning} className="flex items-start gap-2">
                  <AlertTriangle aria-hidden size={16} className="mt-0.5 shrink-0" />
                  {warning}
                </li>
              ))}
            </ul>
          </div>
        )}

        <div className="mt-6 grid grid-cols-2 gap-4 border-t border-white/10 pt-4 text-sm sm:grid-cols-4">
          <div>
            <dt className="text-slate-400">Modelo</dt>
            <dd className="font-medium text-white">{model.name}</dd>
          </div>
          <div>
            <dt className="text-slate-400">Versión</dt>
            <dd className="font-medium text-white">{model.version}</dd>
          </div>
          <div>
            <dt className="text-slate-400">Algoritmo</dt>
            <dd className="font-medium text-white">{algoritmoAmigable}</dd>
          </div>
          <div>
            <dt className="text-slate-400">R² / RMSE</dt>
            <dd className="font-medium text-white">
              {r2 ?? "—"} / {rmse ?? "—"}
            </dd>
          </div>
        </div>
      </GlassCard>
    </motion.div>
  );
}
