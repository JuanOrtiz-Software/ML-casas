"use client";

import { CalendarDays, Cpu, Layers } from "lucide-react";
import { GlassCard } from "./GlassCard";
import { formatCurrency, formatDate } from "@/lib/format";
import type { ModelInfoResponse } from "@/lib/types";

export function ModelInfo({ info }: { info: ModelInfoResponse | null }) {
  if (!info) return null;

  const r2 = info.metrics.r2 != null ? info.metrics.r2.toFixed(4) : "—";
  const rmse =
    info.metrics.rmse != null ? formatCurrency(info.metrics.rmse) : "—";

  return (
    <GlassCard className="p-6">
      <h2 className="mb-4 flex items-center gap-2 text-lg font-semibold text-white">
        <Cpu aria-hidden size={20} className="text-indigo-300" />
        Información del modelo
      </h2>
      <dl className="grid grid-cols-1 gap-x-6 gap-y-3 text-sm sm:grid-cols-2">
        <div className="flex items-center gap-2">
          <dt className="text-slate-400">Modelo:</dt>
          <dd className="font-medium text-white">
            {info.name} <span className="text-slate-500">v{info.version}</span>
          </dd>
        </div>
        <div className="flex items-center gap-2">
          <dt className="text-slate-400">Algoritmo:</dt>
          <dd className="font-medium text-white">{info.algorithm_name}</dd>
        </div>
        <div className="flex items-center gap-2">
          <dt className="text-slate-400">R²:</dt>
          <dd className="font-medium text-white">{r2}</dd>
        </div>
        <div className="flex items-center gap-2">
          <dt className="text-slate-400">RMSE:</dt>
          <dd className="font-medium text-white">{rmse}</dd>
        </div>
        <div className="flex items-center gap-2 sm:col-span-2">
          <CalendarDays aria-hidden size={16} className="text-slate-500" />
          <dt className="text-slate-400">Entrenado:</dt>
          <dd className="font-medium text-white">{formatDate(info.trained_at)}</dd>
        </div>
        <div className="flex items-center gap-2 sm:col-span-2">
          <Layers aria-hidden size={16} className="text-slate-500" />
          <dt className="text-slate-400">Alias:</dt>
          <dd className="font-medium text-white">{info.alias}</dd>
        </div>
      </dl>
    </GlassCard>
  );
}
