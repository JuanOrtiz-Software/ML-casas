"use client";

import { useEffect, useState } from "react";
import { getHealth } from "@/lib/api";

type ApiState = "online" | "online-no-model" | "offline";

const ESTADOS: Record<ApiState, { texto: string; textoCls: string; puntoCls: string }> = {
  online: {
    texto: "API ONLINE",
    textoCls: "text-emerald-300",
    puntoCls: "bg-emerald-400",
  },
  "online-no-model": {
    texto: "API ONLINE · SIN MODELO",
    textoCls: "text-amber-300",
    puntoCls: "bg-amber-400",
  },
  offline: {
    texto: "API OFFLINE",
    textoCls: "text-rose-300",
    puntoCls: "bg-rose-400",
  },
};

const INTERVALO_MS = 20_000;

export function ApiStatus() {
  const [estado, setEstado] = useState<ApiState>("offline");

  useEffect(() => {
    let activo = true;

    const comprobar = async () => {
      try {
        const health = await getHealth();
        if (!activo) return;
        setEstado(health.model_loaded ? "online" : "online-no-model");
      } catch {
        if (activo) setEstado("offline");
      }
    };

    comprobar();
    const intervalo = setInterval(comprobar, INTERVALO_MS);
    return () => {
      activo = false;
      clearInterval(intervalo);
    };
  }, []);

  const actual = ESTADOS[estado];

  return (
    <div className="flex items-center gap-2" aria-live="polite">
      <span
        className={`h-2.5 w-2.5 rounded-full ${actual.puntoCls} animate-pulse`}
        aria-hidden
      />
      <span className={`text-xs font-semibold tracking-wide ${actual.textoCls}`}>
        ● {actual.texto}
      </span>
    </div>
  );
}
