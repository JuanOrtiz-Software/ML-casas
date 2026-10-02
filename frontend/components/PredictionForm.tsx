"use client";

import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { RotateCcw, Sparkles } from "lucide-react";
import type {
  HousePredictionRequest,
  ModelInfoResponse,
  VariableInfo,
} from "@/lib/types";

type FormValues = Record<keyof HousePredictionRequest, string>;
type FormErrors = Partial<Record<keyof HousePredictionRequest, string>>;

const CAMPOS_NUMERICOS: (keyof HousePredictionRequest)[] = [
  "metros_construidos",
  "metros_terreno",
  "habitaciones",
  "banos",
  "antiguedad",
  "distancia_centro",
  "garage",
  "jardin",
  "pisos",
];

const VALORES_INICIALES: FormValues = {
  metros_construidos: "120",
  metros_terreno: "200",
  habitaciones: "3",
  banos: "2",
  antiguedad: "5",
  distancia_centro: "4.2",
  garage: "1",
  jardin: "1",
  pisos: "2",
  zona: "residencial",
};

interface CampoConfig {
  nombre: keyof HousePredictionRequest;
  etiqueta: string;
  paso: string;
}

const CAMPOS_NUMERICOS_CONFIG: CampoConfig[] = [
  { nombre: "metros_construidos", etiqueta: "Metros construidos", paso: "any" },
  { nombre: "metros_terreno", etiqueta: "Metros de terreno", paso: "any" },
  { nombre: "habitaciones", etiqueta: "Habitaciones", paso: "1" },
  { nombre: "banos", etiqueta: "Baños", paso: "1" },
  { nombre: "antiguedad", etiqueta: "Antigüedad", paso: "1" },
  { nombre: "distancia_centro", etiqueta: "Distancia al centro", paso: "any" },
  { nombre: "pisos", etiqueta: "Pisos", paso: "1" },
];

function construirMapaVariables(
  modelInfo: ModelInfoResponse | null,
): Map<string, VariableInfo> {
  const mapa = new Map<string, VariableInfo>();
  modelInfo?.variables.forEach((variable) => mapa.set(variable.nombre, variable));
  return mapa;
}

function validar(values: FormValues, mapa: Map<string, VariableInfo>): FormErrors {
  const errores: FormErrors = {};

  for (const campo of CAMPOS_NUMERICOS) {
    const crudo = values[campo].trim();
    if (crudo === "") {
      errores[campo] = "Campo obligatorio.";
      continue;
    }
    const numero = Number(crudo);
    if (Number.isNaN(numero)) {
      errores[campo] = "Debe ser un número.";
      continue;
    }
    const rango = mapa.get(campo)?.rango;
    if (rango && (numero < rango[0] || numero > rango[1])) {
      errores[campo] = `Debe estar entre ${rango[0]} y ${rango[1]}.`;
    }
  }

  const construidos = Number(values.metros_construidos);
  const terreno = Number(values.metros_terreno);
  const pisos = Number(values.pisos);
  if (
    !Number.isNaN(construidos) &&
    !Number.isNaN(terreno) &&
    !Number.isNaN(pisos) &&
    pisos > 0 &&
    construidos / pisos > terreno
  ) {
    errores.metros_terreno =
      "El terreno debe ser al menos igual a metros_construidos / pisos.";
  }

  return errores;
}

function construirPayload(values: FormValues): HousePredictionRequest {
  return {
    metros_construidos: Number(values.metros_construidos),
    metros_terreno: Number(values.metros_terreno),
    habitaciones: Number(values.habitaciones),
    banos: Number(values.banos),
    antiguedad: Number(values.antiguedad),
    distancia_centro: Number(values.distancia_centro),
    garage: Number(values.garage),
    jardin: Number(values.jardin),
    pisos: Number(values.pisos),
    zona: values.zona,
  };
}

export function PredictionForm({
  modelInfo,
  onPredict,
  isLoading,
}: {
  modelInfo: ModelInfoResponse | null;
  onPredict: (payload: HousePredictionRequest) => void;
  isLoading: boolean;
}) {
  const [values, setValues] = useState<FormValues>(VALORES_INICIALES);
  const [errores, setErrores] = useState<FormErrors>({});

  const mapa = useMemo(() => construirMapaVariables(modelInfo), [modelInfo]);
  const zonas = mapa.get("zona")?.categorias ?? [];

  function actualizar(campo: keyof HousePredictionRequest, valor: string) {
    const siguientes = { ...values, [campo]: valor };
    setValues(siguientes);
    setErrores(validar(siguientes, mapa));
  }

  function restaurar() {
    setValues(VALORES_INICIALES);
    setErrores(validar(VALORES_INICIALES, mapa));
  }

  function enviar(evento: React.FormEvent<HTMLFormElement>) {
    evento.preventDefault();
    const erroresActuales = validar(values, mapa);
    setErrores(erroresActuales);
    if (Object.keys(erroresActuales).length > 0) {
      return;
    }
    onPredict(construirPayload(values));
  }

  const campoClase =
    "w-full rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-sm text-white outline-none transition focus:border-indigo-400/60 focus:ring-2 focus:ring-indigo-400/30";

  return (
    <form onSubmit={enviar} noValidate>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        {CAMPOS_NUMERICOS_CONFIG.map(({ nombre, etiqueta, paso }) => (
          <div key={nombre}>
            <label
              htmlFor={`campo-${nombre}`}
              className="mb-1 block text-sm font-medium text-slate-200"
            >
              {etiqueta}
            </label>
            <input
              id={`campo-${nombre}`}
              type="number"
              step={paso}
              value={values[nombre]}
              onChange={(evento) => actualizar(nombre, evento.target.value)}
              aria-invalid={Boolean(errores[nombre])}
              aria-describedby={errores[nombre] ? `error-${nombre}` : undefined}
              className={campoClase}
            />
            {errores[nombre] && (
              <p id={`error-${nombre}`} className="mt-1 text-xs text-rose-300">
                {errores[nombre]}
              </p>
            )}
          </div>
        ))}

        <div>
          <label htmlFor="campo-garage" className="mb-1 block text-sm font-medium text-slate-200">
            Garage
          </label>
          <select
            id="campo-garage"
            value={values.garage}
            onChange={(evento) => actualizar("garage", evento.target.value)}
            className={campoClase}
          >
            <option value="1">Sí</option>
            <option value="0">No</option>
          </select>
        </div>

        <div>
          <label htmlFor="campo-jardin" className="mb-1 block text-sm font-medium text-slate-200">
            Jardín
          </label>
          <select
            id="campo-jardin"
            value={values.jardin}
            onChange={(evento) => actualizar("jardin", evento.target.value)}
            className={campoClase}
          >
            <option value="1">Sí</option>
            <option value="0">No</option>
          </select>
        </div>

        <div className="sm:col-span-2">
          <label htmlFor="campo-zona" className="mb-1 block text-sm font-medium text-slate-200">
            Zona
          </label>
          <select
            id="campo-zona"
            value={values.zona}
            onChange={(evento) => actualizar("zona", evento.target.value)}
            className={campoClase}
          >
            {zonas.map((zona) => (
              <option key={zona.valor} value={zona.valor}>
                {zona.etiqueta}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="mt-6 flex flex-col gap-3 sm:flex-row">
        <motion.button
          type="submit"
          disabled={isLoading}
          whileHover={isLoading ? undefined : { scale: 1.02 }}
          whileTap={isLoading ? undefined : { scale: 0.98 }}
          className="inline-flex flex-1 items-center justify-center gap-2 rounded-xl bg-indigo-500 px-6 py-3 font-semibold text-white shadow-lg shadow-indigo-500/25 transition hover:bg-indigo-400 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {isLoading ? (
            <>
              <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/40 border-t-white" />
              ANALIZANDO PROPIEDAD...
            </>
          ) : (
            <>
              <Sparkles aria-hidden size={18} />
              PREDECIR PRECIO
            </>
          )}
        </motion.button>
        <button
          type="button"
          onClick={restaurar}
          disabled={isLoading}
          className="inline-flex items-center justify-center gap-2 rounded-xl border border-white/15 px-6 py-3 text-sm font-medium text-slate-200 transition hover:bg-white/10 disabled:opacity-60"
        >
          <RotateCcw aria-hidden size={16} />
          Restaurar ejemplo
        </button>
      </div>
    </form>
  );
}
