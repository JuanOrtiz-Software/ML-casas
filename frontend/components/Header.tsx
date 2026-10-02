"use client";

import { motion } from "framer-motion";
import { House } from "lucide-react";

export function Header() {
  return (
    <motion.header
      initial={{ opacity: 0, y: -16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="flex items-center gap-4"
    >
      <div className="glass-card flex h-12 w-12 items-center justify-center text-indigo-300">
        <House aria-hidden size={24} />
      </div>
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-white sm:text-3xl">
          HOUSEPRICE ML
        </h1>
        <p className="text-sm text-slate-300 sm:text-base">
          Predicción inteligente de precios inmobiliarios
        </p>
      </div>
    </motion.header>
  );
}
