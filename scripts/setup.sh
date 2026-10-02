#!/usr/bin/env bash
# Crea el entorno virtual e instala las dependencias de Python y de Node.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> Creando entorno virtual en .venv"
python3 -m venv .venv

echo "==> Instalando dependencias de Python"
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

echo "==> Instalando dependencias de Node (frontend)"
cd frontend
npm install

echo "==> Entorno listo."
