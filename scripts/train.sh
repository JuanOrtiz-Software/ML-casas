#!/usr/bin/env bash
# Ejecuta el entrenamiento de los cinco modelos y el registro del campeón.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

.venv/bin/python ml/train.py
