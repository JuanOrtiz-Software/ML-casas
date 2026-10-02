#!/usr/bin/env bash
# Genera y valida el dataset sintético.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

.venv/bin/python data/generate_dataset.py
.venv/bin/python data/validate_dataset.py
