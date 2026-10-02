#!/usr/bin/env bash
# Comprueba los endpoints de la API con curl y valida los códigos HTTP.
#
# Uso:  ./scripts/test_api.sh
#        API_URL=http://127.0.0.1:9000 ./scripts/test_api.sh
#
# Termina con código distinto de cero si alguna comprobación falla.
set -euo pipefail

API_URL="${API_URL:-http://localhost:8000}"

# Formatea JSON con jq si está disponible; en caso contrario imprime tal cual.
format_json() {
  if command -v jq >/dev/null 2>&1; then
    jq .
  else
    cat
  fi
}

# Ejecuta curl, guarda el cuerpo en un archivo temporal y devuelve el código HTTP.
request() {
  local method="$1"
  local path="$2"
  local body="${3:-}"
  local body_file
  body_file="$(mktemp)"
  local code
  if [[ -n "$body" ]]; then
    code="$(curl -sS -o "$body_file" -w '%{http_code}' -X "$method" "$API_URL$path" \
      -H 'Content-Type: application/json' -d "$body")"
  else
    code="$(curl -sS -o "$body_file" -w '%{http_code}' -X "$method" "$API_URL$path")"
  fi
  format_json < "$body_file" >&2
  rm -f "$body_file"
  echo "$code"
}

# Comprueba que el código HTTP coincida con el esperado.
check_code() {
  local description="$1"
  local expected="$2"
  local actual="$3"
  if [[ "$actual" != "$expected" ]]; then
    echo "✗ $description: se esperaba HTTP $expected, se obtuvo $actual"
    return 1
  fi
  echo "✓ $description: HTTP $actual"
  return 0
}

result=0

echo "=== GET /health ==="
health_code="$(request GET "/health")"
check_code "/health" "200" "$health_code" || result=1
echo

echo "=== GET /model-info ==="
info_code="$(request GET "/model-info")"
check_code "/model-info" "200" "$info_code" || result=1
echo

echo "=== POST /predict (válido) ==="
pred_code="$(request POST "/predict" '{"metros_construidos":120,"metros_terreno":200,"habitaciones":3,"banos":2,"antiguedad":5,"distancia_centro":4.2,"garage":1,"jardin":1,"pisos":2,"zona":"residencial"}')"
check_code "/predict válido" "200" "$pred_code" || result=1
echo

echo "=== POST /predict (inválido: garage = 2) ==="
bad_code="$(request POST "/predict" '{"metros_construidos":120,"metros_terreno":200,"habitaciones":3,"banos":2,"antiguedad":5,"distancia_centro":4.2,"garage":2,"jardin":1,"pisos":2,"zona":"residencial"}')"
check_code "/predict inválido" "422" "$bad_code" || result=1

if [[ "$result" -ne 0 ]]; then
  echo
  echo "Alguna comprobación falló."
  exit 1
fi

echo
echo "Todas las comprobaciones pasaron."
