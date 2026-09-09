#!/usr/bin/env bash
# Run the OpenAI-compatible speech shim locally.
#
# Reads RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID from the environment or
# .env.local, and requires SHIM_API_KEY — the token callers present as
# "Authorization: Bearer <token>".
#
# Usage: SHIM_API_KEY=<token> RUNPOD_ENDPOINT_ID=<id> bash scripts/run_shim.sh
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

if [[ -z "${SHIM_API_KEY:-}" ]]; then
  echo "SHIM_API_KEY must be set; it is the bearer token callers present." >&2
  exit 2
fi

export SHIM_HOST="${SHIM_HOST:-127.0.0.1}"
export SHIM_PORT="${SHIM_PORT:-8080}"

echo "shim listening on http://${SHIM_HOST}:${SHIM_PORT}/v1/audio/speech" >&2
exec "${PYTHON:-python3}" -u scripts/openai_shim.py
