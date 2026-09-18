#!/usr/bin/env bash
# =============================================================================
# run_all.sh — set up and run the Emotion-Aware Multilingual Hybrid Recommender
#
# Usage:
#   ./run_all.sh                 # venv + install + verify + full pipeline
#   ./run_all.sh --no-venv       # skip venv creation (use current environment)
#   ./run_all.sh --no-install    # skip pip install
#   ./run_all.sh --verify-only   # only run the correctness checks
#   ./run_all.sh -- <main args>  # pass everything after -- straight to main.py
#
# Examples:
#   ./run_all.sh -- --max_samples 100000
#   ./run_all.sh -- --mode train --device cpu --epochs 10
#   ./run_all.sh --no-venv -- --mode ablation
# =============================================================================
set -euo pipefail

DO_VENV=1
DO_INSTALL=1
VERIFY_ONLY=0
MAIN_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-venv)     DO_VENV=0; shift ;;
    --no-install)  DO_INSTALL=0; shift ;;
    --verify-only) VERIFY_ONLY=1; shift ;;
    --)            shift; MAIN_ARGS=("$@"); break ;;
    -h|--help)     sed -n '2,20p' "$0"; exit 0 ;;
    *)             echo "unknown option: $1 (use -- to pass args to main.py)"; exit 1 ;;
  esac
done

cd "$(dirname "$0")"
PY="${PYTHON:-python3}"

echo "==> Python: $($PY --version 2>&1)"

if [[ "$DO_VENV" -eq 1 ]]; then
  if [[ ! -d .venv ]]; then
    echo "==> Creating virtual environment (.venv)"
    "$PY" -m venv .venv
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  PY=python
fi

if [[ "$DO_INSTALL" -eq 1 ]]; then
  echo "==> Installing requirements"
  "$PY" -m pip install --upgrade pip >/dev/null
  "$PY" -m pip install -r requirements.txt
fi

echo "==> Verifying pipeline correctness (no model download needed)"
"$PY" main.py --mode verify --device "${DEVICE:-auto}"

if [[ "$VERIFY_ONLY" -eq 1 ]]; then
  echo "==> verify-only: done."
  exit 0
fi

echo "==> Running pipeline: main.py ${MAIN_ARGS[*]:-}"
"$PY" main.py "${MAIN_ARGS[@]:-}"

echo "==> Done. See results/ (figures/ tables/ final_results/ …) for outputs."
