#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" -m pip install -r requirements-local-tools.txt
"$PYTHON_BIN" scripts/setup_local_classifiers.py --model all

if [[ "${WITH_RAG_MODELS:-0}" == "1" ]]; then
  "$PYTHON_BIN" -m pip install -r requirements-rag.txt
  "$PYTHON_BIN" scripts/setup_rag.py
fi

if [[ "${WITH_OCR:-0}" == "1" ]]; then
  "$PYTHON_BIN" -m pip install -r requirements-ocr.txt
  "$PYTHON_BIN" scripts/setup_ocr.py
fi

echo "Local models are ready. No API key was read or written."
