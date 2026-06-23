#!/usr/bin/env bash
# Launch the Mervin/Mervis llama-cpp model-switching server (serve.py).
# Model GGUFs live in phi4mini/ (and optionally gemma4e4b/); they are gitignored.
set -euo pipefail
cd "$(dirname "$0")"
exec uv run serve.py
