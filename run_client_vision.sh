#!/bin/bash
# Runs mac_client.py with screenshot/vision context enabled
# Uses gemma3:4b as vision model (already in memory, fast)
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"
export LLM_API_FORMAT=ollama
export VISION_MODEL=gemma3:4b
exec /usr/bin/arch -arm64 "$REPO_DIR/.venv/bin/python3" mac_client.py
