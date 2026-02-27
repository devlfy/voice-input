#!/bin/bash
# Runs mac_client.py with Terminal's microphone permission
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"
export LLM_API_FORMAT=ollama
exec /usr/bin/arch -arm64 "$REPO_DIR/.venv/bin/python3" mac_client.py --no-screenshot
