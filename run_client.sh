#!/bin/bash
# Runs mac_client.py with Terminal's microphone permission
cd /Users/laaf/voice-input
export LLM_API_FORMAT=ollama
exec /usr/bin/arch -arm64 /Users/laaf/voice-input/.venv/bin/python3 mac_client.py --no-screenshot
