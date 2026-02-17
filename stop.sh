#!/usr/bin/env bash
# voice-input server stopper
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
PIDS=$(pgrep -f "$DIR/ws_server.py" 2>/dev/null || true)

if [[ -z "$PIDS" ]]; then
  echo "ws_server.py is not running"
  exit 0
fi

echo "Stopping ws_server.py (PID: $PIDS)..."
kill $PIDS
sleep 1

# Force kill if still alive
REMAINING=$(pgrep -f "$DIR/ws_server.py" 2>/dev/null || true)
if [[ -n "$REMAINING" ]]; then
  echo "Force killing (PID: $REMAINING)..."
  kill -9 $REMAINING
fi

echo "Stopped."
