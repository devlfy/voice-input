#!/usr/bin/env bash
# voice-input installer for macOS (Apple Silicon)
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$REPO_DIR/.venv"
APP_SRC="$REPO_DIR/VoiceInput.app"
APP_DEST="/Applications/VoiceInput.app"

echo "=== voice-input installer ==="
echo "Repo: $REPO_DIR"
echo ""

# ── 1. Python venv ──────────────────────────────────────────────────────────
if [ ! -d "$VENV" ]; then
    echo "[1/5] Creating Python virtual environment..."
    python3 -m venv "$VENV"
else
    echo "[1/5] Virtual environment already exists, skipping."
fi

# ── 2. Python dependencies ──────────────────────────────────────────────────
echo "[2/5] Installing Python dependencies..."
"$VENV/bin/pip" install --upgrade pip -q
"$VENV/bin/pip" install -r "$REPO_DIR/requirements.txt" -q

# PyObjC (menu bar app)
"$VENV/bin/pip" install pyobjc-core pyobjc-framework-Cocoa pyobjc-framework-AVFoundation -q

# Audio and hotkeys
"$VENV/bin/pip" install sounddevice pynput pyperclip -q

echo "    Python dependencies installed."

# ── 3. Ollama models ─────────────────────────────────────────────────────────
echo "[3/5] Pulling Ollama models (this may take a few minutes)..."

if ! command -v ollama &>/dev/null; then
    echo ""
    echo "    ERROR: Ollama is not installed."
    echo "    Install it from https://ollama.com then re-run this script."
    exit 1
fi

# Check if Ollama is running; start it if not
if ! ollama list &>/dev/null 2>&1; then
    echo "    Starting Ollama..."
    ollama serve &>/dev/null &
    sleep 3
fi

echo "    Pulling gemma3:4b (refinement model)..."
ollama pull gemma3:4b

echo "    Pulling qwen2.5:7b (translation model)..."
ollama pull qwen2.5:7b

echo "    Models ready."

# ── 4. Install VoiceInput.app ────────────────────────────────────────────────
echo "[4/5] Installing VoiceInput.app to /Applications..."

if [ ! -d "$APP_SRC" ]; then
    echo "    ERROR: $APP_SRC not found. Make sure you cloned the full repo."
    exit 1
fi

# Update the launcher script with the correct repo path
LAUNCHER="$APP_SRC/Contents/MacOS/VoiceInput"
cat > "$LAUNCHER" <<EOF
#!/bin/bash
REPO_DIR="$REPO_DIR"
VENV_PYTHON="\$REPO_DIR/.venv/bin/python3"
exec /usr/bin/arch -arm64 "\$VENV_PYTHON" "\$REPO_DIR/menubar_app.py"
EOF
chmod +x "$LAUNCHER"

# Update run_client.sh with correct paths
cat > "$REPO_DIR/run_client.sh" <<EOF
#!/bin/bash
cd "$REPO_DIR"
export LLM_API_FORMAT=ollama
exec /usr/bin/arch -arm64 "$VENV/bin/python3" mac_client.py --no-screenshot
EOF
chmod +x "$REPO_DIR/run_client.sh"

# Copy app to /Applications (requires no special permission on macOS for user apps)
rm -rf "$APP_DEST"
cp -r "$APP_SRC" "$APP_DEST"
echo "    VoiceInput.app installed to /Applications."

# ── 5. Permissions reminder ──────────────────────────────────────────────────
echo "[5/5] Done!"
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          REQUIRED: Grant macOS permissions to Terminal       ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║                                                              ║"
echo "║  System Settings → Privacy & Security:                       ║"
echo "║                                                              ║"
echo "║  ✓ Microphone       → add Terminal                           ║"
echo "║  ✓ Accessibility    → add Terminal                           ║"
echo "║                                                              ║"
echo "║  (These are one-time grants. The app uses Terminal           ║"
echo "║   internally to inherit these permissions.)                  ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Launch: open /Applications/VoiceInput.app"
echo ""
echo "Shortcuts:"
echo "  Left  ⌥  — transcribe + refine in French"
echo "  Right ⌥  — auto-detect language + translate FR↔EN"
echo ""
