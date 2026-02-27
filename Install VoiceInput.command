#!/usr/bin/env bash
# voice-input — DMG installer
# Double-click this file in Finder to install everything.

set -e

# ── Helpers ──────────────────────────────────────────────────────────────────
info()  { echo "  $*"; }
step()  { echo ""; echo "▶ $*"; }
ok()    { echo "  ✓ $*"; }
fail()  { echo ""; echo "✗ ERROR: $*" >&2; echo ""; osascript -e "display alert \"voice-input install failed\" message \"$*\"" 2>/dev/null; exit 1; }

clear
echo "╔══════════════════════════════════════════════════════╗"
echo "║         voice-input — Installer                      ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── Locate repo inside the DMG ────────────────────────────────────────────────
# This script lives inside the mounted DMG next to VoiceInput.app
DMG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
info "DMG location: $DMG_DIR"

# ── Choose install location ───────────────────────────────────────────────────
INSTALL_DIR="$HOME/voice-input"
step "Install directory: $INSTALL_DIR"

if [ -d "$INSTALL_DIR" ]; then
    info "Directory already exists — updating in place."
else
    mkdir -p "$INSTALL_DIR"
    ok "Created $INSTALL_DIR"
fi

# ── Copy repo files from DMG ──────────────────────────────────────────────────
step "Copying files..."
rsync -a --exclude='.DS_Store' \
    "$DMG_DIR/voice-input-src/" "$INSTALL_DIR/"
ok "Files copied."

# ── Python venv ───────────────────────────────────────────────────────────────
step "Setting up Python virtual environment..."
VENV="$INSTALL_DIR/.venv"

if ! command -v python3 &>/dev/null; then
    fail "Python 3 not found. Install from https://www.python.org or via Homebrew: brew install python"
fi

PYVER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
info "Python $PYVER found."

if [ ! -d "$VENV" ]; then
    /usr/bin/arch -arm64 python3 -m venv "$VENV"
    ok "Virtual environment created."
else
    ok "Virtual environment already exists."
fi

# ── Python dependencies ───────────────────────────────────────────────────────
step "Installing Python dependencies..."
"$VENV/bin/pip" install --upgrade pip -q
"$VENV/bin/pip" install -r "$INSTALL_DIR/requirements.txt" -q
"$VENV/bin/pip" install \
    pyobjc-core \
    pyobjc-framework-Cocoa \
    pyobjc-framework-AVFoundation \
    sounddevice \
    pynput \
    pyperclip -q
ok "Python dependencies installed."

# ── Ollama ────────────────────────────────────────────────────────────────────
step "Checking Ollama..."
if ! command -v ollama &>/dev/null; then
    echo ""
    echo "  Ollama is not installed."
    echo "  Opening https://ollama.com for you to download it."
    echo "  After installing Ollama, re-run this installer."
    open "https://ollama.com"
    fail "Ollama not found. Install it and run this installer again."
fi
ok "Ollama found."

# Make sure Ollama server is running
if ! ollama list &>/dev/null 2>&1; then
    info "Starting Ollama..."
    open -a Ollama 2>/dev/null || ollama serve &>/dev/null &
    sleep 4
fi

step "Pulling Ollama models (may take several minutes on first run)..."
info "gemma3:4b — text refinement (~2.5 GB download)"
ollama pull gemma3:4b
ok "gemma3:4b ready."

info "qwen2.5:7b — translation (~4.7 GB download)"
ollama pull qwen2.5:7b
ok "qwen2.5:7b ready."

# ── Write launcher and client scripts with correct paths ──────────────────────
step "Configuring scripts for $INSTALL_DIR..."

cat > "$INSTALL_DIR/run_client.sh" <<EOF
#!/bin/bash
cd "$INSTALL_DIR"
export LLM_API_FORMAT=ollama
exec /usr/bin/arch -arm64 "$VENV/bin/python3" mac_client.py --no-screenshot
EOF
chmod +x "$INSTALL_DIR/run_client.sh"

cat > "$INSTALL_DIR/VoiceInput.app/Contents/MacOS/VoiceInput" <<EOF
#!/bin/bash
REPO_DIR="$INSTALL_DIR"
VENV_PYTHON="\$REPO_DIR/.venv/bin/python3"
exec /usr/bin/arch -arm64 "\$VENV_PYTHON" "\$REPO_DIR/menubar_app.py"
EOF
chmod +x "$INSTALL_DIR/VoiceInput.app/Contents/MacOS/VoiceInput"
ok "Scripts configured."

# ── Install VoiceInput.app to /Applications ───────────────────────────────────
step "Installing VoiceInput.app to /Applications..."
rm -rf "/Applications/VoiceInput.app"
cp -r "$INSTALL_DIR/VoiceInput.app" "/Applications/VoiceInput.app"
ok "VoiceInput.app installed."

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║   Installation complete!                             ║"
echo "╠══════════════════════════════════════════════════════╣"
echo "║                                                      ║"
echo "║  REQUIRED — grant Terminal these permissions:        ║"
echo "║                                                      ║"
echo "║  System Settings → Privacy & Security:               ║"
echo "║    ✓ Microphone    → add Terminal                    ║"
echo "║    ✓ Accessibility → add Terminal                    ║"
echo "║                                                      ║"
echo "║  Then open VoiceInput from /Applications             ║"
echo "║                                                      ║"
echo "║  Left  ⌥  — transcribe (French)                     ║"
echo "║  Right ⌥  — translate FR ↔ EN                       ║"
echo "║                                                      ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# Open System Settings → Privacy → Microphone
osascript -e 'tell application "System Settings" to activate' 2>/dev/null || true

# Launch the app
sleep 1
open /Applications/VoiceInput.app
