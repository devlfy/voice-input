#!/usr/bin/env bash
# Build VoiceInput.dmg
# Run from the repo root: bash build_dmg.sh
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DMG_NAME="VoiceInput"
DMG_OUT="$REPO_DIR/$DMG_NAME.dmg"
STAGING="$REPO_DIR/.dmg-staging"

echo "=== Building $DMG_NAME.dmg ==="

# ── Clean staging area ─────────────────────────────────────────────────────
rm -rf "$STAGING"
mkdir -p "$STAGING"

# ── Copy app bundle ────────────────────────────────────────────────────────
echo "[1/4] Copying VoiceInput.app..."
cp -r "$REPO_DIR/VoiceInput.app" "$STAGING/VoiceInput.app"

# ── Copy installer script ──────────────────────────────────────────────────
echo "[2/4] Copying installer..."
cp "$REPO_DIR/Install VoiceInput.command" "$STAGING/Install VoiceInput.command"
chmod +x "$STAGING/Install VoiceInput.command"

# ── Bundle repo source (needed by the installer) ───────────────────────────
echo "[3/4] Bundling repo source..."
mkdir -p "$STAGING/voice-input-src"
rsync -a \
    --exclude='.git' \
    --exclude='.venv' \
    --exclude='.DS_Store' \
    --exclude='*.dmg' \
    --exclude='.dmg-staging' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='session.md' \
    --exclude='export_session.py' \
    "$REPO_DIR/" "$STAGING/voice-input-src/"

# ── Create DMG ─────────────────────────────────────────────────────────────
echo "[4/4] Creating DMG..."
rm -f "$DMG_OUT"

create-dmg \
    --volname "$DMG_NAME" \
    --volicon "$REPO_DIR/VoiceInput.app/Contents/Resources/AppIcon.icns" \
    --window-pos 200 120 \
    --window-size 600 400 \
    --icon-size 100 \
    --icon "VoiceInput.app" 140 180 \
    --icon "Install VoiceInput.command" 420 180 \
    --hide-extension "VoiceInput.app" \
    --app-drop-link 300 320 \
    "$DMG_OUT" \
    "$STAGING" \
    2>/dev/null || \
create-dmg \
    --volname "$DMG_NAME" \
    --window-pos 200 120 \
    --window-size 600 400 \
    --icon-size 100 \
    --icon "VoiceInput.app" 140 180 \
    --icon "Install VoiceInput.command" 420 180 \
    --hide-extension "VoiceInput.app" \
    "$DMG_OUT" \
    "$STAGING"

# ── Cleanup ────────────────────────────────────────────────────────────────
rm -rf "$STAGING"

echo ""
echo "Done: $DMG_OUT ($(du -sh "$DMG_OUT" | cut -f1))"
