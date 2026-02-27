# voice-input

Push-to-talk voice transcription and translation for macOS, with a floating HUD overlay and a native menu bar app. Runs entirely on-device — no cloud, no API keys.

**No cloud services. Your voice and screen data never leave your machine.**

## Features

- **Left ⌥ (Option)** — hold to record, release to transcribe and paste refined text
- **Right ⌥ (Option)** — hold to record, release to auto-detect language and translate (FR ↔ EN)
- Floating HUD overlay shows live status (Recording → Transcribing → Refining → Done)
- Menu bar app — no Terminal needed after install
- Runs on local models via [Ollama](https://ollama.com) — fully offline
- French by default; auto-detects language for translation
- Real-time partial transcription during recording
- Screen-aware context via Vision model (optional)

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- [Homebrew](https://brew.sh)
- [Ollama](https://ollama.com) installed and running
- Python 3.11+

## Install

### Option A — DMG (easiest)

1. Download **VoiceInput.dmg** from the [Releases](https://github.com/devlfy/voice-input/releases) page
2. Open the DMG
3. Double-click **Install VoiceInput** — it installs everything automatically:
   - Python virtual environment
   - All Python dependencies
   - Ollama models (`gemma3:4b` for refinement, `qwen2.5:7b` for translation)
   - `VoiceInput.app` → `/Applications`

> **Note:** [Ollama](https://ollama.com) must be installed first. The installer will open the Ollama website if it's missing.

### Option B — Terminal

```bash
git clone https://github.com/devlfy/voice-input.git
cd voice-input
bash install.sh
```

The installer will:
1. Create a Python virtual environment (`.venv`)
2. Install all Python dependencies
3. Pull the required Ollama models (`gemma3:4b`, `qwen2.5:7b`)
4. Install `VoiceInput.app` to `/Applications`
5. Print macOS permission setup steps

### Build the DMG yourself

```bash
git clone https://github.com/devlfy/voice-input.git
cd voice-input
bash build_dmg.sh
# → VoiceInput.dmg
```

## macOS Permissions (required after install)

Open **System Settings** and grant the following to **Terminal**:

| Permission | Path |
|---|---|
| Microphone | Privacy & Security → Microphone |
| Accessibility | Privacy & Security → Accessibility |

> These are one-time grants. The menu bar app launches the client through Terminal to inherit mic and accessibility permissions automatically.

## Usage

### Menu bar app (recommended)

Open **VoiceInput** from `/Applications` or Spotlight. A microphone icon (🎙) appears in the menu bar.

| Shortcut | Action |
|---|---|
| Hold **Left ⌥**, speak, release | Transcribe → refine in French → paste |
| Hold **Right ⌥**, speak, release | Auto-detect language → translate FR↔EN → paste |

Right-click the menu bar icon for: Restart, Open Server Log, Open Client Log, Quit.

### Manual (without the menu bar app)

Terminal 1 — start the server:
```bash
cd voice-input
LLM_API_FORMAT=ollama LLM_MODEL=gemma3:4b WHISPER_MODEL=small .venv/bin/python3 ws_server.py
```

Terminal 2 — start the client:
```bash
cd voice-input
bash run_client.sh
```

## Models

| Role | Model | Notes |
|---|---|---|
| Speech-to-text | Whisper `small` | Fast, ~1 GB RAM |
| Text refinement | `gemma3:4b` | ~3 GB RAM |
| Translation | `qwen2.5:7b` | ~5 GB RAM, used for Right ⌥ |

You can substitute any Ollama-compatible model by editing environment variables in `run_client.sh` and `menubar_app.py`.

### Recommended configs by hardware

| Hardware | Whisper | LLM | Total RAM |
|---|---|---|---|
| Mac 16 GB | `small` | `gemma3:4b` | ~9 GB |
| Mac 32 GB | `large-v3-turbo` | `qwen2.5:7b` | ~15 GB |

## Architecture

```
VoiceInput.app (menu bar)
    └── menubar_app.py          NSStatusBar menu, process management
         ├── ws_server.py        WebSocket server (Whisper + Ollama)
         └── mac_client.py       Audio capture, HUD overlay, hotkeys
              └── run_client.sh  Env wrapper launched via Terminal for mic permission
```

Audio pipeline:
```
Microphone → sounddevice → WebSocket → faster-whisper → Ollama LLM → clipboard → paste
```

## Prompts

Language and translation prompts are in `prompts/`:

| File | Purpose |
|---|---|
| `fr.json` | Refine transcription in French |
| `en.json` | Refine transcription in English |
| `translate_fr.json` | Translate any language → French |
| `translate_en.json` | Translate any language → English |
| `ja.json`, `zh.json`, `ko.json` | Additional languages |

To add a new language, create `prompts/{code}.json`:

```json
{
  "system_prompt": "...",
  "user_template": "Format this:\n```\n{text}\n```",
  "few_shot": [
    { "user": "...", "assistant": "..." }
  ]
}
```

## Translation (Right ⌥)

The Right ⌥ shortcut uses auto-detection:

- Speak in **French** → result pasted in **English**
- Speak in **English** → result pasted in **French**

Detection is done by Whisper. The translation model (`qwen2.5:7b`) handles the actual conversion.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL` | `gemma3:4b` | Ollama model for text refinement |
| `LLM_API_FORMAT` | `ollama` | Must be `ollama` for local inference |
| `WHISPER_MODEL` | `small` | Whisper model size |
| `WHISPER_DEVICE` | `auto` | `auto`, `cuda`, or `cpu` |
| `DEFAULT_LANGUAGE` | `None` | Language hint (`fr`, `en`, etc.), or None for auto |

## Troubleshooting

**Menu bar icon stuck on ⏳**
- Check that Terminal has Accessibility permission in System Settings
- Try Restart from the menu bar menu
- Make sure Ollama is running: `ollama serve`

**No sound / microphone not working**
- Terminal must have Microphone permission in System Settings
- The menu bar app launches the client through Terminal automatically

**Translation always goes one direction**
- Right ⌥ auto-detects source language via Whisper
- If Whisper misidentifies the language (can happen with short phrases), try speaking more

**Slow responses**
- Ensure `LLM_API_FORMAT=ollama` is set (done automatically via `run_client.sh`)
- `qwen2.5:7b` needs ~5 GB RAM; close memory-intensive apps if needed

## Linux / NVIDIA GPU

```bash
git clone https://github.com/xuiltul/voice-input
cd voice-input
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
ollama pull gemma3:4b

# Start server with CUDA
PYVER=$(.venv/bin/python -c 'import sys;print(f"python{sys.version_info.major}.{sys.version_info.minor}")')
LD_LIBRARY_PATH=".venv/lib/$PYVER/site-packages/nvidia/cublas/lib:.venv/lib/$PYVER/site-packages/nvidia/cudnn/lib" \
  LLM_API_FORMAT=ollama .venv/bin/python ws_server.py
```

Connect from Mac:
```bash
pip3 install sounddevice numpy websockets pynput
python3 mac_client.py --server ws://YOUR_SERVER_IP:8991
```

## HTTP API

```bash
# Transcribe + refine
curl -X POST http://localhost:8990/transcribe \
  -H "Content-Type: audio/wav" \
  --data-binary @recording.wav

# Whisper only
curl -X POST "http://localhost:8990/transcribe?raw=true" \
  -H "Content-Type: audio/wav" \
  --data-binary @recording.wav
```

## License

MIT
