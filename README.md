# voice-input

Local voice input with screen-aware context. Push-to-talk on Mac, transcribed by Whisper, refined by a local LLM — all running on your own GPU.

**No cloud services. Your voice and screen data never leave your network.**

## Features

- **Push-to-talk** — Hold left Option key on Mac to record, release to transcribe
- **Real-time streaming** — Audio streamed every 2s with partial transcription results during recording
- **Screen-aware context** — Focused window screenshot captured at recording start, analyzed by a vision model (qwen3-vl) to extract active tab text content, and used to inform text refinement
- **LLM text refinement** — Removes filler words, adds punctuation, formats lists as bullet points, and fixes recognition errors
- **Multi-language support** — Language-specific prompts for Japanese, English, Chinese, and Korean (auto-detected or configurable)
- **Floating HUD** — macOS cursor-following status overlay showing recording/transcribing/refining state. Background turns **green** when vision analysis completes during recording, so you know screen context will be used
- **Auto-paste + Enter** — Result pasted via Cmd+V and Enter sent automatically. Hold Ctrl during recording to paste without Enter
- **Voice slash commands** — Say "スラッシュ ヘルプ" or "slash help" to input `/help` directly. Commands auto-loaded from `~/.claude/skills/` at startup. Pasted without Enter so you can review before submitting

## How it works

```
Mac (Push-to-Talk)              Server (GPU)
─────────────────               ─────────────
[Hold Option key]
  ├─ Capture screenshot ──────→ Vision analysis (qwen3-vl) ──┐
  ├─ Record audio                                          │
  ├─ Stream chunks (2s) ─────→ Whisper partial results     │
  │                              ↓ (shown in HUD)          │
[Release key]                                              │
  └─ Send final audio ───────→ Whisper final transcribe ───┤
                                                           │
                                LLM refine (context+text) ←┘
                                         │
  Auto-paste via Cmd+V  ←───── Result ───┘
```

The screenshot analysis runs **in parallel** with recording. When it finishes, the HUD turns green to signal that screen context is available. Partial transcription results are displayed during recording, giving immediate feedback.

## Quick start

### Server (Linux with NVIDIA GPU)

```bash
# Clone
git clone https://github.com/xuiltul/voice-input
cd voice-input

# Setup Python environment
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Install Ollama (https://ollama.com)
ollama pull gpt-oss:20b    # Text refinement (or any model you prefer)
ollama pull qwen3-vl:8b-instruct  # Screenshot analysis (if running vision locally)

# Start WebSocket server (LD_LIBRARY_PATH required for pip-installed CUDA libs)
LD_LIBRARY_PATH=".venv/lib/$(python3 -c 'import sys;print(f"python{sys.version_info.major}.{sys.version_info.minor}")')/site-packages/nvidia/cublas/lib:.venv/lib/$(python3 -c 'import sys;print(f"python{sys.version_info.major}.{sys.version_info.minor}")')/site-packages/nvidia/cudnn/lib" \
  .venv/bin/python ws_server.py
```

> **CUDA note:** When NVIDIA libraries (cublas, cudnn) are installed via pip into the venv, they are not on the default library search path. You must set `LD_LIBRARY_PATH` to include the venv's `nvidia/*/lib` directories, or Whisper will fail with `Library libcublas.so.12 is not found`. The path includes the Python version directory (e.g., `python3.13`), so use the shell snippet above to auto-detect it.

> **VRAM note:** Whisper (~3 GB) stays loaded. gpt-oss (~12 GB) loads on demand. For best performance, run the vision model on a separate GPU server via `VISION_SERVERS` to avoid model swapping.

### Docker

```bash
# Build
docker build -t voice-input .

# Run with GPU (vision on local Ollama)
docker run --gpus all -p 8991:8991 \
  -e LLM_MODEL=gpt-oss:20b \
  -e VISION_MODEL=qwen3-vl:8b-instruct \
  voice-input

# Run with remote vision server (avoids VRAM contention)
docker run --gpus all -p 8991:8991 \
  -e LLM_MODEL=gpt-oss:20b \
  -e VISION_MODEL=qwen3-vl:8b-instruct \
  -e VISION_SERVERS=http://vision-gpu:11434 \
  voice-input

# Persist Ollama models across restarts
docker run --gpus all -p 8991:8991 \
  -v ollama-data:/root/.ollama \
  voice-input
```

### Mac-only setup (Apple Silicon, 16 GB)

No Linux server needed. Everything runs on your Mac using Ollama + CPU Whisper.

```bash
# Clone
git clone https://github.com/xuiltul/voice-input
cd voice-input

# Setup
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Install Ollama (https://ollama.com)
ollama pull gemma3:4b       # Lightweight refinement model (~3 GB)

# Start server (CPU Whisper + small model, no CUDA needed)
WHISPER_MODEL=small LLM_MODEL=gemma3:4b .venv/bin/python ws_server.py
```

Then in another terminal:

```bash
python3 mac_client.py --server ws://localhost:8991 --model gemma3:4b
```

**Memory budget (16 GB Unified Memory):**

| Component | Memory | Notes |
|-----------|--------|-------|
| macOS | ~5 GB | System overhead |
| Whisper `small` | ~1 GB | CPU inference, int8 quantization |
| `gemma3:4b` | ~3 GB | Fast refinement via Ollama |
| **Total** | **~9 GB** | Leaves headroom for other apps |

> **Tip:** If you have 32 GB+, use `WHISPER_MODEL=large-v3-turbo` and `LLM_MODEL=qwen2.5:7b` for better accuracy. Vision (`qwen3-vl:8b-instruct`) can also fit but adds ~5 GB.

### Mac client

```bash
pip3 install sounddevice numpy websockets pynput
scp your-server:~/voice-input/mac_client.py ~/
python3 mac_client.py --server ws://YOUR_SERVER_IP:8991
```

Grant these permissions in System Settings > Privacy & Security:
- **Microphone** → Terminal
- **Accessibility** → Terminal (for key monitoring + auto-paste)
- **Screen Recording** → Terminal (for screenshot context)

### Auto-start (Automator app)

macOS requires a `.app` bundle to grant privacy permissions (Accessibility, Input Monitoring, etc.). A raw `python3` process launched by launchd cannot receive these permissions. The recommended approach is to wrap the client in an Automator application:

1. **Copy the client script**

```bash
mkdir -p ~/voice-input
cp mac_client.py ~/voice-input/
```

2. **Create an Automator app**

   - Open **Automator.app** → choose **Application**
   - Add a **Run Shell Script** action
   - Set Shell to `/bin/bash` and paste:

```bash
cd ~/voice-input && /usr/bin/python3 mac_client.py --server ws://YOUR_SERVER_IP:8991
```

   - Save as `~/Applications/VoiceInput.app`

3. **Grant permissions** in System Settings > Privacy & Security:
   - **Accessibility** → VoiceInput.app
   - **Input Monitoring** → VoiceInput.app
   - **Microphone** → VoiceInput.app
   - **Screen Recording** → VoiceInput.app

4. **Add to Login Items** for auto-start:
   - System Settings > General > Login Items → add VoiceInput.app

Double-click VoiceInput.app to launch. The HUD appears at the bottom of the screen when the client is running.

## Usage

### Push-to-talk

1. **Hold left Option/Alt** — Recording starts, screenshot captured, streaming begins
2. **During recording** — Partial transcription shown in floating HUD at screen bottom
3. **HUD turns green** — Vision analysis of your screen is complete; screen context will be used for refinement
4. **Release** — Final audio transcribed with VAD → LLM refines with screen context → result pasted + **Enter sent**
5. **Hold left Option/Alt + Ctrl** — Same as above, but paste only (no Enter) — useful for text editors

### HUD indicator colors

| Color | Meaning |
|-------|---------|
| Dark (default) | Recording in progress, vision analysis not yet complete |
| **Green** | Vision analysis complete — screen context will be used for text refinement |

The HUD automatically resets to dark at the start of each new recording.

### Practical tips

- **Short recordings (under ~10s):** Vision analysis may not finish in time. The HUD staying dark means your text will be refined without screen context. This is fine for simple dictation
- **Longer recordings:** The HUD will turn green during recording, meaning the LLM will use your screen content to improve accuracy (e.g., recognizing technical terms visible on screen)
- **For best accuracy with technical terms:** Wait until the HUD turns green before releasing the key. This gives the vision model time to read your screen and provide context to the LLM
- **Claude Code / chat apps:** Use the default Alt mode — text is pasted and Enter is sent automatically, submitting your message instantly
- **Text editors / documents:** Hold Alt + Ctrl — text is pasted without pressing Enter, so you can review before submitting
- **Dictation in any language:** The system auto-detects the language from your speech. You can also set a language hint with `--language en` for better accuracy

### Voice slash commands

Say "スラッシュ" (or "slash") followed by a command name to input a slash command instead of dictated text. The LLM matches your spoken words to the closest available command.

**Examples:**
- 「スラッシュ ヘルプ」→ `/help`
- 「スラッシュ コミット」→ `/commit`
- 「スラッシュ イシュートゥーピーアール 123」→ `/issue-to-pr 123`
- 「スラッシュ ピーディーエフ」→ `/pdf`
- 「slash compact」→ `/compact`

Commands are auto-loaded from `~/.claude/skills/*/SKILL.md` at client startup, plus built-in Claude Code commands (/help, /clear, /compact, /cost, /doctor, /init, /fast). Slash commands are always pasted without Enter so you can review before submitting.

If no command matches, the system falls back to normal text refinement.

### Client options

```
python3 mac_client.py [options]

  -s, --server URL      WebSocket server (default: ws://YOUR_SERVER_IP:8991)
  -l, --language CODE   Language hint for Whisper (default: ja)
  -m, --model NAME      Ollama model for refinement (default: gpt-oss:20b)
  --raw                 Skip LLM refinement, Whisper output only
  -p, --prompt TEXT     Custom refinement instructions
  --no-paste            Copy to clipboard only, don't auto-paste
  --no-screenshot       Disable screenshot context analysis
```

### Server CLI

```bash
# WebSocket server (for Mac client)
voice-input serve ws --port 8991

# HTTP API server
voice-input serve --port 8990

# Transcribe a file directly
voice-input recording.mp3
voice-input recording.mp3 --raw
voice-input recording.mp3 --output json
voice-input recording.mp3 --language en --model qwen3:30b
```

### HTTP API

```bash
# Transcribe + refine
curl -X POST http://localhost:8990/transcribe \
  -H "Content-Type: audio/wav" \
  --data-binary @recording.wav

# Whisper only (skip LLM)
curl -X POST "http://localhost:8990/transcribe?raw=true" \
  -H "Content-Type: audio/wav" \
  --data-binary @recording.wav

# With language hint and custom prompt
curl -X POST "http://localhost:8990/transcribe?language=en&prompt=Format%20as%20meeting%20notes" \
  -H "Content-Type: audio/wav" \
  --data-binary @recording.wav
```

Response:
```json
{
  "text": "Refined text here",
  "raw_text": "Raw Whisper output",
  "language": "ja",
  "duration": 5.2,
  "processing_time": {
    "transcribe": 0.3,
    "refine": 4.1,
    "total": 4.4
  }
}
```

## Multi-language prompts

Refinement prompts are stored in the `prompts/` directory as JSON files, one per language:

```
prompts/
├── ja.json    # Japanese (default)
├── en.json    # English
├── zh.json    # Chinese
└── ko.json    # Korean
```

The language is determined by:
1. **Client config** — `--language` flag on the client
2. **Whisper auto-detection** — If no language is specified, Whisper detects it and the matching prompt is used

To add a new language, create `prompts/{lang_code}.json` with this structure:

```json
{
  "system_prompt": "Your system prompt here...",
  "user_template": "Please format the following text.\n```\n{text}\n```",
  "few_shot": [
    {
      "user": "Please format the following text.\n```\nraw input example\n```",
      "assistant": "Formatted output example"
    }
  ],
  "context_prefix": "Context information (from screenshot analysis):",
  "custom_prompt_prefix": "Additional instructions:"
}
```

If a language has no matching prompt file, it falls back to English, then Japanese.

> **Prompt design note:** Keep the system prompt concise (~400 chars for Japanese). Small models (7B-20B) with `think: "low"` degrade significantly when the prompt is too long — they start dropping content or garbling words instead of formatting. Use few-shot examples (max 2) to demonstrate behavior rather than writing exhaustive rules in the system prompt.

## Architecture

| Component | Role | Tech |
|-----------|------|------|
| `voice_input.py` | Core pipeline: Whisper + Ollama LLM refinement + Vision | faster-whisper (CUDA), Ollama API |
| `ws_server.py` | WebSocket server, orchestrates streaming pipeline | Python, websockets |
| `mac_client.py` | Push-to-talk, screenshot, HUD overlay, clipboard paste | Python, pynput, sounddevice, PyObjC |
| `prompts/` | Language-specific refinement prompts | JSON |
| `transcribe.py` | Standalone Whisper CLI tool | faster-whisper |

### Models

| Model | Purpose | VRAM | Lifecycle |
|-------|---------|------|-----------|
| `large-v3-turbo` | Whisper speech recognition | ~3 GB | Loaded once at startup, stays in memory. Auto-detects CUDA/CPU |
| `gpt-oss:20b` | Text refinement (configurable) | ~12 GB | Managed by Ollama (load on demand), `think: "low"` for speed |
| `qwen3-vl:8b-instruct` | Active tab text extraction (focused window screenshot) | ~5 GB | Runs on separate GPU server (no local VRAM usage) |

### WebSocket protocol

**Streaming mode** (recommended):

```
Client → Server: {"type": "stream_start", "screenshot": "<base64>"}
Client → Server: <binary WAV chunks> (every 2 seconds, cumulative audio)
Server → Client: {"type": "partial", "text": "..."}  (after each chunk)
Server → Client: {"type": "status", "stage": "vision_ready"}  (when screenshot analysis completes)
Client → Server: <final binary WAV> → {"type": "stream_end"}
Server → Client: {"type": "status", "stage": "refining"}
Server → Client: {"type": "result", "text": "...", "raw_text": "...", ...}
```

**Legacy mode** (single-shot):

```
Client → Server: {"type": "audio_with_screenshot", "screenshot": "<base64>"}
Client → Server: <binary WAV> (complete recording)
Server → Client: {"type": "result", ...}
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL for text refinement |
| `LLM_MODEL` | `gpt-oss:20b` | Model for text refinement |
| `VISION_MODEL` | `qwen3-vl:8b-instruct` | Model for active tab text extraction |
| `VISION_SERVERS` | *(unset = local Ollama)* | Comma-separated Ollama URLs for remote vision inference |
| `WHISPER_MODEL` | `large-v3-turbo` | Whisper model name (`small`, `medium`, `large-v3-turbo`) |
| `WHISPER_DEVICE` | `auto` | Whisper device (`auto`, `cuda`, `cpu`) |
| `WHISPER_COMPUTE_TYPE` | `default` | Whisper compute type (`default`, `float16`, `int8`). `default` = float16 for CUDA, int8 for CPU |
| `DEFAULT_LANGUAGE` | `ja` | Default language for transcription |
| `VOICE_INPUT_SERVER` | `ws://localhost:8991` | Mac client: default WebSocket server URL |

**Example: separate vision server** (recommended for single-GPU setups):

```bash
export VISION_MODEL=qwen3-vl:8b-instruct
export VISION_SERVERS=http://gpu-server-1:11434,http://gpu-server-2:11434
python ws_server.py
```

### VRAM management

- **Whisper** (~3 GB) is loaded once at server startup as a singleton
- **LLM** (~12 GB for gpt-oss:20b) stays loaded in Ollama with `think: "low"` for fast inference (~0.5s)
- **Vision** runs on local Ollama by default. Set `VISION_SERVERS` to offload to separate GPU(s) and avoid model swapping. Uses an active-tab-focused prompt to maximize text extraction from the focused window
- Screenshots are captured at full resolution (no resize) for best OCR accuracy. Analysis runs in parallel with recording; if not ready when refinement starts, proceeds without context
- VAD (Voice Activity Detection) is disabled for streaming chunks but enabled for final transcription

## Troubleshooting

### `Library libcublas.so.12 is not found or cannot be loaded`

This happens when NVIDIA CUDA libraries installed via pip (nvidia-cublas-cu12, nvidia-cudnn-cu12) are not on the library search path. Set `LD_LIBRARY_PATH` to include the venv's nvidia lib directories:

```bash
# Auto-detect Python version in venv
PYVER=$(.venv/bin/python -c 'import sys;print(f"python{sys.version_info.major}.{sys.version_info.minor}")')
export LD_LIBRARY_PATH=".venv/lib/$PYVER/site-packages/nvidia/cublas/lib:.venv/lib/$PYVER/site-packages/nvidia/cudnn/lib"
.venv/bin/python ws_server.py
```

This is required because pip installs CUDA libraries under `.venv/lib/pythonX.Y/site-packages/nvidia/*/lib/` which is not a standard library search path. The exact directory name changes with the Python version (e.g., `python3.13`, `python3.12`).

## Why?

Cloud voice input services send your audio and screen content to external servers. This tool does everything locally on your own hardware — your voice, screen data, and all processing stay on your network.

## License

MIT
