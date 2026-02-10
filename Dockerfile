FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv curl \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama (LLM backend)
RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app

COPY requirements.txt .
RUN python3 -m venv /app/.venv \
    && /app/.venv/bin/pip install --no-cache-dir -r requirements.txt

COPY voice_input.py ws_server.py transcribe.py ./

ENV PATH="/app/.venv/bin:$PATH"

# Whisper model is downloaded on first run (~1.5GB)
# Ollama models need to be pulled separately

EXPOSE 8991

# Startup script: start Ollama, pull models, then run WebSocket server
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["--port", "8991"]
