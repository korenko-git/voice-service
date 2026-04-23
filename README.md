# Voice Chat Service

A self-contained Docker service for real-time Russian voice chat using WebSockets. Implements a full pipeline: **Speech-to-Text → LLM → Stress Accentuation → Text-to-Speech**, optimized for low perceived latency via streaming synthesis.

## Pipeline

```
Client (WebSocket)
    │
    │  PCM audio (16kHz mono int16)
    ▼
┌─────────────────────────────────────────┐
│            Voice Service :8000          │
│                                         │
│  1. faster-whisper Large v3 Turbo       │  STT (CUDA)
│         ↓                               │
│  2. OpenAI-compatible LLM API           │  LLM
│         ↓ token by token                │
│  3. silero-stress                       │  Accentuation
│         ↓                               │
│  4. F5-TTS (Russian model)              │  TTS
└─────────────────────────────────────────┘
    │
    │  JSON tokens  +  WAV chunks
    ▼
Client
```

### Low-Latency Strategy

LLM output is split into sentences in real time. Each sentence is sent to TTS immediately after the period/exclamation/question mark — the client starts playing audio before the LLM has finished generating the full response.

***

## Requirements

- Docker + Docker Compose
- NVIDIA GPU with CUDA 12.1+
- Any **OpenAI-compatible LLM API** accessible from the container (LM Studio, Ollama, vLLM, llama.cpp server, OpenAI, etc.)

***

## Setup

### 1. Project structure

```
voice-service/
├── main.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env                      ← create from .env.example
├── .env.example
└── models/
    ├── faster-whisper-turbo/     ← CTranslate2 model (download below)
    ├── f5tts-russian/
    │   ├── model_last_inference.safetensors
    │   └── vocab.txt
    └── voices/                   ← voice profiles (see Voice Selection)
        └── default/
            ├── ref.wav           ← reference audio (5–12s, WAV 24kHz mono)
            └── ref.txt           ← exact transcript of ref.wav
```

### 2. Create `.env`

```bash
cp .env.example .env
```

Edit `.env` with your values — see the [Environment Variables](#environment-variables) section.

### 3. Download Whisper model

`faster-whisper` requires CTranslate2 format. Download with:

```bash
pip install huggingface-hub
huggingface-cli download Systran/faster-whisper-large-v3 \
  --local-dir ./models/faster-whisper-turbo \
  --local-dir-use-symlinks False
```

### 4. Download F5-TTS Russian model

```
👉 https://huggingface.co/Misha24-10/F5-TTS_RUSSIAN/tree/main/F5TTS_v1_Base_accent_tune
```

Download `model_last_inference.safetensors` and `vocab.txt`, place into `models/f5tts-russian/`.

### 5. Prepare reference audio

Place reference voice files in `models/voices/default/`:

```bash
# Convert from mp3, trim to 10 seconds
ffmpeg -i input.mp3 -t 10 -ar 24000 -ac 1 -sample_fmt s16 models/voices/default/ref.wav

# Write the exact transcript
echo "Текст того, что говорится в записи." > models/voices/default/ref.txt
```

### 6. Build and run

```bash
docker compose up --build
```

***

## Environment Variables

All configuration lives in `.env`. The `.env.example` file documents every variable:

```ini
# LLM — any OpenAI-compatible endpoint
LLM_URL=http://host.docker.internal:1234/v1
LLM_MODEL=local-model

# Whisper
WHISPER_PATH=/app/models/faster-whisper-turbo
WHISPER_COMPUTE_TYPE=float16        # float16 | int8 (int8 saves ~700MB VRAM)

# F5-TTS
F5_MODEL_PATH=/app/models/f5tts-russian/model_last_inference.safetensors
F5_VOCAB_PATH=/app/models/f5tts-russian/vocab.txt
VOICES_DIR=/app/models/voices
DEFAULT_VOICE=default

# LLM behaviour
SYSTEM_PROMPT_FILE=/app/system_prompt.txt

# Session / Redis
SESSION_TTL=3600
HISTORY_MAX_TURNS=20
```

> **LLM endpoint note:** if running LM Studio or Ollama locally on Windows/macOS, use `host.docker.internal` as the hostname. For remote or cloud endpoints use the full URL. The service speaks plain OpenAI chat completions API — any compatible backend works.

***

## WebSocket API

**Endpoint:** `ws://localhost:8000/ws`

### Session lifecycle

```
Client                          Server
  │── { type: "init", session_id }──▶│
  │◀── { type: "init_ack", has_history, has_context }──│
  │
  │  (optional) set character context:
  │── { type: "context", text: "..." }──▶│
  │◀── { type: "context_ack" }──│
  │
  │  (optional) select voice:
  │── { type: "set_voice", voice: "alice" }──▶│
  │◀── { type: "voice_ack", voice: "alice" }──│
  │
  │  (optional) restore history in UI:
  │── { type: "get_history" }──▶│
  │◀── { type: "history", messages: [...] }──│
```

### Audio turn

Send one complete utterance as a **binary frame** (raw PCM):

| Format | Value |
|---|---|
| Encoding | PCM signed 16-bit little-endian |
| Sample rate | 16000 Hz |
| Channels | Mono |

The server responds with interleaved frames:

**JSON frames** (text):

```json
{ "type": "stt",   "text": "Привет, как дела?" }
{ "type": "token", "text": "Отлично" }
{ "type": "done" }
```

**Binary frames**: WAV audio chunks (one per sentence). Play each chunk as it arrives for lowest latency.

### Control messages

```json
{ "type": "reset" }        // clear history for this session
{ "type": "get_history" }  // fetch full message history
{ "type": "set_voice", "voice": "alice" }
```

***

## Voice Selection

The service supports multiple voice profiles. Each profile is a folder inside `models/voices/` containing a reference WAV and its transcript:

```
models/voices/
├── default/
│   ├── ref.wav
│   └── ref.txt
├── alice/
│   ├── ref.wav
│   └── ref.txt
└── narrator/
    ├── ref.wav
    └── ref.txt
```

### REST endpoint

```bash
# List available voices
curl http://localhost:8000/voices
# { "voices": ["alice", "default", "narrator"] }
```

### Selecting a voice via WebSocket

Send `set_voice` before or during a session. The selected voice persists for the entire session and is stored in Redis:

```json
{ "type": "set_voice", "voice": "alice" }
```

If the requested voice does not exist the server responds with `{ "type": "error", "text": "Voice 'alice' not found" }` and keeps the current voice.

***

## Health Check

```bash
curl http://localhost:8000/health
# {"status":"ok","models":["whisper","accentor","f5tts","redis"],"redis":true}
```

***

## VRAM Usage

| Model | float16 | int8 |
|---|---|---|
| faster-whisper Large v3 Turbo | ~1.5 GB | ~800 MB |
| F5-TTS Russian | ~2.5 GB | — |
| LLM backend (varies) | 4–16 GB | — |

Set `WHISPER_COMPUTE_TYPE=int8` in `.env` to reduce Whisper memory at minimal quality cost.

***

## Related Projects

- [SWivid/F5-TTS](https://github.com/SWivid/F5-TTS) — base F5-TTS implementation
- [snakers4/silero-stress](https://github.com/snakers4/silero-stress) — stress accentuation

## HuggingFace Models Used

- [Systran/faster-whisper-large-v3](https://huggingface.co/Systran/faster-whisper-large-v3) — CTranslate2 Whisper Large v3
- [Misha24-10/F5-TTS_RUSSIAN](https://huggingface.co/Misha24-10/F5-TTS_RUSSIAN/tree/main/F5TTS_v1_Base_accent_tune) — Russian F5-TTS model