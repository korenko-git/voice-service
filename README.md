# Voice Chat Service

Self-contained Docker service for real-time voice chat via WebSockets. Full pipeline: Speech-to-Text → LLM → Stress Accentuation → Text-to-Speech, optimized for low perceived latency via streaming synthesis. F5-TTS with Russian language support, OpenAI-compatible LLM backend, multi-voice profiles, Redis sessions.

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
│  4. F5-TTS                              │  TTS
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
└── models/                       ← models and voice profiles, see models/README.md
```

### 2. Create `.env`

```bash
cp .env.example .env
```

Edit `.env` with your values. Every variable is documented directly in [.env.example](.env.example).

### 3. Prepare models folder

All model downloads, file layout, and voice profile instructions were moved to [models/README.md](models/README.md).

At minimum you need:

- `models/faster-whisper-turbo/` with the CTranslate2 Whisper model
- `models/f5tts-russian/` with `model_last_inference.safetensors` and `vocab.txt`
- `models/voices/default/` with `ref.wav` and `ref.txt`

### 4. Build and run

```bash
docker compose up --build
```

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

The service supports multiple voice profiles stored in `models/voices/`. Full folder layout and preparation tips are documented in [models/README.md](models/README.md).

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
- [Misha24-10/F5-TTS_RUSSIAN](https://huggingface.co/Misha24-10/F5-TTS_RUSSIAN/tree/main/F5TTS_v1_Base_accent_tune) — F5-TTS with russian language support

## License

[MIT License](LICENSE.md)
