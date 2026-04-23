import asyncio, io, os, re, json, wave, time
import numpy as np
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from faster_whisper import WhisperModel
from silero_stress import load_accentor
from f5_tts.api import F5TTS
import redis.asyncio as aioredis

# ── Config ────────────────────────────────────────────────────────────────────
LLM_URL           = os.getenv("LLM_URL",           "http://host.docker.internal:1234/v1")
LLM_MODEL         = os.getenv("LLM_MODEL",          "local-model")
WHISPER_PATH      = os.getenv("WHISPER_PATH",       "/app/models/whisper-large-v3-turbo.gguf")
F5_MODEL_PATH     = os.getenv("F5_MODEL_PATH",      "/app/models/f5tts-russian/model_20000_inference.safetensors")
F5_VOCAB_PATH     = os.getenv("F5_VOCAB_PATH",      "/app/models/f5tts-russian/vocab.txt")
F5_REF_AUDIO      = os.getenv("F5_REF_AUDIO",       "/app/models/ref.wav")
F5_REF_TEXT       = os.getenv("F5_REF_TEXT",        "Привет, это референсный голос.")
SYSTEM_PROMPT_FILE= os.getenv("SYSTEM_PROMPT_FILE", "/app/system_prompt.txt")
REDIS_URL         = os.getenv("REDIS_URL",          "redis://redis:6379")
SESSION_TTL       = int(os.getenv("SESSION_TTL",    "3600"))   # секунды
HISTORY_MAX_TURNS = int(os.getenv("HISTORY_MAX_TURNS", "20"))  # макс. пар user/assistant

SENTENCE_END = re.compile(r'(?<=[.!?…,;])\s+')
models: dict = {}

# ── Redis helpers ─────────────────────────────────────────────────────────────

def _session_key(session_id: str) -> str:
    return f"voice:session:{session_id}"

async def session_get(r: aioredis.Redis, session_id: str) -> dict:
    raw = await r.get(_session_key(session_id))
    if raw:
        return json.loads(raw)
    return {"history": [], "context": ""}

async def session_save(r: aioredis.Redis, session_id: str, data: dict):
    await r.setex(_session_key(session_id), SESSION_TTL, json.dumps(data, ensure_ascii=False))

async def session_delete(r: aioredis.Redis, session_id: str):
    await r.delete(_session_key(session_id))

# ── App lifespan ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("⏳ Loading Whisper...")
    models["whisper"] = WhisperModel(
        WHISPER_PATH,
        device="cuda",
        compute_type="float16",
    )

    print("⏳ Loading silero-stress...")
    models["accentor"] = load_accentor()

    print("⏳ Loading F5-TTS...")
    models["f5tts"] = F5TTS(
        model="F5TTS_v1_Base",
        ckpt_file=F5_MODEL_PATH,
        vocab_file=F5_VOCAB_PATH,
        device="cuda",
    )

    if os.path.exists(SYSTEM_PROMPT_FILE):
        with open(SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
            models["system_prompt"] = f.read().strip()
    else:
        models["system_prompt"] = os.getenv("SYSTEM_PROMPT", "Ты голосовой ассистент.")

    print("⏳ Connecting to Redis...")
    models["redis"] = aioredis.from_url(REDIS_URL, decode_responses=True)
    await models["redis"].ping()

    print("✅ All systems ready")
    yield

    await models["redis"].aclose()
    models.clear()

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Audio / TTS helpers ───────────────────────────────────────────────────────

def transcribe(audio_bytes: bytes) -> str:
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    segments, _ = models["whisper"].transcribe(
        audio_np,
        language="ru",
        beam_size=5,
        vad_filter=True,       
        vad_parameters=dict(min_silence_duration_ms=500),
    )
    return " ".join(s.text for s in segments).strip()

def synthesize(text: str) -> bytes:
    if len(text) > 150:
        text = text[:150]
    accented = models["accentor"](text)
    wav, sr, _ = models["f5tts"].infer(
        ref_file=F5_REF_AUDIO,
        ref_text=F5_REF_TEXT,
        gen_text=accented,
    )
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((wav * 32767).astype(np.int16).tobytes())
    return buf.getvalue()

async def stream_llm(messages: list):
    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream("POST", f"{LLM_URL}/chat/completions", json={
            "model": LLM_MODEL, "messages": messages,
            "stream": True, "temperature": 0.7, "max_tokens": 512,
        }) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    token = json.loads(data)["choices"][0]["delta"].get("content", "")
                    if token:
                        yield token
                except Exception:
                    pass

# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def voice_chat(ws: WebSocket):
    await ws.accept()
    r: aioredis.Redis = models["redis"]
    base_prompt = models["system_prompt"]
    session_id: str | None = None

    try:
        while True:
            message = await ws.receive()

            # ── JSON-фреймы ───────────────────────────────────────────
            if "text" in message:
                data = json.loads(message["text"])
                mtype = data.get("type")

                # Инициализация / реконнект
                if mtype == "init":
                    session_id = data["session_id"]
                    session = await session_get(r, session_id)
                    await ws.send_json({
                        "type": "init_ack",
                        "session_id": session_id,
                        "has_history": len(session["history"]) > 0,
                        "has_context": bool(session["context"]),
                    })
                    continue

                if not session_id:
                    await ws.send_json({"type": "error", "text": "Send init first"})
                    continue

                # Контекст персонажа
                if mtype == "context":
                    session = await session_get(r, session_id)
                    session["context"] = data.get("text", "")
                    session["history"] = []  # смена персонажа = новая история
                    await session_save(r, session_id, session)
                    await ws.send_json({"type": "context_ack"})
                    continue

                # Сброс истории
                if mtype == "reset":
                    await session_delete(r, session_id)
                    await ws.send_json({"type": "reset_ack"})
                    continue

                # Получить текущую историю (для отображения в UI)
                if mtype == "get_history":
                    session = await session_get(r, session_id)
                    await ws.send_json({"type": "history", "messages": session["history"]})
                    continue

            # ── Бинарный фрейм (аудио PCM) ───────────────────────────
            elif "bytes" in message:
                if not session_id:
                    continue

                session = await session_get(r, session_id)
                history: list = session["history"]
                context: str  = session["context"]

                # Обрезаем историю до последних N пар
                if len(history) > HISTORY_MAX_TURNS * 2:
                    history = history[-(HISTORY_MAX_TURNS * 2):]

                system_content = base_prompt
                if context:
                    system_content += f"\n\n{context}"

                messages = [{"role": "system", "content": system_content}, *history]

                # STT
                user_text = transcribe(message["bytes"])
                await ws.send_json({"type": "stt", "text": user_text})
                if not user_text:
                    continue

                messages.append({"role": "user", "content": user_text})
                history.append({"role": "user", "content": user_text})

                sentence_buf = ""
                full_reply   = ""

                # LLM → TTS стриминг
                async for token in stream_llm(messages):
                    sentence_buf += token
                    full_reply   += token
                    await ws.send_json({"type": "token", "text": token})

                    parts = SENTENCE_END.split(sentence_buf)
                    if len(parts) > 1:
                        for sentence in parts[:-1]:
                            s = sentence.strip()
                            if s:
                                wav = await asyncio.to_thread(synthesize, s)
                                await ws.send_bytes(wav)
                        sentence_buf = parts[-1]

                if sentence_buf.strip():
                    wav = await asyncio.to_thread(synthesize, sentence_buf.strip())
                    await ws.send_bytes(wav)

                await ws.send_json({"type": "done"})

                # Сохраняем обновлённую историю в Redis
                history.append({"role": "assistant", "content": full_reply})
                session["history"] = history
                await session_save(r, session_id, session)

    except WebSocketDisconnect:
        pass  # история сохранена в Redis, сессия живёт SESSION_TTL секунд


@app.get("/health")
async def health():
    redis_ok = False
    try:
        await models["redis"].ping()
        redis_ok = True
    except Exception:
        pass
    return {"status": "ok", "models": list(models.keys()), "redis": redis_ok}