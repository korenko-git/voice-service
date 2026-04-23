import asyncio, io, os, re, json, wave
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
LLM_URL            = os.getenv("LLM_URL",            "http://host.docker.internal:1234/v1")
LLM_MODEL          = os.getenv("LLM_MODEL",           "local-model")
WHISPER_PATH       = os.getenv("WHISPER_PATH",        "/app/models/faster-whisper-turbo")
WHISPER_COMPUTE    = os.getenv("WHISPER_COMPUTE_TYPE","float16")
F5_MODEL_PATH      = os.getenv("F5_MODEL_PATH",       "/app/models/f5tts-russian/model_last_inference.safetensors")
F5_VOCAB_PATH      = os.getenv("F5_VOCAB_PATH",       "/app/models/f5tts-russian/vocab.txt")
VOICES_DIR         = os.getenv("VOICES_DIR",          "/app/models/voices")
DEFAULT_VOICE      = os.getenv("DEFAULT_VOICE",       "default")
SYSTEM_PROMPT_FILE = os.getenv("SYSTEM_PROMPT_FILE",  "/app/system_prompt.txt")
REDIS_URL          = os.getenv("REDIS_URL",           "redis://redis:6379")
SESSION_TTL        = int(os.getenv("SESSION_TTL",     "3600"))
HISTORY_MAX_TURNS  = int(os.getenv("HISTORY_MAX_TURNS","20"))

SENTENCE_END = re.compile(r'(?<=[.!?])\s+')
MIN_SENTENCE_LEN = 5
models: dict = {}


# ── Voice helpers ─────────────────────────────────────────────────────────────

def _voice_ref(voice: str) -> tuple[str, str]:
    """Return (ref_wav, ref_text) for a voice profile, falling back to default."""
    d = os.path.join(VOICES_DIR, voice)
    if not os.path.isdir(d):
        d = os.path.join(VOICES_DIR, DEFAULT_VOICE)
    wav  = os.path.join(d, "ref.wav")
    txt  = os.path.join(d, "ref.txt")
    ref_text = open(txt, encoding="utf-8").read().strip() if os.path.exists(txt) else ""
    return wav, ref_text


# ── Redis helpers ─────────────────────────────────────────────────────────────

def _session_key(sid: str) -> str:
    return f"voice:session:{sid}"

async def session_get(r: aioredis.Redis, sid: str) -> dict:
    raw = await r.get(_session_key(sid))
    if raw:
        return json.loads(raw)
    return {"history": [], "context": "", "voice": DEFAULT_VOICE}

async def session_save(r: aioredis.Redis, sid: str, data: dict):
    await r.setex(_session_key(sid), SESSION_TTL, json.dumps(data, ensure_ascii=False))

async def session_delete(r: aioredis.Redis, sid: str):
    await r.delete(_session_key(sid))


# ── App lifespan ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("⏳ Loading Whisper...")
    models["whisper"] = WhisperModel(
        WHISPER_PATH,
        device="cuda",
        compute_type=WHISPER_COMPUTE,
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
        with open(SYSTEM_PROMPT_FILE, encoding="utf-8") as f:
            models["system_prompt"] = f.read().strip()
    else:
        models["system_prompt"] = os.getenv("SYSTEM_PROMPT", "Ты голосовой ассистент.")

    print("⏳ Connecting to Redis...")
    models["redis"] = aioredis.from_url(REDIS_URL, decode_responses=True)
    await models["redis"].ping()

    print("⏳ Warming up F5-TTS...")
    ref_wav, ref_text = _voice_ref(DEFAULT_VOICE)
    await asyncio.to_thread(
        lambda: models["f5tts"].infer(
            ref_file=ref_wav,
            ref_text=ref_text,
            gen_text="Привет.",
        )
    )

    print("✅ All systems ready")
    yield

    await models["redis"].aclose()
    models.clear()


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── REST endpoints ────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    redis_ok = False
    try:
        await models["redis"].ping()
        redis_ok = True
    except Exception:
        pass
    return {"status": "ok", "models": list(models.keys()), "redis": redis_ok}


@app.get("/voices")
async def list_voices():
    if not os.path.isdir(VOICES_DIR):
        return {"voices": []}
    voices = [
        name for name in sorted(os.listdir(VOICES_DIR))
        if os.path.isfile(os.path.join(VOICES_DIR, name, "ref.wav"))
    ]
    return {"voices": voices}


# ── Audio / TTS helpers ───────────────────────────────────────────────────────

def transcribe(audio_bytes: bytes) -> str:
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    
    if len(audio_np) < 16000 * 0.5:
        return ""
    
    segments, info = models["whisper"].transcribe(
        audio_np,
        language="ru",
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        no_speech_threshold=0.6,
        log_prob_threshold=-1.0,
    )
    
    result_parts = []
    for seg in segments:
        # Пропускаем сегменты с низкой уверенностью
        if seg.no_speech_prob > 0.6:
            continue
        if seg.avg_logprob < -1.0:
            continue
        result_parts.append(seg.text)
    
    return " ".join(result_parts).strip()

def synthesize(text: str, ref_wav: str, ref_text: str) -> bytes:
    text = text[:250]
    if len(text) < MIN_SENTENCE_LEN:
        return b""
    accented = models["accentor"](text)
    wav, sr, _ = models["f5tts"].infer(
        ref_file=ref_wav,
        ref_text=ref_text,
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

            # ── JSON frames ───────────────────────────────────────────
            if "text" in message:
                data  = json.loads(message["text"])
                mtype = data.get("type")

                if mtype == "init":
                    session_id = data["session_id"]
                    session = await session_get(r, session_id)
                    await ws.send_json({
                        "type":        "init_ack",
                        "session_id":  session_id,
                        "has_history": len(session["history"]) > 0,
                        "has_context": bool(session["context"]),
                        "voice":       session.get("voice", DEFAULT_VOICE),
                    })
                    continue

                if not session_id:
                    await ws.send_json({"type": "error", "text": "Send init first"})
                    continue

                if mtype == "context":
                    session = await session_get(r, session_id)
                    session["context"] = data.get("text", "")
                    session["history"] = []
                    await session_save(r, session_id, session)
                    await ws.send_json({"type": "context_ack"})
                    continue

                if mtype == "set_voice":
                    voice = data.get("voice", DEFAULT_VOICE)
                    voice_dir = os.path.join(VOICES_DIR, voice)
                    if os.path.isfile(os.path.join(voice_dir, "ref.wav")):
                        session = await session_get(r, session_id)
                        session["voice"] = voice
                        await session_save(r, session_id, session)
                        await ws.send_json({"type": "voice_ack", "voice": voice})
                    else:
                        await ws.send_json({"type": "error", "text": f"Voice '{voice}' not found"})
                    continue

                if mtype == "reset":
                    await session_delete(r, session_id)
                    await ws.send_json({"type": "reset_ack"})
                    continue

                if mtype == "get_history":
                    session = await session_get(r, session_id)
                    await ws.send_json({"type": "history", "messages": session["history"]})
                    continue

            # ── Binary frame (PCM audio) ──────────────────────────────
            elif "bytes" in message:
                if not session_id:
                    continue

                session  = await session_get(r, session_id)
                history: list = session["history"]
                context: str  = session["context"]
                voice:   str  = session.get("voice", DEFAULT_VOICE)
                ref_wav, ref_text = _voice_ref(voice)

                if len(history) > HISTORY_MAX_TURNS * 2:
                    history = history[-(HISTORY_MAX_TURNS * 2):]

                system_content = base_prompt + (f"\n\n{context}" if context else "")
                messages = [{"role": "system", "content": system_content}, *history]

                user_text = transcribe(message["bytes"])
                await ws.send_json({"type": "stt", "text": user_text})
                if not user_text:
                    continue

                messages.append({"role": "user", "content": user_text})
                history.append( {"role": "user", "content": user_text})

                sentence_buf = ""
                full_reply   = ""

                async for token in stream_llm(messages):
                    sentence_buf += token
                    full_reply   += token
                    await ws.send_json({"type": "token", "text": token})

                    parts = SENTENCE_END.split(sentence_buf)
                    if len(parts) > 1:
                        for sentence in parts[:-1]:
                            s = sentence.strip()
                            if len(s) >= MIN_SENTENCE_LEN:
                                wav = await asyncio.to_thread(synthesize, s, ref_wav, ref_text)
                                if wav:
                                    await ws.send_bytes(wav)
                        sentence_buf = parts[-1]

                if sentence_buf.strip() and len(sentence_buf.strip()) >= MIN_SENTENCE_LEN:
                    wav = await asyncio.to_thread(synthesize, sentence_buf.strip(), ref_wav, ref_text)
                    if wav:
                        await ws.send_bytes(wav)

                await ws.send_json({"type": "done"})

                history.append({"role": "assistant", "content": full_reply})
                session["history"] = history
                await session_save(r, session_id, session)

    except WebSocketDisconnect:
        pass
