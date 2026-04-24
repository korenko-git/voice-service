# Models Folder

This folder contains all runtime assets that are too large or too local to keep in git:

- Whisper model files for `faster-whisper`
- F5-TTS checkpoint files
- Voice profiles with reference audio and transcripts

## Expected Layout

```text
models/
├── README.md
├── faster-whisper-turbo/
│   ├── config.json
│   ├── model.bin
│   ├── preprocessor_config.json
│   ├── tokenizer.json
│   └── vocabulary.json
├── f5tts-russian/
│   ├── model_last_inference.safetensors
│   └── vocab.txt
└── voices/
    ├── default/
    │   ├── ref.wav
    │   └── ref.txt
    ├── female_1/
    │   ├── ref.wav
    │   └── ref.txt
    └── ...
```

## Whisper

`faster-whisper` expects a CTranslate2 model directory.

Download it with:

```bash
pip install huggingface-hub
huggingface-cli download Systran/faster-whisper-large-v3 \
  --local-dir ./models/faster-whisper-turbo \
  --local-dir-use-symlinks False
```

The service points to this directory by default:

```ini
WHISPER_PATH=/app/models/faster-whisper-turbo
```

## F5-TTS

Source:

- [Misha24-10/F5-TTS_RUSSIAN](https://huggingface.co/Misha24-10/F5-TTS_RUSSIAN/tree/main/F5TTS_v1_Base_accent_tune)

Put these files into `models/f5tts-russian/`:

- `model_last_inference.safetensors`
- `vocab.txt`

The service uses them through:

```ini
F5_MODEL_PATH=/app/models/f5tts-russian/model_last_inference.safetensors
F5_VOCAB_PATH=/app/models/f5tts-russian/vocab.txt
```

## Voices

Each voice is a separate subfolder inside `models/voices/`.

Minimum required profile:

```text
models/voices/default/
├── ref.wav
└── ref.txt
```

Rules:

- `default/` must exist, because it is used as the startup fallback and warmup voice
- `ref.wav` should be a clean single-speaker reference sample
- `ref.txt` must contain the exact transcript of `ref.wav`

Recommended audio format:

- WAV
- 24 kHz
- mono
- 5 to 12 seconds

Example conversion:

```bash
ffmpeg -i input.mp3 -t 10 -ar 24000 -ac 1 -sample_fmt s16 models/voices/default/ref.wav
```

Example transcript creation:

```bash
echo "Текст того, что говорится в записи." > models/voices/default/ref.txt
```

To add another voice, create another folder:

```text
models/voices/female_1/
├── ref.wav
└── ref.txt
```

The API exposes available voices via `GET /voices`, and the client can switch them with the WebSocket message:

```json
{ "type": "set_voice", "voice": "female_1" }
```
