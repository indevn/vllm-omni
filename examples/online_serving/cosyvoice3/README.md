# CosyVoice3

## Installation

Please refer to [README.md](https://github.com/vllm-project/vllm-omni/tree/main/README.md)

## Supported Models

| Model | Description |
| --- | --- |
| `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` | CosyVoice3 0.5B voice-cloning model |

## Run examples (CosyVoice3)

### Launch the Server

```bash
# Async-chunk mode (recommended — enables streaming, lower TTFA)
vllm-omni serve FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
    --tokenizer FunAudioLLM/Fun-CosyVoice3-0.5B-2512/CosyVoice-BlankEN \
    --stage-configs-path vllm_omni/model_executor/stage_configs/cosyvoice3_async_chunk.yaml \
    --port 8091 \
    --trust-remote-code \
    --omni

# Sync mode (non-streaming, returns complete audio)
vllm-omni serve FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
    --tokenizer FunAudioLLM/Fun-CosyVoice3-0.5B-2512/CosyVoice-BlankEN \
    --stage-configs-path vllm_omni/model_executor/stage_configs/cosyvoice3.yaml \
    --port 8091 \
    --trust-remote-code \
    --omni
```

Alternatively, use the convenience script:

```bash
./run_server.sh              # async_chunk mode (default)
./run_server.sh async_chunk  # async_chunk mode
./run_server.sh sync         # sync mode
```

### Send TTS Request

Get into the example folder:

```bash
cd examples/online_serving/cosyvoice3
```

#### Non-streaming request (returns a complete WAV file)

```bash
# Voice cloning with a remote reference audio
python openai_speech_client.py \
    --text "CosyVoice is undergoing a comprehensive upgrade." \
    --ref-audio https://raw.githubusercontent.com/FunAudioLLM/CosyVoice/main/asset/zero_shot_prompt.wav \
    --ref-text "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"

# Voice cloning with a local reference audio
python openai_speech_client.py \
    --text "Hello, this is a cloned voice." \
    --ref-audio /path/to/reference.wav \
    --ref-text "Transcript of the reference audio." \
    --output cloned.wav
```

#### Streaming request (PCM chunks, lowest latency)

Requires the server to be running in `async_chunk` mode.

```bash
# Stream with a remote reference audio
python streaming_speech_client.py \
    --text "CosyVoice is undergoing a comprehensive upgrade." \
    --ref-audio https://raw.githubusercontent.com/FunAudioLLM/CosyVoice/main/asset/zero_shot_prompt.wav \
    --ref-text "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"

# Stream and play in real time with sox
curl -s -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "CosyVoice is undergoing a comprehensive upgrade.",
        "ref_audio": "https://raw.githubusercontent.com/FunAudioLLM/CosyVoice/main/asset/zero_shot_prompt.wav",
        "ref_text": "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。",
        "stream": true,
        "response_format": "pcm"
    }' | play -t raw -r 24000 -e signed -b 16 -c 1 -
```

#### Send request via curl

```bash
# Non-streaming WAV
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "CosyVoice is undergoing a comprehensive upgrade.",
        "ref_audio": "https://raw.githubusercontent.com/FunAudioLLM/CosyVoice/main/asset/zero_shot_prompt.wav",
        "ref_text": "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"
    }' --output output.wav
```

### Using Python httpx

```python
import httpx

# Non-streaming
response = httpx.post(
    "http://localhost:8091/v1/audio/speech",
    json={
        "input": "CosyVoice is undergoing a comprehensive upgrade.",
        "ref_audio": "https://raw.githubusercontent.com/FunAudioLLM/CosyVoice/main/asset/zero_shot_prompt.wav",
        "ref_text": "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。",
    },
    timeout=300.0,
)
with open("output.wav", "wb") as f:
    f.write(response.content)

# Streaming (PCM chunks, 24 kHz mono 16-bit signed)
with httpx.stream(
    "POST",
    "http://localhost:8091/v1/audio/speech",
    json={
        "input": "CosyVoice is undergoing a comprehensive upgrade.",
        "ref_audio": "https://raw.githubusercontent.com/FunAudioLLM/CosyVoice/main/asset/zero_shot_prompt.wav",
        "ref_text": "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。",
        "stream": True,
        "response_format": "pcm",
    },
    timeout=300.0,
) as response:
    for chunk in response.iter_bytes():
        # Each chunk is raw 16-bit signed PCM at 24 kHz mono.
        pass
```

## API Reference

### Endpoint

```
POST /v1/audio/speech
Content-Type: application/json
```

### Request Body

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `input` | string | **Yes** | Text to synthesize |
| `ref_audio` | string | No | Reference audio: http(s) URL or base64 data URL (`data:audio/wav;base64,...`) |
| `ref_text` | string | No | Transcript of the reference audio (enables ICL voice cloning) |
| `stream` | bool | No | Stream raw PCM chunks (requires `response_format="pcm"`) |
| `response_format` | string | No | `wav` (default), `mp3`, `flac`, `pcm` |
| `model` | string | No | Model name (optional when serving a single model) |

### Response

- Non-streaming: binary audio data with `Content-Type: audio/wav` (or the requested format).
- Streaming (`stream=true`): chunked `audio/pcm` — raw 16-bit signed PCM at 24 kHz mono.

## Streaming

Set `stream=true` with `response_format="pcm"` to receive raw PCM audio chunks as they are decoded
(one chunk per Code2Wav window, default 25 frames ≈ 50 ms):

**Constraints:**
- `stream=true` requires `response_format="pcm"`.
- Requires the server to be launched with `cosyvoice3_async_chunk.yaml`.

## Troubleshooting

1. **No audio output**: Verify the model loaded correctly and the stage config path is correct.
2. **Connection refused**: Check the server is running on the expected port.
3. **ref_audio fetch fails**: Ensure the URL is publicly accessible or use a base64 data URL for local files.
4. **Streaming not working**: Confirm the server uses `cosyvoice3_async_chunk.yaml`, not `cosyvoice3.yaml`.
