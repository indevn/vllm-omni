# CosyVoice3 Online Serving

CosyVoice3 is served through the standard OpenAI-compatible speech endpoints.
It requires reference audio plus the transcript of that reference audio.

## Launch Server

```bash
vllm-omni serve FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
    --tokenizer FunAudioLLM/Fun-CosyVoice3-0.5B-2512/CosyVoice-BlankEN \
    --deploy-config vllm_omni/deploy/cosyvoice3.yaml \
    --port 8091 \
    --trust-remote-code \
    --omni
```

Or:

```bash
cd examples/online_serving/cosyvoice3
./run_server.sh async_chunk
```

The bundled `vllm_omni/deploy/cosyvoice3.yaml` defaults to `async_chunk: true`
for lower latency and streaming audio. Use `./run_server.sh sync`, or add
`--no-async-chunk` to the serve command, for the legacy sync pipeline.

## Non-Streaming Speech

```bash
python speech_client.py \
    --text "CosyVoice is undergoing a comprehensive upgrade." \
    --ref-audio https://raw.githubusercontent.com/FunAudioLLM/CosyVoice/main/asset/zero_shot_prompt.wav \
    --ref-text "希望你以后能够做的比我还好呦。" \
    --output cosyvoice3_output.wav
```

Equivalent curl request:

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "CosyVoice is undergoing a comprehensive upgrade.",
        "ref_audio": "https://raw.githubusercontent.com/FunAudioLLM/CosyVoice/main/asset/zero_shot_prompt.wav",
        "ref_text": "希望你以后能够做的比我还好呦。",
        "response_format": "wav"
    }' \
    --output cosyvoice3_output.wav
```

## HTTP Streaming PCM

The `/v1/audio/speech` endpoint can stream raw PCM chunks when the server is
launched in the default async-chunk mode:

```bash
curl -s -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "CosyVoice is undergoing a comprehensive upgrade.",
        "ref_audio": "https://raw.githubusercontent.com/FunAudioLLM/CosyVoice/main/asset/zero_shot_prompt.wav",
        "ref_text": "希望你以后能够做的比我还好呦。",
        "stream": true,
        "response_format": "pcm"
    }' | play -t raw -r 24000 -e signed -b 16 -c 1 -
```

## WebSocket Streaming Text Input

The `/v1/audio/speech/stream` endpoint accepts text incrementally and generates
audio per sentence. It is useful when upstream text arrives gradually, for
example from ASR.

```bash
python streaming_speech_client.py \
    --text "CosyVoice is undergoing a comprehensive upgrade. Streaming starts sentence by sentence." \
    --ref-audio https://raw.githubusercontent.com/FunAudioLLM/CosyVoice/main/asset/zero_shot_prompt.wav \
    --ref-text "希望你以后能够做的比我还好呦。" \
    --output-dir cosyvoice3_streaming_output
```

To receive progressive PCM frames for each sentence:

```bash
python streaming_speech_client.py \
    --text "CosyVoice is undergoing a comprehensive upgrade. Streaming starts sentence by sentence." \
    --ref-audio https://raw.githubusercontent.com/FunAudioLLM/CosyVoice/main/asset/zero_shot_prompt.wav \
    --ref-text "希望你以后能够做的比我还好呦。" \
    --response-format pcm \
    --stream-audio
```

WebSocket message flow:

```json
{"type": "session.config", "ref_audio": "...", "ref_text": "...", "response_format": "pcm", "stream_audio": true}
{"type": "input.text", "text": "CosyVoice is undergoing a comprehensive upgrade."}
{"type": "input.done"}
```

Server responses include `audio.start`, one or more binary audio frames,
`audio.done`, and `session.done`.
