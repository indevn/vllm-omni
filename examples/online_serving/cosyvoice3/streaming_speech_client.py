"""Streaming client for CosyVoice3 via /v1/audio/speech with stream=True.

Connects to the /v1/audio/speech endpoint with stream=True and response_format=pcm,
receives raw 16-bit signed PCM chunks at 24 kHz mono as they are decoded, and
saves the result as a WAV file.

Requires the server to be running with cosyvoice3_async_chunk.yaml stage config.

Examples:
    # Stream with a remote reference audio
    python streaming_speech_client.py \\
        --text "CosyVoice is undergoing a comprehensive upgrade." \\
        --ref-audio https://raw.githubusercontent.com/FunAudioLLM/CosyVoice/main/asset/zero_shot_prompt.wav \\
        --ref-text "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"

    # Stream with a local reference audio
    python streaming_speech_client.py \\
        --text "Hello, this is a cloned voice." \\
        --ref-audio /path/to/reference.wav \\
        --ref-text "Transcript of the reference audio." \\
        --output stream_output.wav
"""

import argparse
import base64
import os
import struct
import sys

import httpx

DEFAULT_API_BASE = "http://localhost:8091"
DEFAULT_API_KEY = "EMPTY"
DEFAULT_MODEL = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"

# CosyVoice3 outputs 24 kHz mono 16-bit PCM
SAMPLE_RATE = 24000


def encode_audio_to_base64(audio_path: str) -> str:
    """Encode a local audio file to base64 data URL."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    audio_path_lower = audio_path.lower()
    if audio_path_lower.endswith(".wav"):
        mime_type = "audio/wav"
    elif audio_path_lower.endswith((".mp3", ".mpeg")):
        mime_type = "audio/mpeg"
    elif audio_path_lower.endswith(".flac"):
        mime_type = "audio/flac"
    elif audio_path_lower.endswith(".ogg"):
        mime_type = "audio/ogg"
    else:
        mime_type = "audio/wav"

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{audio_b64}"


def _write_wav_header(f, num_samples: int, sample_rate: int = SAMPLE_RATE) -> None:
    """Write a WAV header for 16-bit mono PCM to an open file."""
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = num_samples * block_align
    f.write(
        struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            36 + data_size,
            b"WAVE",
            b"fmt ",
            16,
            1,  # PCM
            num_channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
            b"data",
            data_size,
        )
    )


def run_streaming(args) -> None:
    """Stream PCM chunks from the server and save as a WAV file."""
    payload: dict = {
        "model": args.model,
        "input": args.text,
        "stream": True,
        "response_format": "pcm",
    }

    if args.ref_audio:
        if args.ref_audio.startswith(("http://", "https://")):
            payload["ref_audio"] = args.ref_audio
        elif args.ref_audio.startswith("data:"):
            payload["ref_audio"] = args.ref_audio
        else:
            payload["ref_audio"] = encode_audio_to_base64(args.ref_audio)
    if args.ref_text:
        payload["ref_text"] = args.ref_text

    api_url = f"{args.api_base}/v1/audio/speech"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }

    print(f"Streaming from {api_url} ...")
    if args.ref_audio:
        print(f"Ref audio: {args.ref_audio[:60]}...")

    pcm_chunks: list[bytes] = []
    chunk_count = 0

    with httpx.stream("POST", api_url, json=payload, headers=headers, timeout=300.0) as response:
        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.read().decode()}", file=sys.stderr)
            return
        for chunk in response.iter_bytes():
            if chunk:
                pcm_chunks.append(chunk)
                chunk_count += 1
                sys.stdout.write(f"\r  Received chunk {chunk_count} ({sum(len(c) for c in pcm_chunks)} bytes total)")
                sys.stdout.flush()

    print()

    if not pcm_chunks:
        print("No audio received.", file=sys.stderr)
        return

    pcm_data = b"".join(pcm_chunks)
    num_samples = len(pcm_data) // 2  # 16-bit = 2 bytes per sample
    output_path = args.output or "cosyvoice3_stream_output.wav"

    with open(output_path, "wb") as f:
        _write_wav_header(f, num_samples)
        f.write(pcm_data)

    duration = num_samples / SAMPLE_RATE
    print(f"Saved {duration:.2f}s of audio ({chunk_count} chunks) to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Streaming client for CosyVoice3 via /v1/audio/speech",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=DEFAULT_API_BASE,
        help=f"API base URL (default: {DEFAULT_API_BASE})",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=DEFAULT_API_KEY,
        help="API key (default: EMPTY)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model name/path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text to synthesize",
    )
    parser.add_argument(
        "--ref-audio",
        type=str,
        default=None,
        help="Reference audio: local file path, http(s) URL, or base64 data URL",
    )
    parser.add_argument(
        "--ref-text",
        type=str,
        default=None,
        help="Transcript of the reference audio (enables ICL voice cloning)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output WAV file path (default: cosyvoice3_stream_output.wav)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_streaming(args)
