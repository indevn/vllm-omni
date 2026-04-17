"""OpenAI-compatible client for CosyVoice3 via /v1/audio/speech endpoint.

This script demonstrates how to use the OpenAI-compatible speech API
to generate audio from text using CosyVoice3 voice-cloning model.

Examples:
    # Voice cloning with a remote reference audio
    python openai_speech_client.py \\
        --text "CosyVoice is undergoing a comprehensive upgrade." \\
        --ref-audio https://raw.githubusercontent.com/FunAudioLLM/CosyVoice/main/asset/zero_shot_prompt.wav \\
        --ref-text "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"

    # Voice cloning with a local reference audio
    python openai_speech_client.py \\
        --text "Hello, this is a cloned voice." \\
        --ref-audio /path/to/reference.wav \\
        --ref-text "Transcript of the reference audio." \\
        --output cloned.wav

    # Without reference audio (uses model default voice)
    python openai_speech_client.py --text "Hello world"
"""

import argparse
import base64
import os

import httpx

DEFAULT_API_BASE = "http://localhost:8091"
DEFAULT_API_KEY = "EMPTY"
DEFAULT_MODEL = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"


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


def run_tts_generation(args) -> None:
    """Run TTS generation via OpenAI-compatible /v1/audio/speech API."""
    payload = {
        "model": args.model,
        "input": args.text,
        "response_format": args.response_format,
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

    print(f"Model: {args.model}")
    print(f"Text: {args.text}")
    if args.ref_audio:
        print(f"Ref audio: {args.ref_audio[:60]}...")
    print("Generating audio...")

    api_url = f"{args.api_base}/v1/audio/speech"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }

    with httpx.Client(timeout=300.0) as client:
        response = client.post(api_url, json=payload, headers=headers)

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return

    try:
        text = response.content.decode("utf-8")
        if text.startswith('{"error"'):
            print(f"Error: {text}")
            return
    except UnicodeDecodeError:
        pass

    output_path = args.output or "cosyvoice3_output.wav"
    with open(output_path, "wb") as f:
        f.write(response.content)
    print(f"Audio saved to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible client for CosyVoice3 via /v1/audio/speech",
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
        "--response-format",
        type=str,
        default="wav",
        choices=["wav", "mp3", "flac", "pcm", "aac", "opus"],
        help="Audio output format (default: wav)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output audio file path (default: cosyvoice3_output.wav)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_tts_generation(args)
