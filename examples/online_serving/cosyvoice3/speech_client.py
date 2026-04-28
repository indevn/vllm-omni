"""Client for CosyVoice3 via the /v1/audio/speech endpoint."""

import argparse
import base64
from pathlib import Path

import httpx

DEFAULT_API_BASE = "http://localhost:8091"
DEFAULT_API_KEY = "EMPTY"
DEFAULT_MODEL = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"


def encode_audio_to_base64(audio_path: str) -> str:
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    suffix = path.suffix.lower()
    if suffix == ".wav":
        mime_type = "audio/wav"
    elif suffix in (".mp3", ".mpeg"):
        mime_type = "audio/mpeg"
    elif suffix == ".flac":
        mime_type = "audio/flac"
    elif suffix == ".ogg":
        mime_type = "audio/ogg"
    else:
        mime_type = "audio/wav"

    audio_b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{audio_b64}"


def normalize_ref_audio(ref_audio: str) -> str:
    if ref_audio.startswith(("http://", "https://", "data:", "file:")):
        return ref_audio
    return encode_audio_to_base64(ref_audio)


def main() -> None:
    parser = argparse.ArgumentParser(description="CosyVoice3 /v1/audio/speech client")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL)
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--ref-audio", required=True, help="Reference audio path, URL, data URL, or file URI")
    parser.add_argument("--ref-text", required=True, help="Transcript of the reference audio")
    parser.add_argument(
        "--response-format",
        default="wav",
        choices=["wav", "mp3", "flac", "pcm", "aac", "opus"],
    )
    parser.add_argument("--output", "-o", default="cosyvoice3_output.wav")
    args = parser.parse_args()

    payload = {
        "model": args.model,
        "input": args.text,
        "ref_audio": normalize_ref_audio(args.ref_audio),
        "ref_text": args.ref_text,
        "response_format": args.response_format,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }

    url = f"{args.api_base}/v1/audio/speech"
    with httpx.Client(timeout=300.0, trust_env=False) as client:
        response = client.post(url, json=payload, headers=headers)

    if response.status_code != 200:
        raise SystemExit(f"Error {response.status_code}: {response.text}")

    Path(args.output).write_bytes(response.content)
    print(f"Audio saved to: {args.output}")


if __name__ == "__main__":
    main()
