#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit CosyVoice3 local quality against upstream reference audio.

This script focuses on user-perceived quality impact:

1. Generate upstream reference audio for fixed seeds.
2. Generate local sync audio with PR-aligned sampling.
3. Transcribe both with Whisper.
4. Compare text completeness, durations, and end-tail behavior.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import whisper
from scipy.io import wavfile
from vllm.sampling_params import SamplingParams

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.model_executor.models.cosyvoice3.tokenizer import get_qwen_tokenizer

MODEL = Path("/home/fz/workspace/llm_tts/models/Fun-CosyVoice3-0.5B")
PROMPT_WAV = Path(
    "/home/fz/workspace/vllm-omni-cosyvoice3/vllm-omni/artifacts/cosyvoice3_chunk_boundary/zero_shot_prompt.wav"
)
DEFAULT_STAGE_CONFIG = Path("/tmp/cosyvoice3_sync_tokenizer_lowmem.yaml")
DEFAULT_OUTPUT_DIR = Path("artifacts/cosyvoice3_quality_audit")
PROMPT_TEXT = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"
SYNTH_TEXT = (
    "CosyVoice is undergoing a comprehensive upgrade, providing more accurate, "
    "stable, faster, and better voice generation capabilities."
)
STOP_TOKEN_ID = 6562


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit CosyVoice3 local quality against upstream audio")
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--prompt-wav", type=Path, default=PROMPT_WAV)
    parser.add_argument("--stage-config", type=Path, default=DEFAULT_STAGE_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-upstream", action="store_true")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[20260325, 20260326, 20260327, 20260328, 20260329],
    )
    parser.add_argument("--whisper-model", default="base")
    return parser.parse_args()


def load_prompt_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    return np.asarray(audio, dtype=np.float32), int(sr)


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    wavfile.write(path, sr, (clipped * 32767.0).astype(np.int16))


def tensor_to_np_1d(x: Any) -> np.ndarray:
    if x is None:
        return np.zeros((0,), dtype=np.float32)
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "float"):
        x = x.float()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x, dtype=np.float32).reshape(-1)


def text_tokenizer(model_dir: Path):
    return get_qwen_tokenizer(
        token_path=str(model_dir / "CosyVoice-BlankEN"),
        skip_special_tokens=True,
        version="cosyvoice3",
    )


def aligned_sampling_params(text_token_len: int, seed: int) -> SamplingParams:
    return SamplingParams(
        temperature=1.0,
        top_p=0.8,
        top_k=25,
        repetition_penalty=2.0,
        stop_token_ids=[STOP_TOKEN_ID],
        min_tokens=int(text_token_len * 2.0),
        max_tokens=int(text_token_len * 20.0),
        seed=seed,
    )


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def levenshtein_distance(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    dp = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        prev = dp[0]
        dp[0] = i
        for j, y in enumerate(b, start=1):
            cur = dp[j]
            if x == y:
                dp[j] = prev
            else:
                dp[j] = min(prev + 1, dp[j] + 1, dp[j - 1] + 1)
            prev = cur
    return dp[-1]


def text_error_metrics(ref: str, hyp: str) -> dict[str, Any]:
    ref_norm = normalize_text(ref)
    hyp_norm = normalize_text(hyp)
    ref_words = ref_norm.split()
    hyp_words = hyp_norm.split()
    ref_chars = list(ref_norm.replace(" ", ""))
    hyp_chars = list(hyp_norm.replace(" ", ""))
    wer = 0.0 if not ref_words else levenshtein_distance(ref_words, hyp_words) / len(ref_words)
    cer = 0.0 if not ref_chars else levenshtein_distance(ref_chars, hyp_chars) / len(ref_chars)
    return {
        "ref_norm": ref_norm,
        "hyp_norm": hyp_norm,
        "wer": float(wer),
        "cer": float(cer),
    }


def moving_rms(audio: np.ndarray, frame: int, hop: int) -> np.ndarray:
    if audio.size < frame:
        return np.zeros((0,), dtype=np.float32)
    out = []
    for start in range(0, audio.size - frame + 1, hop):
        seg = audio[start : start + frame]
        out.append(float(np.sqrt(np.mean(seg * seg) + 1e-12)))
    return np.asarray(out, dtype=np.float32)


def tail_metrics(audio: np.ndarray, sr: int) -> dict[str, float]:
    if audio.size == 0:
        return {
            "duration_s": 0.0,
            "end_silence_ms": 0.0,
            "tail_500ms_rms": 0.0,
            "tail_1000ms_rms": 0.0,
            "tail_1000ms_voiced_ratio": 0.0,
        }
    frame = max(1, int(sr * 0.02))
    hop = max(1, int(sr * 0.01))
    rms = moving_rms(audio, frame=frame, hop=hop)
    if rms.size == 0:
        end_silence_ms = 0.0
        voiced_ratio = 0.0
    else:
        threshold = max(1e-4, float(rms.max()) * 0.1)
        voiced = rms >= threshold
        last_voiced = np.where(voiced)[0]
        if last_voiced.size == 0:
            end_silence_ms = float(len(rms) * hop / sr * 1000.0)
        else:
            trailing_frames = int(len(rms) - 1 - last_voiced[-1])
            end_silence_ms = float(trailing_frames * hop / sr * 1000.0)
        last_n = max(1, int(1.0 / (hop / sr)))
        voiced_ratio = float(voiced[-last_n:].mean())

    tail_500 = audio[-int(sr * 0.5) :] if audio.size >= int(sr * 0.5) else audio
    tail_1000 = audio[-int(sr * 1.0) :] if audio.size >= int(sr * 1.0) else audio
    return {
        "duration_s": float(audio.size / sr),
        "end_silence_ms": end_silence_ms,
        "tail_500ms_rms": float(np.sqrt(np.mean(tail_500 * tail_500) + 1e-12)),
        "tail_1000ms_rms": float(np.sqrt(np.mean(tail_1000 * tail_1000) + 1e-12)),
        "tail_1000ms_voiced_ratio": voiced_ratio,
    }


def envelope_corr(a: np.ndarray, b: np.ndarray, sr: int) -> float:
    frame = max(1, int(sr * 0.02))
    hop = max(1, int(sr * 0.01))
    ea = moving_rms(a, frame=frame, hop=hop)
    eb = moving_rms(b, frame=frame, hop=hop)
    n = min(len(ea), len(eb))
    if n < 8:
        return 0.0
    ea = ea[:n].astype(np.float64)
    eb = eb[:n].astype(np.float64)
    ea -= ea.mean()
    eb -= eb.mean()
    denom = np.linalg.norm(ea) * np.linalg.norm(eb)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(ea, eb) / denom)


def transcribe_audio(asr_model: Any, wav_path: Path) -> str:
    result = asr_model.transcribe(str(wav_path), language="en", fp16=False)
    return str(result.get("text", "")).strip()


def collect_stage0_info(omni: Omni) -> dict[str, Any]:
    stage0 = omni.stage_list[0]
    stage0_outputs = getattr(stage0, "engine_outputs", None) or []
    if not stage0_outputs:
        return {"error": "no stage0 outputs"}
    completion = stage0_outputs[0].outputs[0]
    stop_reason = getattr(completion, "stop_reason", None)
    try:
        stop_reason_val = None if stop_reason is None else int(stop_reason)
    except Exception:
        stop_reason_val = None
    token_ids = list(getattr(completion, "token_ids", []) or [])
    return {
        "finish_reason": getattr(completion, "finish_reason", None),
        "stop_reason": stop_reason_val,
        "num_tokens": len(token_ids),
        "tail_tokens": token_ids[-20:],
    }


class LocalSyncRunner:
    def __init__(self, model_dir: Path, stage_config: Path, prompt_audio: np.ndarray, prompt_sr: int):
        self.prompt_audio = prompt_audio
        self.prompt_sr = prompt_sr
        tok = text_tokenizer(model_dir)
        self.text_token_len = len(tok.encode(SYNTH_TEXT, allowed_special="all"))
        self.omni = Omni(
            model=str(model_dir),
            stage_configs_path=str(stage_config),
            stage_init_timeout=300,
            batch_timeout=10,
            init_timeout=300,
        )

    def close(self) -> None:
        self.omni.close()

    def generate(self, seed: int) -> tuple[np.ndarray, int, dict[str, Any]]:
        sampling_params_list = self.omni.default_sampling_params_list
        sampling_params_list[0] = aligned_sampling_params(self.text_token_len, seed)
        outputs = self.omni.generate(
            [
                {
                    "prompt": SYNTH_TEXT,
                    "multi_modal_data": {"audio": (self.prompt_audio, self.prompt_sr)},
                    "modalities": ["audio"],
                    "mm_processor_kwargs": {"prompt_text": PROMPT_TEXT},
                }
            ],
            sampling_params_list,
        )
        mm = outputs[0].multimodal_output
        sr_val = mm.get("sr", 24000)
        if isinstance(sr_val, list) and sr_val:
            sr_val = sr_val[-1]
        if hasattr(sr_val, "item"):
            sr_val = sr_val.item()
        sr = int(sr_val)
        chunks_raw = mm.get("audio", [])
        if not isinstance(chunks_raw, list):
            chunks_raw = [chunks_raw]
        chunks = [tensor_to_np_1d(x) for x in chunks_raw if x is not None]
        chunks = [x for x in chunks if x.size > 0]
        audio = np.concatenate(chunks, axis=0) if chunks else np.zeros((0,), dtype=np.float32)
        stage0 = collect_stage0_info(self.omni)
        return audio, sr, stage0


class UpstreamRunner:
    def __init__(self, model_dir: Path):
        from cosyvoice.cli.cosyvoice import CosyVoice3

        self.model = CosyVoice3(str(model_dir), load_trt=False, load_vllm=False, fp16=False)

    def generate(self, seed: int, prompt_wav: Path) -> tuple[np.ndarray, int]:
        from cosyvoice.utils.common import set_all_random_seed

        set_all_random_seed(seed)
        random.seed(seed)
        outputs = list(
            self.model.inference_zero_shot(
                SYNTH_TEXT,
                PROMPT_TEXT,
                str(prompt_wav),
                stream=False,
                text_frontend=False,
            )
        )
        chunks = [tensor_to_np_1d(item["tts_speech"]) for item in outputs]
        chunks = [x for x in chunks if x.size > 0]
        audio = np.concatenate(chunks, axis=0) if chunks else np.zeros((0,), dtype=np.float32)
        return audio, int(self.model.sample_rate)

    def close(self) -> None:
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prompt_audio, prompt_sr = load_prompt_audio(args.prompt_wav)
    asr_model = whisper.load_model(args.whisper_model, device="cpu")
    upstream = None
    local = None
    per_seed: list[dict[str, Any]] = []
    try:
        if args.skip_upstream:
            for seed in args.seeds:
                seed_dir = args.output_dir / f"seed_{seed}"
                upstream_wav = seed_dir / "upstream.wav"
                upstream_audio, upstream_sr = load_prompt_audio(upstream_wav)
                upstream_text = transcribe_audio(asr_model, upstream_wav)
                upstream_text_metrics = text_error_metrics(SYNTH_TEXT, upstream_text)
                per_seed.append(
                    {
                        "seed": seed,
                        "upstream": {
                            "wav": str(upstream_wav),
                            "sample_rate": upstream_sr,
                            "transcript": upstream_text,
                            "text_metrics": upstream_text_metrics,
                            "tail_metrics": tail_metrics(upstream_audio, upstream_sr),
                        },
                    }
                )
        else:
            upstream = UpstreamRunner(args.model)
            for seed in args.seeds:
                seed_dir = args.output_dir / f"seed_{seed}"
                seed_dir.mkdir(parents=True, exist_ok=True)

                upstream_audio, upstream_sr = upstream.generate(seed, args.prompt_wav)
                upstream_wav = seed_dir / "upstream.wav"
                write_wav(upstream_wav, upstream_audio, upstream_sr)

                upstream_text = transcribe_audio(asr_model, upstream_wav)
                upstream_text_metrics = text_error_metrics(SYNTH_TEXT, upstream_text)
                per_seed.append(
                    {
                        "seed": seed,
                        "upstream": {
                            "wav": str(upstream_wav),
                            "sample_rate": upstream_sr,
                            "transcript": upstream_text,
                            "text_metrics": upstream_text_metrics,
                            "tail_metrics": tail_metrics(upstream_audio, upstream_sr),
                        },
                    }
                )

            upstream.close()
            upstream = None
        local = LocalSyncRunner(args.model, args.stage_config, prompt_audio, prompt_sr)

        for item in per_seed:
            seed = int(item["seed"])
            seed_dir = args.output_dir / f"seed_{seed}"
            upstream_audio, upstream_sr = load_prompt_audio(seed_dir / "upstream.wav")

            local_audio, local_sr, local_stage0 = local.generate(seed)
            local_wav = seed_dir / "local_sync.wav"
            write_wav(local_wav, local_audio, local_sr)

            local_text = transcribe_audio(asr_model, local_wav)
            local_text_metrics = text_error_metrics(SYNTH_TEXT, local_text)
            item["local_sync"] = {
                "wav": str(local_wav),
                "sample_rate": local_sr,
                "stage0": local_stage0,
                "transcript": local_text,
                "text_metrics": local_text_metrics,
                "tail_metrics": tail_metrics(local_audio, local_sr),
            }
            item["comparison"] = {
                "duration_delta_s": float(local_audio.size / local_sr - upstream_audio.size / upstream_sr),
                "envelope_corr": envelope_corr(upstream_audio, local_audio, local_sr),
                "wer_delta_local_minus_upstream": float(
                    local_text_metrics["wer"] - item["upstream"]["text_metrics"]["wer"]
                ),
                "cer_delta_local_minus_upstream": float(
                    local_text_metrics["cer"] - item["upstream"]["text_metrics"]["cer"]
                ),
            }

        summary = {
            "seeds": args.seeds,
            "target_text": SYNTH_TEXT,
            "num_seeds": len(per_seed),
            "local_worse_wer_count": int(
                sum(item["comparison"]["wer_delta_local_minus_upstream"] > 0 for item in per_seed)
            ),
            "local_worse_cer_count": int(
                sum(item["comparison"]["cer_delta_local_minus_upstream"] > 0 for item in per_seed)
            ),
            "mean_duration_delta_s": float(np.mean([item["comparison"]["duration_delta_s"] for item in per_seed])),
            "mean_envelope_corr": float(np.mean([item["comparison"]["envelope_corr"] for item in per_seed])),
            "mean_local_wer": float(np.mean([item["local_sync"]["text_metrics"]["wer"] for item in per_seed])),
            "mean_upstream_wer": float(np.mean([item["upstream"]["text_metrics"]["wer"] for item in per_seed])),
            "mean_local_cer": float(np.mean([item["local_sync"]["text_metrics"]["cer"] for item in per_seed])),
            "mean_upstream_cer": float(np.mean([item["upstream"]["text_metrics"]["cer"] for item in per_seed])),
            "per_seed": per_seed,
        }
    finally:
        if upstream is not None:
            upstream.close()
        if local is not None:
            local.close()

    out_path = args.output_dir / "report.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
