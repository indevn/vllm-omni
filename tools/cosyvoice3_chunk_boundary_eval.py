#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CosyVoice3 chunk-boundary two-round experiment utility.

Run one round:
  python tools/cosyvoice3_chunk_boundary_eval.py run --round baseline

Compare two rounds:
  python tools/cosyvoice3_chunk_boundary_eval.py compare \
    --base artifacts/cosyvoice3_chunk_boundary/baseline \
    --new artifacts/cosyvoice3_chunk_boundary/optimized
"""

from __future__ import annotations

import argparse
import functools
import io
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import numpy as np
from scipy.io import wavfile
from vllm.sampling_params import SamplingParams

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from vllm_omni.entrypoints.omni import Omni

MODEL_REPO_ID = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
LOCAL_MODEL_SNAPSHOT = (
    "/home/fz/workspace/llm_tts/models/Fun-CosyVoice3-0.5B"
)
PROMPT_WAV_URL = "https://raw.githubusercontent.com/FunAudioLLM/CosyVoice/main/asset/zero_shot_prompt.wav"
PROMPT_TEXT = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"
SYNTH_TEXT = (
    "CosyVoice is undergoing a comprehensive upgrade, providing more accurate, "
    "stable, faster, and better voice generation capabilities."
)
DEFAULT_OUTPUT_ROOT = Path("artifacts/cosyvoice3_chunk_boundary")


def stage_config_path(name: str) -> str:
    return str(Path("vllm_omni/model_executor/stage_configs") / name)


@functools.lru_cache(maxsize=1)
def _load_prompt_from_url() -> tuple[np.ndarray, int]:
    with urlopen(PROMPT_WAV_URL, timeout=30) as resp:
        data = resp.read()
    sr, audio = wavfile.read(io.BytesIO(data))
    if np.issubdtype(audio.dtype, np.integer):
        max_abs = max(abs(np.iinfo(audio.dtype).min), abs(np.iinfo(audio.dtype).max))
        audio = audio.astype(np.float32) / float(max_abs)
    else:
        audio = audio.astype(np.float32)
    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    return np.asarray(audio, dtype=np.float32), int(sr)


def load_prompt_wav(prompt_wav: str | None) -> tuple[np.ndarray, int]:
    if prompt_wav:
        sr, audio = wavfile.read(prompt_wav)
        if np.issubdtype(audio.dtype, np.integer):
            max_abs = max(abs(np.iinfo(audio.dtype).min), abs(np.iinfo(audio.dtype).max))
            audio = audio.astype(np.float32) / float(max_abs)
        else:
            audio = audio.astype(np.float32)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        return np.asarray(audio, dtype=np.float32), int(sr)
    try:
        return _load_prompt_from_url()
    except Exception as e:
        print(f"[WARN] failed to download official prompt wav ({e}), fallback to temporary local synthetic prompt.")
        sr = 24000
        t = np.linspace(0.0, 2.0, int(sr * 2.0), endpoint=False, dtype=np.float32)
        # Temporary fallback prompt: deterministic two-tone + low noise.
        audio = (
            0.15 * np.sin(2.0 * np.pi * 220.0 * t)
            + 0.08 * np.sin(2.0 * np.pi * 440.0 * t)
            + 0.01 * np.random.default_rng(42).standard_normal(size=t.shape).astype(np.float32)
        )
        return np.asarray(np.clip(audio, -1.0, 1.0), dtype=np.float32), sr


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    clipped = np.clip(audio, -1.0, 1.0)
    wavfile.write(path, sr, (clipped * 32767.0).astype(np.int16))


def aligned_stage0_sampling(text: str) -> SamplingParams:
    text_len = max(1, len(text.split()))
    return SamplingParams(
        temperature=1.0,
        top_p=0.8,
        top_k=25,
        repetition_penalty=2.0,
        stop_token_ids=[6562],
        min_tokens=text_len * 2,
        max_tokens=text_len * 20,
    )


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


@dataclass
class BoundaryMetric:
    boundary_index: int
    sample_index: int
    abs_jump: float
    rms_ratio: float
    hf_ratio: float
    centroid_shift_hz: float


class SimpleOmniRunner:
    def __init__(self, model_name: str, stage_cfg: str):
        self.omni = Omni(
            model=model_name,
            stage_configs_path=stage_cfg,
            stage_init_timeout=300,
            batch_timeout=10,
            init_timeout=300,
        )
        self.model_name = model_name

    def close(self) -> None:
        if hasattr(self.omni, "close"):
            self.omni.close()

    def __enter__(self) -> "SimpleOmniRunner":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def get_default_sampling_params_list(self) -> list[Any]:
        return [st.default_sampling_params for st in self.omni.stage_list]

    def _get_omni_inputs(
        self,
        prompt: str,
        audios: tuple[np.ndarray, int],
        mm_processor_kwargs: dict[str, Any] | None,
        modalities: list[str] | None,
    ) -> list[dict[str, Any]]:
        user_content = "<|audio_bos|><|audio_pad|><|audio_eos|>" + prompt
        full_prompt = (
            "<|im_start|>system\n"
            "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
            "capable of perceiving auditory and visual inputs, as well as generating text and speech."
            "<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        payload: dict[str, Any] = {
            "prompt": full_prompt,
            "multi_modal_data": {"audio": audios},
        }
        if modalities:
            payload["modalities"] = modalities
        if mm_processor_kwargs:
            payload["mm_processor_kwargs"] = mm_processor_kwargs
        return [payload]

    def generate_multimodal(
        self,
        prompts: str,
        audios: tuple[np.ndarray, int],
        mm_processor_kwargs: dict[str, Any] | None,
        modalities: list[str] | None,
        sampling_params_list: list[Any],
    ) -> list[Any]:
        omni_inputs = self._get_omni_inputs(
            prompt=prompts,
            audios=audios,
            mm_processor_kwargs=mm_processor_kwargs,
            modalities=modalities,
        )
        return self.omni.generate(omni_inputs, sampling_params_list)


def spectral_centroid_hz(sig: np.ndarray, sr: int) -> float:
    if sig.size == 0:
        return 0.0
    window = np.hanning(sig.size).astype(np.float32)
    spec = np.abs(np.fft.rfft(sig * window)) + 1e-8
    freqs = np.fft.rfftfreq(sig.size, d=1.0 / float(sr))
    return float((freqs * spec).sum() / spec.sum())


def boundary_metrics(audio: np.ndarray, chunk_lengths: list[int], sr: int, half_window: int = 240) -> list[BoundaryMetric]:
    metrics: list[BoundaryMetric] = []
    if audio.size == 0 or len(chunk_lengths) <= 1:
        return metrics

    boundaries = np.cumsum(np.asarray(chunk_lengths, dtype=np.int64))[:-1]
    eps = 1e-8
    for i, b in enumerate(boundaries.tolist(), start=1):
        if b <= 0 or b >= int(audio.size):
            continue
        left = audio[max(0, b - half_window) : b]
        right = audio[b : min(audio.size, b + half_window)]
        if left.size == 0 or right.size == 0:
            continue
        jump = float(abs(float(audio[b]) - float(audio[b - 1])))
        left_rms = float(np.sqrt(np.mean(left * left)) + eps)
        right_rms = float(np.sqrt(np.mean(right * right)) + eps)
        left_hf = float(np.sqrt(np.mean(np.diff(left) ** 2)) + eps) if left.size > 1 else eps
        right_hf = float(np.sqrt(np.mean(np.diff(right) ** 2)) + eps) if right.size > 1 else eps
        left_cent = spectral_centroid_hz(left, sr)
        right_cent = spectral_centroid_hz(right, sr)
        metrics.append(
            BoundaryMetric(
                boundary_index=i,
                sample_index=b,
                abs_jump=jump,
                rms_ratio=right_rms / left_rms,
                hf_ratio=right_hf / left_hf,
                centroid_shift_hz=abs(right_cent - left_cent),
            )
        )
    return metrics


def summarize_metrics(items: list[BoundaryMetric]) -> dict[str, float]:
    if not items:
        return {
            "num_boundaries": 0.0,
            "mean_abs_jump": 0.0,
            "p95_abs_jump": 0.0,
            "mean_hf_ratio": 0.0,
            "p95_hf_ratio": 0.0,
            "mean_centroid_shift_hz": 0.0,
            "p95_centroid_shift_hz": 0.0,
        }
    jumps = np.asarray([x.abs_jump for x in items], dtype=np.float64)
    hfr = np.asarray([x.hf_ratio for x in items], dtype=np.float64)
    cents = np.asarray([x.centroid_shift_hz for x in items], dtype=np.float64)
    return {
        "num_boundaries": float(len(items)),
        "mean_abs_jump": float(jumps.mean()),
        "p95_abs_jump": float(np.percentile(jumps, 95)),
        "mean_hf_ratio": float(hfr.mean()),
        "p95_hf_ratio": float(np.percentile(hfr, 95)),
        "mean_centroid_shift_hz": float(cents.mean()),
        "p95_centroid_shift_hz": float(np.percentile(cents, 95)),
    }


def collect_stage0_info(runner: SimpleOmniRunner) -> dict[str, Any]:
    stage0 = runner.omni.stage_list[0]
    stage0_outputs = getattr(stage0, "engine_outputs", None) or []
    if not stage0_outputs:
        return {"error": "no stage0 outputs"}
    completion = stage0_outputs[0].outputs[0]
    stop_reason = getattr(completion, "stop_reason", None)
    try:
        stop_reason_val: int | None = None if stop_reason is None else int(stop_reason)
    except Exception:
        stop_reason_val = None
    return {
        "finish_reason": getattr(completion, "finish_reason", None),
        "stop_reason": stop_reason_val,
        "num_tokens": len(getattr(completion, "token_ids", []) or []),
    }


def run_single_config(stage_cfg: str, output_dir: Path, prompt_wav: str | None, model_path: str) -> dict[str, Any]:
    prompt_audio, prompt_sr = load_prompt_wav(prompt_wav)
    with SimpleOmniRunner(model_path, stage_cfg=stage_cfg) as runner:
        sampling_params_list = runner.get_default_sampling_params_list()
        sampling_params_list[0] = aligned_stage0_sampling(SYNTH_TEXT)
        outputs = runner.generate_multimodal(
            prompts=SYNTH_TEXT,
            audios=(prompt_audio, prompt_sr),
            mm_processor_kwargs={"prompt_text": PROMPT_TEXT},
            modalities=["audio"],
            sampling_params_list=sampling_params_list,
        )
        if not outputs:
            raise RuntimeError("No outputs returned from generate_multimodal")
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
        chunk_lengths = [int(x.size) for x in chunks if x.size > 0]
        chunks = [x for x in chunks if x.size > 0]
        full_audio = np.concatenate(chunks, axis=0) if chunks else np.zeros((0,), dtype=np.float32)

        output_dir.mkdir(parents=True, exist_ok=True)
        write_wav(output_dir / "full.wav", full_audio, sr)
        for i, c in enumerate(chunks):
            write_wav(output_dir / f"chunk_{i:03d}.wav", c, sr)

        metrics = boundary_metrics(full_audio, chunk_lengths, sr)
        metrics_payload = [asdict(x) for x in metrics]
        summary = summarize_metrics(metrics)
        stage0_info = collect_stage0_info(runner)
        payload = {
            "stage_config": stage_cfg,
            "sample_rate": sr,
            "full_audio_samples": int(full_audio.size),
            "duration_s": float(full_audio.size / sr) if sr > 0 else 0.0,
            "num_chunks": len(chunks),
            "chunk_lengths": chunk_lengths,
            "stage0": stage0_info,
            "boundary_summary": summary,
            "boundary_metrics": metrics_payload,
        }
        (output_dir / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload


def run_round(
    round_name: str,
    output_root: Path,
    prompt_wav: str | None,
    model_path: str,
    sync_stage_config: str,
    async_stage_config: str,
) -> None:
    round_dir = output_root / round_name
    async_payload = run_single_config(async_stage_config, round_dir / "async_chunk", prompt_wav, model_path)
    sync_payload = run_single_config(sync_stage_config, round_dir / "sync", prompt_wav, model_path)
    summary = {
        "round": round_name,
        "async_chunk": {
            "duration_s": async_payload["duration_s"],
            "stage0": async_payload["stage0"],
            "boundary_summary": async_payload["boundary_summary"],
        },
        "sync": {
            "duration_s": sync_payload["duration_s"],
            "stage0": sync_payload["stage0"],
        },
    }
    round_dir.mkdir(parents=True, exist_ok=True)
    (round_dir / "round_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def compare_rounds(base_dir: Path, new_dir: Path) -> None:
    base = json.loads((base_dir / "async_chunk" / "metrics.json").read_text(encoding="utf-8"))
    new = json.loads((new_dir / "async_chunk" / "metrics.json").read_text(encoding="utf-8"))
    keys = [
        "mean_abs_jump",
        "p95_abs_jump",
        "mean_hf_ratio",
        "p95_hf_ratio",
        "mean_centroid_shift_hz",
        "p95_centroid_shift_hz",
    ]
    report: dict[str, Any] = {
        "base": str(base_dir),
        "new": str(new_dir),
        "improvements": {},
        "stage0_base": base.get("stage0", {}),
        "stage0_new": new.get("stage0", {}),
        "duration_base_s": base.get("duration_s", 0.0),
        "duration_new_s": new.get("duration_s", 0.0),
    }
    for k in keys:
        b = float(base["boundary_summary"].get(k, 0.0))
        n = float(new["boundary_summary"].get(k, 0.0))
        rel = 0.0 if b == 0 else (n - b) / b
        report["improvements"][k] = {"base": b, "new": n, "relative_change": rel}
    out_path = new_dir / "comparison_vs_baseline.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="CosyVoice3 chunk-boundary experiment tool")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run one experiment round")
    p_run.add_argument("--round", required=True, help="Round name, e.g. baseline/optimized")
    p_run.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Output root directory")
    p_run.add_argument("--prompt-wav", default=None, help="Optional local prompt wav path")
    p_run.add_argument(
        "--model-path",
        default=LOCAL_MODEL_SNAPSHOT if Path(LOCAL_MODEL_SNAPSHOT).exists() else MODEL_REPO_ID,
        help="Local model dir or HF repo id",
    )
    p_run.add_argument(
        "--sync-stage-config",
        default=stage_config_path("cosyvoice3.yaml"),
        help="Path to sync stage config yaml",
    )
    p_run.add_argument(
        "--async-stage-config",
        default=stage_config_path("cosyvoice3_async_chunk.yaml"),
        help="Path to async stage config yaml",
    )

    p_cmp = sub.add_parser("compare", help="Compare two rounds")
    p_cmp.add_argument("--base", required=True, help="Baseline round directory")
    p_cmp.add_argument("--new", required=True, help="New round directory")

    args = parser.parse_args()
    if args.cmd == "run":
        run_round(
            round_name=args.round,
            output_root=Path(args.output_root),
            prompt_wav=args.prompt_wav,
            model_path=args.model_path,
            sync_stage_config=args.sync_stage_config,
            async_stage_config=args.async_stage_config,
        )
    else:
        compare_rounds(base_dir=Path(args.base), new_dir=Path(args.new))


if __name__ == "__main__":
    main()
