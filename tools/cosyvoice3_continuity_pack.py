#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate continuity analysis and a listening pack for one experiment round."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile


def read_wav(path: Path) -> tuple[int, np.ndarray]:
    sr, audio = wavfile.read(path)
    if np.issubdtype(audio.dtype, np.integer):
        scale = max(abs(np.iinfo(audio.dtype).min), abs(np.iinfo(audio.dtype).max))
        audio = audio.astype(np.float32) / float(scale)
    else:
        audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    return int(sr), np.asarray(audio, dtype=np.float32)


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    wavfile.write(path, sr, (clipped * 32767.0).astype(np.int16))


def normalize_corr(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n < 64:
        return 0.0
    a = np.asarray(a[:n], dtype=np.float64)
    b = np.asarray(b[:n], dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def best_lag_corr(a: np.ndarray, b: np.ndarray, max_lag: int = 256) -> tuple[float, int]:
    best = (-1.0, 0)
    n = min(len(a), len(b))
    if n < 64:
        return best
    a = np.asarray(a[:n], dtype=np.float64)
    b = np.asarray(b[:n], dtype=np.float64)
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            aa = a[lag:]
            bb = b[: len(aa)]
        else:
            bb = b[-lag:]
            aa = a[: len(bb)]
        if len(aa) < 64:
            continue
        corr = normalize_corr(aa, bb)
        if corr > best[0]:
            best = (corr, lag)
    return best


def spec_distance(a: np.ndarray, b: np.ndarray, sr: int) -> float:
    n = min(len(a), len(b))
    if n < 256:
        return 0.0
    a = np.asarray(a[:n], dtype=np.float64)
    b = np.asarray(b[:n], dtype=np.float64)
    _, _, s1 = signal.spectrogram(a, fs=sr, window="hann", nperseg=256, noverlap=192, mode="magnitude")
    _, _, s2 = signal.spectrogram(b, fs=sr, window="hann", nperseg=256, noverlap=192, mode="magnitude")
    m = min(s1.shape[1], s2.shape[1])
    if m == 0:
        return 0.0
    return float(np.mean(np.abs(np.log1p(s1[:, :m]) - np.log1p(s2[:, :m]))))


def load_metrics(round_dir: Path) -> dict:
    return json.loads((round_dir / "async_chunk" / "metrics.json").read_text(encoding="utf-8"))


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def severity_score(metric: dict, best_corr: float) -> float:
    rms_ratio = float(metric.get("rms_ratio", 1.0))
    hf_ratio = float(metric.get("hf_ratio", 1.0))
    centroid = float(metric.get("centroid_shift_hz", 0.0))
    abs_jump = float(metric.get("abs_jump", 0.0))
    rms_term = abs(math.log(max(rms_ratio, 1e-6)))
    hf_term = abs(math.log(max(hf_ratio, 1e-6)))
    return rms_term + hf_term + centroid / 500.0 + abs_jump * 10.0 + max(0.0, 0.6 - best_corr)


def make_questionnaire(round_dir: Path, report: dict, review_dir: Path) -> None:
    lines = [
        "# CosyVoice3 Listening Questionnaire",
        "",
        f"- Round: `{round_dir.name}`",
        f"- Async full wav: `{round_dir / 'async_chunk' / 'full.wav'}`",
        f"- Sync full wav: `{round_dir / 'sync' / 'full.wav'}`",
        "",
        "## Full Audio Questions",
        "",
        "1. Async 相比 Sync，是否还有明显爆音 / click / pop？",
        "2. Async 是否还有吞字、重复、拖影或重叠感？",
        "3. 给 Async 的整体连续性打分（1-5，5 为最好）。",
        "",
        "## Boundary Clips",
        "",
    ]
    for item in report["worst_boundaries"]:
        idx = int(item["boundary_index"])
        lines.extend(
            [
                f"### Boundary {idx:02d}",
                "",
                f"- Async clip: `{review_dir / f'boundary_{idx:02d}_async.wav'}`",
                f"- Sync clip: `{review_dir / f'boundary_{idx:02d}_sync.wav'}`",
                f"- Severity score: `{item['severity_score']:.4f}`",
                f"- Async tail/head best corr: `{item['tail_head_best_corr']:.4f}` @ lag `{item['tail_head_best_lag']}`",
                f"- Chunk-vs-sync head/mid/tail corr: `{item['chunk_sync_head_corr']:.4f}` / `{item['chunk_sync_mid_corr']:.4f}` / `{item['chunk_sync_tail_corr']:.4f}`",
                "",
                "Questions:",
                f"1. Boundary {idx:02d} 的 Async clip 是否比 Sync 更容易听到边界感？",
                "2. 这段更像 click/pop、重复，还是少了一截内容？",
                "3. 给这一段 Async 连续性打分（1-5）。",
                "",
            ]
        )
    (round_dir / "listening_questionnaire.md").write_text("\n".join(lines), encoding="utf-8")


def analyze_round(round_dir: Path, *, top_k: int, context_ms: int) -> dict:
    metrics = load_metrics(round_dir)
    chunk_lengths = metrics["chunk_lengths"]
    boundary_metrics = metrics["boundary_metrics"]

    async_dir = round_dir / "async_chunk"
    sync_dir = round_dir / "sync"
    async_chunks = sorted(async_dir.glob("chunk_*.wav"))
    sr_async, async_full = read_wav(async_dir / "full.wav")
    sr_sync, sync_full = read_wav(sync_dir / "full.wav")
    if sr_async != sr_sync:
        raise ValueError(f"Sample-rate mismatch: async={sr_async}, sync={sr_sync}")
    sr = sr_async

    starts = np.cumsum([0] + chunk_lengths[:-1])
    context_samples = max(1, int(sr * context_ms / 1000.0))
    worst_boundaries: list[dict[str, float | int]] = []
    tail_head_best_corrs: list[float] = []
    tail_head_best_lags: list[float] = []
    chunk_sync_head_corrs: list[float] = []
    chunk_sync_mid_corrs: list[float] = []
    chunk_sync_tail_corrs: list[float] = []
    chunk_sync_head_lags: list[float] = []
    chunk_sync_mid_lags: list[float] = []
    chunk_sync_tail_lags: list[float] = []
    chunk_sync_head_spec: list[float] = []
    chunk_sync_mid_spec: list[float] = []
    chunk_sync_tail_spec: list[float] = []

    for i, (chunk_start, chunk_len) in enumerate(zip(starts, chunk_lengths)):
        async_chunk = async_full[chunk_start : chunk_start + chunk_len]
        sync_chunk = sync_full[chunk_start : chunk_start + chunk_len]
        seg = min(2048, len(async_chunk), len(sync_chunk))
        if seg >= 64:
            head_corr, head_lag = best_lag_corr(sync_chunk[:seg], async_chunk[:seg], max_lag=128)
            tail_corr, tail_lag = best_lag_corr(sync_chunk[-seg:], async_chunk[-seg:], max_lag=128)
            mid_start = max(0, len(async_chunk) // 2 - seg // 2)
            mid_corr, mid_lag = best_lag_corr(
                sync_chunk[mid_start : mid_start + seg],
                async_chunk[mid_start : mid_start + seg],
                max_lag=128,
            )
            chunk_sync_head_corrs.append(head_corr)
            chunk_sync_mid_corrs.append(mid_corr)
            chunk_sync_tail_corrs.append(tail_corr)
            chunk_sync_head_lags.append(abs(head_lag))
            chunk_sync_mid_lags.append(abs(mid_lag))
            chunk_sync_tail_lags.append(abs(tail_lag))
            chunk_sync_head_spec.append(spec_distance(async_chunk[:seg], sync_chunk[:seg], sr))
            chunk_sync_mid_spec.append(
                spec_distance(async_chunk[mid_start : mid_start + seg], sync_chunk[mid_start : mid_start + seg], sr)
            )
            chunk_sync_tail_spec.append(spec_distance(async_chunk[-seg:], sync_chunk[-seg:], sr))

        if i == 0 or i >= len(async_chunks):
            continue
        _, prev_chunk = read_wav(async_chunks[i - 1])
        _, cur_chunk = read_wav(async_chunks[i])
        win = min(context_samples, len(prev_chunk), len(cur_chunk))
        if win >= 64:
            tail_head_corr, tail_head_lag = best_lag_corr(prev_chunk[-win:], cur_chunk[:win], max_lag=min(256, win // 2))
        else:
            tail_head_corr, tail_head_lag = 0.0, 0
        tail_head_best_corrs.append(tail_head_corr)
        tail_head_best_lags.append(abs(tail_head_lag))

        metric = boundary_metrics[i - 1]
        severity = severity_score(metric, tail_head_corr)
        worst_boundaries.append(
            {
                "boundary_index": int(metric["boundary_index"]),
                "sample_index": int(metric["sample_index"]),
                "severity_score": float(severity),
                "tail_head_best_corr": float(tail_head_corr),
                "tail_head_best_lag": int(tail_head_lag),
                "abs_jump": float(metric["abs_jump"]),
                "rms_ratio": float(metric["rms_ratio"]),
                "hf_ratio": float(metric["hf_ratio"]),
                "centroid_shift_hz": float(metric["centroid_shift_hz"]),
                "chunk_sync_head_corr": float(chunk_sync_head_corrs[i]),
                "chunk_sync_mid_corr": float(chunk_sync_mid_corrs[i]),
                "chunk_sync_tail_corr": float(chunk_sync_tail_corrs[i]),
                "chunk_sync_head_spec_dist": float(chunk_sync_head_spec[i]),
                "chunk_sync_mid_spec_dist": float(chunk_sync_mid_spec[i]),
                "chunk_sync_tail_spec_dist": float(chunk_sync_tail_spec[i]),
            }
        )

    worst_boundaries = sorted(worst_boundaries, key=lambda x: x["severity_score"], reverse=True)[:top_k]
    review_dir = round_dir / "review_clips"
    for item in worst_boundaries:
        sample_index = int(item["sample_index"])
        left = max(0, sample_index - context_samples)
        right = min(len(async_full), sample_index + context_samples)
        write_wav(review_dir / f"boundary_{item['boundary_index']:02d}_async.wav", async_full[left:right], sr)
        write_wav(review_dir / f"boundary_{item['boundary_index']:02d}_sync.wav", sync_full[left:right], sr)

    front_boundary_metrics = boundary_metrics[:-2] if len(boundary_metrics) > 2 else boundary_metrics
    report = {
        "round": round_dir.name,
        "sample_rate": sr,
        "stage0_async": metrics.get("stage0", {}),
        "duration_async_s": float(len(async_full) / sr),
        "duration_sync_s": float(len(sync_full) / sr),
        "sample_delta_async_minus_sync": int(len(async_full) - len(sync_full)),
        "boundary_summary_all": metrics.get("boundary_summary", {}),
        "boundary_summary_front": {
            "mean_abs_jump": float(np.mean([m["abs_jump"] for m in front_boundary_metrics])) if front_boundary_metrics else 0.0,
            "mean_hf_ratio": float(np.mean([m["hf_ratio"] for m in front_boundary_metrics])) if front_boundary_metrics else 0.0,
            "mean_centroid_shift_hz": (
                float(np.mean([m["centroid_shift_hz"] for m in front_boundary_metrics])) if front_boundary_metrics else 0.0
            ),
        },
        "tail_head_continuity": {
            "best_corr": summarize(tail_head_best_corrs),
            "abs_best_lag_samples": summarize(tail_head_best_lags),
        },
        "chunk_vs_sync": {
            "head_corr": summarize(chunk_sync_head_corrs),
            "mid_corr": summarize(chunk_sync_mid_corrs),
            "tail_corr": summarize(chunk_sync_tail_corrs),
            "head_abs_lag_samples": summarize(chunk_sync_head_lags),
            "mid_abs_lag_samples": summarize(chunk_sync_mid_lags),
            "tail_abs_lag_samples": summarize(chunk_sync_tail_lags),
            "head_spec_dist": summarize(chunk_sync_head_spec),
            "mid_spec_dist": summarize(chunk_sync_mid_spec),
            "tail_spec_dist": summarize(chunk_sync_tail_spec),
        },
        "worst_boundaries": worst_boundaries,
    }
    (round_dir / "continuity_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    make_questionnaire(round_dir, report, review_dir)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="CosyVoice3 continuity analysis pack")
    parser.add_argument("--round-dir", required=True, help="Round directory under artifacts/cosyvoice3_chunk_boundary")
    parser.add_argument("--top-k", type=int, default=3, help="How many worst boundaries to export")
    parser.add_argument("--context-ms", type=int, default=180, help="Clip context around each boundary")
    args = parser.parse_args()

    round_dir = Path(args.round_dir)
    report = analyze_round(round_dir, top_k=max(1, args.top_k), context_ms=max(20, args.context_ms))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
