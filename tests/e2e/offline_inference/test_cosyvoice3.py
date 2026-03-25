# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Offline E2E smoke test for CosyVoice3 (text + prompt audio -> audio).

This test is aligned with the PR #498 reproduction defaults documented in AGENTS.md:
- temperature=1.0, top_p=0.8, top_k=25, repetition_penalty=2.0
- stop_token_ids=[6562]
- min_tokens/max_tokens derived from text length via CosyVoice3Config ratios
"""

from __future__ import annotations

import functools
import io
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pytest
import soundfile as sf
from vllm.sampling_params import SamplingParams

from tests.utils import hardware_test

MODEL = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"

# Official prompt audio path in FunAudioLLM/CosyVoice repo.
_PROMPT_WAV_URL = "https://raw.githubusercontent.com/FunAudioLLM/CosyVoice/main/asset/zero_shot_prompt.wav"

# PR-aligned prompt transcript and synthesis text (see AGENTS.md).
PROMPT_TEXT = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"
SYNTH_TEXT = (
    "CosyVoice is undergoing a comprehensive upgrade, providing more accurate, "
    "stable, faster, and better voice generation capabilities."
)


def _stage_config(name: str) -> str:
    return str(
        Path(__file__).parent.parent.parent.parent
        / "vllm_omni"
        / "model_executor"
        / "stage_configs"
        / name
    )


STAGE_CONFIGS = [
    _stage_config("cosyvoice3.yaml"),
    _stage_config("cosyvoice3_async_chunk.yaml"),
]

TEST_PARAMS = [(MODEL, cfg) for cfg in STAGE_CONFIGS]


@functools.lru_cache(maxsize=1)
def _load_prompt_wav() -> tuple[np.ndarray, int]:
    with urlopen(_PROMPT_WAV_URL, timeout=30) as resp:
        data = resp.read()
    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    return np.asarray(audio, dtype=np.float32), int(sr)


def _aligned_stage0_sampling(*, text: str) -> SamplingParams:
    # PR #498 aligned: min/max tokens derived from "text length" via ratios.
    # The PR-aligned reproductions use word count for English sentences.
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


def _concat_audio(audio_val) -> np.ndarray:
    import torch

    if isinstance(audio_val, list):
        tensors = []
        for t in audio_val:
            if t is None:
                continue
            if hasattr(t, "detach"):
                t = t.detach()
            if hasattr(t, "cpu"):
                t = t.cpu()
            if hasattr(t, "float"):
                t = t.float()
            if isinstance(t, torch.Tensor):
                tensors.append(t.reshape(-1))
        if not tensors:
            return np.zeros((0,), dtype=np.float32)
        return torch.cat(tensors, dim=-1).numpy().astype(np.float32, copy=False)

    if hasattr(audio_val, "detach"):
        audio_val = audio_val.detach()
    if hasattr(audio_val, "cpu"):
        audio_val = audio_val.cpu()
    if hasattr(audio_val, "float"):
        audio_val = audio_val.float()
    if hasattr(audio_val, "numpy"):
        audio_val = audio_val.numpy()
    audio_np = np.asarray(audio_val, dtype=np.float32)
    return audio_np.reshape(-1)


def _build_raw_inputs(prompt_audio: tuple[np.ndarray, int]) -> list[dict[str, object]]:
    return [
        {
            "prompt": SYNTH_TEXT,
            "multi_modal_data": {"audio": prompt_audio},
            "modalities": ["audio"],
            "mm_processor_kwargs": {"prompt_text": PROMPT_TEXT},
        }
    ]


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_runner", TEST_PARAMS, indirect=True)
def test_cosyvoice3_e2e_pr498_aligned(omni_runner) -> None:
    """CosyVoice3 offline E2E: verify stop behavior and audio duration is sane."""
    prompt_audio, prompt_sr = _load_prompt_wav()

    sampling_params_list = omni_runner.get_default_sampling_params_list()
    sampling_params_list[0] = _aligned_stage0_sampling(text=SYNTH_TEXT)

    outputs = omni_runner.omni.generate(_build_raw_inputs((prompt_audio, prompt_sr)), sampling_params_list)

    assert outputs, "No outputs returned"
    audio_mm = outputs[0].multimodal_output
    assert "audio" in audio_mm, "No audio output found"

    audio = _concat_audio(audio_mm["audio"])
    assert audio.size > 0, "Generated audio is empty"

    # Sample rate is typically returned as a list of tensors (one per request).
    sr_val = audio_mm.get("sr", 24000)
    if isinstance(sr_val, list) and sr_val:
        sr_val = sr_val[-1]
    if hasattr(sr_val, "item"):
        sr_val = sr_val.item()
    sr = int(sr_val)
    assert sr == 24000, f"Unexpected sample_rate={sr}"

    duration_s = audio.size / sr
    assert 2.8 <= duration_s <= 4.2, f"Unexpected duration={duration_s:.3f}s (samples={audio.size}, sr={sr})"

    # Verify stage-0 stopped by EOS (not max_tokens loop).
    stage0 = omni_runner.omni.stage_list[0]
    stage0_outputs = getattr(stage0, "engine_outputs", None) or []
    assert len(stage0_outputs) >= 1, "Stage-0 produced no engine outputs"
    completion = stage0_outputs[0].outputs[0]
    finish_reason = getattr(completion, "finish_reason", None)
    stop_reason = getattr(completion, "stop_reason", None)
    num_tokens = len(getattr(completion, "token_ids", []) or [])

    assert finish_reason == "stop", f"Stage-0 finish_reason={finish_reason}, expected 'stop'"
    assert int(stop_reason) == 6562, f"Stage-0 stop_reason={stop_reason}, expected 6562"
    assert 80 <= num_tokens <= 100, f"Stage-0 num_tokens={num_tokens}, expected ~88"
