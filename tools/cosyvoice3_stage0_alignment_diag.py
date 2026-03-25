#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Diagnose CosyVoice3 stage-0 alignment between upstream HF and local Omni.

This tool answers three questions with one fixed-seed run:

1. Are text/prompt/audio inputs aligned between local and upstream?
2. Does local stage-0 generate the same token trajectory as upstream HF?
3. If not, where does the sequence first diverge?

Run with:
  PYTHONNOUSERSITE=1 \
  PYTHONPATH=/home/fz/workspace/vllm-omni-cosyvoice3/vllm-omni:/home/fz/workspace/llm_tts/third_party/CosyVoice \
  /home/fz/miniconda3/envs/env-vllm-omni-018/bin/python \
  tools/cosyvoice3_stage0_alignment_diag.py \
  --output artifacts/cosyvoice3_chunk_boundary/fix_s10_stage0_alignment_diag/report.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime
import soundfile as sf
import torch
import torchaudio
import whisper
from vllm.sampling_params import SamplingParams

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.model_executor.models.cosyvoice3.tokenizer import get_qwen_tokenizer
from vllm_omni.model_executor.models.cosyvoice3.utils import log_mel_spectrogram

MODEL = Path("/home/fz/workspace/llm_tts/models/Fun-CosyVoice3-0.5B")
PROMPT_WAV = Path(
    "/home/fz/workspace/vllm-omni-cosyvoice3/vllm-omni/artifacts/cosyvoice3_chunk_boundary/zero_shot_prompt.wav"
)
DEFAULT_STAGE_CONFIG = Path("/tmp/cosyvoice3_sync_tokenizer.yaml")
PROMPT_TEXT = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"
SYNTH_TEXT = (
    "CosyVoice is undergoing a comprehensive upgrade, providing more accurate, "
    "stable, faster, and better voice generation capabilities."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--prompt-wav", type=Path, default=PROMPT_WAV)
    parser.add_argument("--stage-config", type=Path, default=DEFAULT_STAGE_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260325)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--repetition-penalty", type=float, default=2.0)
    parser.add_argument("--stop-token-id", type=int, default=6562)
    parser.add_argument("--min-token-ratio", type=float, default=2.0)
    parser.add_argument("--max-token-ratio", type=float, default=20.0)
    parser.add_argument("--greedy", action="store_true")
    return parser.parse_args()


def load_prompt_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    return np.asarray(audio, dtype=np.float32), int(sr)


def load_audio_tensor(path: Path, target_sr: int) -> torch.Tensor:
    audio, sr = load_prompt_audio(path)
    speech = torch.from_numpy(audio).unsqueeze(0)
    if int(sr) == int(target_sr):
        return speech
    return torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(speech)


def text_tokenizer(model_dir: Path):
    return get_qwen_tokenizer(
        token_path=str(model_dir / "CosyVoice-BlankEN"),
        skip_special_tokens=True,
        version="cosyvoice3",
    )


def aligned_sampling_params(
    *,
    text_token_len: int,
    seed: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    stop_token_id: int,
    min_token_ratio: float,
    max_token_ratio: float,
) -> SamplingParams:
    return SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        stop_token_ids=[stop_token_id],
        min_tokens=int(text_token_len * min_token_ratio),
        max_tokens=int(text_token_len * max_token_ratio),
        seed=seed,
    )


def speech_token_from_upstream_formula(prompt_wav: Path, model_dir: Path) -> torch.Tensor:
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    session = onnxruntime.InferenceSession(
        str(model_dir / "speech_tokenizer_v3.onnx"),
        sess_options=option,
        providers=["CPUExecutionProvider"],
    )
    speech = load_audio_tensor(prompt_wav, 16000)
    feat = whisper.log_mel_spectrogram(speech, n_mels=128)
    ids = session.run(
        None,
        {
            session.get_inputs()[0].name: feat.detach().cpu().numpy(),
            session.get_inputs()[1].name: np.array([feat.shape[2]], dtype=np.int32),
        },
    )[0].flatten().tolist()
    return torch.tensor([ids], dtype=torch.int32)


def speech_token_from_local_formula(prompt_wav: Path, model_dir: Path) -> torch.Tensor:
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    session = onnxruntime.InferenceSession(
        str(model_dir / "speech_tokenizer_v3.onnx"),
        sess_options=option,
        providers=["CPUExecutionProvider"],
    )
    speech = load_audio_tensor(prompt_wav, 16000)
    feat = log_mel_spectrogram(speech, n_mels=128)
    ids = session.run(
        None,
        {
            session.get_inputs()[0].name: feat.detach().cpu().numpy(),
            session.get_inputs()[1].name: np.array([feat.shape[2]], dtype=np.int32),
        },
    )[0].flatten().tolist()
    return torch.tensor([ids], dtype=torch.int32)


def build_upstream_llm(model_dir: Path) -> Any:
    from cosyvoice.llm.llm import CosyVoice3LM, Qwen2Encoder
    from cosyvoice.utils.common import ras_sampling

    qwen = Qwen2Encoder(pretrain_path=str(model_dir / "CosyVoice-BlankEN"))
    llm = CosyVoice3LM(
        llm_input_size=896,
        llm_output_size=896,
        speech_token_size=6561,
        llm=qwen,
        sampling=ras_sampling,
        length_normalized_loss=True,
        lsm_weight=0.0,
        mix_ratio=[5, 15],
    )
    state = torch.load(model_dir / "llm.pt", map_location="cpu", weights_only=True)
    llm.load_state_dict(state, strict=True)
    llm.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return llm.to(device)


def inspect_inputs(model_dir: Path, prompt_wav: Path) -> dict[str, Any]:
    tok = text_tokenizer(model_dir)
    prompt_ids = tok.encode(PROMPT_TEXT, allowed_special="all")
    text_ids = tok.encode(SYNTH_TEXT, allowed_special="all")
    upstream_pst = speech_token_from_upstream_formula(prompt_wav, model_dir)
    local_pst = speech_token_from_local_formula(prompt_wav, model_dir)

    llm = build_upstream_llm(model_dir)
    try:
        device = next(llm.parameters()).device
        prompt_ids_t = torch.tensor([prompt_ids], dtype=torch.int64, device=device)
        text_ids_t = torch.tensor([text_ids], dtype=torch.int64, device=device)
        concat_ids_t = torch.cat([prompt_ids_t, text_ids_t], dim=1)
        pst = upstream_pst.to(device)

        text_emb_direct = llm.llm.model.model.embed_tokens(concat_ids_t)
        sos = llm.speech_embedding.weight[llm.sos].reshape(1, 1, -1)
        task = llm.speech_embedding.weight[llm.task_id].reshape(1, 1, -1)
        pst_emb = llm.speech_embedding(pst)
        lm_input_direct = torch.cat([sos, text_emb_direct, task, pst_emb], dim=1)

        fake_prefix = torch.ones((1, 2 + pst.shape[1]), dtype=concat_ids_t.dtype, device=device)
        fake_input_ids = torch.cat([fake_prefix, concat_ids_t], dim=1)
        embed_tokens = llm.llm.model.model.embed_tokens(fake_input_ids)
        lm_input_localstyle = torch.cat([sos, embed_tokens[:, 2 + pst.shape[1] :], task, pst_emb], dim=1)

        mask = torch.tril(torch.ones((1, lm_input_direct.shape[1], lm_input_direct.shape[1]), device=device)).to(torch.bool)
        y1, _ = llm.llm.forward_one_step(lm_input_direct, masks=mask, cache=None)
        y2, _ = llm.llm.forward_one_step(lm_input_localstyle, masks=mask, cache=None)
        logits1 = llm.llm_decoder(y1[:, -1]).float()
        logits2 = llm.llm_decoder(y2[:, -1]).float()
        topk1 = torch.topk(logits1[0], k=10)
    finally:
        del llm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "prompt_text_len": len(prompt_ids),
        "synth_text_len": len(text_ids),
        "concat_text_len": len(prompt_ids) + len(text_ids),
        "prompt_text_ids": prompt_ids,
        "synth_text_ids": text_ids,
        "prompt_speech_len_upstream": int(upstream_pst.shape[1]),
        "prompt_speech_len_local": int(local_pst.shape[1]),
        "prompt_speech_same": bool(torch.equal(upstream_pst, local_pst)),
        "prompt_speech_num_diff": int(upstream_pst.ne(local_pst).sum().item()),
        "lm_input_shape_direct": list(lm_input_direct.shape),
        "lm_input_shape_localstyle": list(lm_input_localstyle.shape),
        "lm_input_max_abs_diff": float((lm_input_direct - lm_input_localstyle).abs().max().item()),
        "first_step_logits_max_abs_diff": float((logits1 - logits2).abs().max().item()),
        "first_step_top10_ids": topk1.indices.tolist(),
        "first_step_top10_logits": [float(x) for x in topk1.values.tolist()],
    }


def run_upstream_hf(
    *,
    model_dir: Path,
    prompt_wav: Path,
    seed: int,
    top_k: int,
    stop_token_id: int,
    min_token_ratio: float,
    max_token_ratio: float,
    greedy: bool,
) -> dict[str, Any]:
    from cosyvoice.utils.common import set_all_random_seed

    set_all_random_seed(seed)
    random.seed(seed)

    tok = text_tokenizer(model_dir)
    prompt_ids = tok.encode(PROMPT_TEXT, allowed_special="all")
    text_ids = tok.encode(SYNTH_TEXT, allowed_special="all")
    llm = build_upstream_llm(model_dir)

    try:
        device = next(llm.parameters()).device
        prompt_ids_t = torch.tensor([prompt_ids], dtype=torch.int64, device=device)
        text_ids_t = torch.tensor([text_ids], dtype=torch.int64, device=device)
        concat_ids_t = torch.cat([prompt_ids_t, text_ids_t], dim=1)
        pst = speech_token_from_upstream_formula(prompt_wav, model_dir).to(device)

        text_emb = llm.llm.model.model.embed_tokens(concat_ids_t)
        sos = llm.speech_embedding.weight[llm.sos].reshape(1, 1, -1)
        task = llm.speech_embedding.weight[llm.task_id].reshape(1, 1, -1)
        pst_emb = llm.speech_embedding(pst)
        lm_input = torch.cat([sos, text_emb, task, pst_emb], dim=1)

        min_len = int(len(text_ids) * min_token_ratio)
        max_len = int(len(text_ids) * max_token_ratio)
        out_tokens: list[int] = []
        cache = None
        finish_reason = "length"
        stop_reason = None

        for i in range(max_len):
            seq_len = lm_input.shape[1] if cache is None else lm_input.shape[1] + cache[0][0].size(2)
            mask = torch.tril(torch.ones((1, seq_len, seq_len), device=device)).to(torch.bool)
            y_pred, cache = llm.llm.forward_one_step(lm_input, masks=mask, cache=cache)
            logp = llm.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
            if greedy:
                token_scores = logp.squeeze(dim=0).clone()
                if i < min_len:
                    token_scores[stop_token_id] = float("-inf")
                top_id = int(torch.argmax(token_scores).item())
            else:
                top_id = llm.sampling_ids(
                    logp.squeeze(dim=0),
                    out_tokens,
                    top_k,
                    ignore_eos=True if i < min_len else False,
                )
            if top_id == stop_token_id:
                finish_reason = "stop"
                stop_reason = int(top_id)
                break
            out_tokens.append(int(top_id))
            lm_input = llm.speech_embedding.weight[top_id].reshape(1, 1, -1)

        return {
            "finish_reason": finish_reason,
            "stop_reason": stop_reason,
            "num_tokens": len(out_tokens),
            "token_ids": out_tokens,
            "head_tokens": out_tokens[:64],
            "tail_tokens": out_tokens[-64:],
        }
    finally:
        del llm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_local_stage0(
    *,
    model_dir: Path,
    stage_config: Path,
    prompt_wav: Path,
    seed: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    stop_token_id: int,
    min_token_ratio: float,
    max_token_ratio: float,
    greedy: bool,
) -> dict[str, Any]:
    prompt_audio, prompt_sr = load_prompt_audio(prompt_wav)
    tok = text_tokenizer(model_dir)
    synth_text_ids = tok.encode(SYNTH_TEXT, allowed_special="all")

    omni = Omni(
        model=str(model_dir),
        stage_configs_path=str(stage_config),
        stage_init_timeout=300,
        batch_timeout=10,
        init_timeout=300,
    )
    try:
        sampling_params_list = omni.default_sampling_params_list
        sampling_params_list[0] = aligned_sampling_params(
            text_token_len=len(synth_text_ids),
            seed=seed,
            temperature=0.0 if greedy else temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stop_token_id=stop_token_id,
            min_token_ratio=min_token_ratio,
            max_token_ratio=max_token_ratio,
        )

        outputs = omni.generate(
            [
                {
                    "prompt": SYNTH_TEXT,
                    "multi_modal_data": {"audio": (prompt_audio, prompt_sr)},
                    "modalities": ["audio"],
                    "mm_processor_kwargs": {"prompt_text": PROMPT_TEXT},
                }
            ],
            sampling_params_list,
        )
        _ = outputs

        completion = omni.stage_list[0].engine_outputs[0].outputs[0]
        token_ids = list(getattr(completion, "token_ids", []) or [])
        return {
            "finish_reason": getattr(completion, "finish_reason", None),
            "stop_reason": getattr(completion, "stop_reason", None),
            "num_tokens": len(token_ids),
            "token_ids": token_ids,
            "head_tokens": token_ids[:64],
            "tail_tokens": token_ids[-64:],
        }
    finally:
        omni.close()


def compare_sequences(local_ids: list[int], upstream_ids: list[int]) -> dict[str, Any]:
    prefix_len = 0
    for a, b in zip(local_ids, upstream_ids):
        if a != b:
            break
        prefix_len += 1

    first_diff_index = prefix_len if prefix_len < min(len(local_ids), len(upstream_ids)) else None
    local_diff_token = local_ids[first_diff_index] if first_diff_index is not None else None
    upstream_diff_token = upstream_ids[first_diff_index] if first_diff_index is not None else None

    return {
        "common_prefix_len": prefix_len,
        "first_diff_index": first_diff_index,
        "local_diff_token": local_diff_token,
        "upstream_diff_token": upstream_diff_token,
        "local_len": len(local_ids),
        "upstream_len": len(upstream_ids),
    }


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    input_checks = inspect_inputs(args.model, args.prompt_wav)
    upstream_hf = run_upstream_hf(
        model_dir=args.model,
        prompt_wav=args.prompt_wav,
        seed=args.seed,
        top_k=args.top_k,
        stop_token_id=args.stop_token_id,
        min_token_ratio=args.min_token_ratio,
        max_token_ratio=args.max_token_ratio,
        greedy=args.greedy,
    )
    local_stage0 = run_local_stage0(
        model_dir=args.model,
        stage_config=args.stage_config,
        prompt_wav=args.prompt_wav,
        seed=args.seed,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        stop_token_id=args.stop_token_id,
        min_token_ratio=args.min_token_ratio,
        max_token_ratio=args.max_token_ratio,
        greedy=args.greedy,
    )
    sequence_compare = compare_sequences(local_stage0["token_ids"], upstream_hf["token_ids"])

    report = {
        "seed": args.seed,
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "stop_token_id": args.stop_token_id,
            "min_token_ratio": args.min_token_ratio,
            "max_token_ratio": args.max_token_ratio,
            "greedy": args.greedy,
        },
        "input_checks": input_checks,
        "upstream_hf": upstream_hf,
        "local_stage0": local_stage0,
        "sequence_compare": sequence_compare,
    }

    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
