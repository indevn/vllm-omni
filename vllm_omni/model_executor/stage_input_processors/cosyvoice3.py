# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from vllm.inputs import TextPrompt

from vllm_omni.inputs.data import OmniTokensPrompt


def _ensure_list(x: Any) -> list[Any]:
    if hasattr(x, "_x"):
        return list(x._x)
    if isinstance(x, list):
        return list(x)
    if isinstance(x, tuple):
        return list(x)
    if x is None:
        return []
    try:
        return list(x)
    except TypeError:
        return [x]


def _to_cpu_tensor(x: Any) -> torch.Tensor | None:
    if isinstance(x, list):
        if not x:
            return None
        x = x[0]
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    return None


def _decode_additional_information(raw_info: Any) -> dict[str, Any]:
    if raw_info is None:
        return {}
    if isinstance(raw_info, dict):
        return raw_info

    entries = getattr(raw_info, "entries", None)
    if not isinstance(entries, dict):
        return {}

    decoded: dict[str, Any] = {}
    for key, entry in entries.items():
        tensor_data = getattr(entry, "tensor_data", None)
        if tensor_data is not None:
            dtype_name = getattr(entry, "tensor_dtype", "float32")
            tensor_shape = getattr(entry, "tensor_shape", None)
            if tensor_shape is None:
                continue
            dt = np.dtype(dtype_name)
            arr = np.frombuffer(tensor_data, dtype=dt).reshape(tensor_shape)
            decoded[key] = torch.from_numpy(arr.copy())
        else:
            decoded[key] = getattr(entry, "list_data", None)
    return decoded


def _as_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        return None
    return ivalue if ivalue > 0 else None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off", ""}:
            return False
    return default


def text2flow(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt = None,
    requires_multimodal_data: bool = True,
):
    """Build stage-1 inputs by prefixing stage-0 prompt ids to its outputs."""
    source_stage_id = engine_input_source[0]
    source_outputs = stage_list[source_stage_id].engine_outputs

    engine_inputs: list[OmniTokensPrompt] = []
    for source_output in source_outputs:
        output = source_output.outputs[0]
        multi_modal_data = output.multimodal_output
        if multi_modal_data is None:
            raise RuntimeError(f"Missing multimodal_output for request {source_output.request_id}")

        output_ids = _ensure_list(output.token_ids)
        prefix_ids = _ensure_list(source_output.prompt_token_ids)
        additional_info = dict(multi_modal_data)
        additional_info["prefix_ids"] = prefix_ids
        engine_inputs.append(OmniTokensPrompt(prompt_token_ids=output_ids, additional_information=additional_info))
    return engine_inputs


def talker2code2wav_async_chunk(
    transfer_manager: Any,
    pooling_output: dict[str, Any] | None,
    request: Any,
    is_finished: bool = False,
) -> dict[str, Any] | None:
    """CosyVoice3 async_chunk processor: talker token stream -> code2wav chunks."""
    request_id = request.external_req_id
    finished = bool(is_finished or request.is_finished())

    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_size = int(cfg.get("codec_chunk_frames", 25))
    left_context_size_cfg = int(cfg.get("codec_left_context_frames", 25))
    code_vocab_size = int(cfg.get("codec_vocab_size", 6561))
    flow_first_n_timesteps = _as_positive_int(cfg.get("flow_first_n_timesteps"))
    flow_tail_n_timesteps = _as_positive_int(cfg.get("flow_tail_n_timesteps"))
    use_hop_policy = _as_bool(cfg.get("codec_use_hop_policy"), default=False)
    token_overlap_len = int(left_context_size_cfg)
    token_min_hop_len = int(chunk_size)
    token_max_hop_len = int(chunk_size)
    stream_scale_factor = 1.0
    first_hop_override = None
    if chunk_size <= 0 or left_context_size_cfg < 0:
        raise ValueError(
            f"Invalid codec chunk config: codec_chunk_frames={chunk_size}, "
            f"codec_left_context_frames={left_context_size_cfg}"
        )
    if use_hop_policy:
        token_overlap_len = int(cfg.get("token_overlap_len", left_context_size_cfg))
        token_min_hop_len = int(cfg.get("token_min_hop_len", chunk_size))
        token_max_hop_len = int(cfg.get("token_max_hop_len", token_min_hop_len))
        stream_scale_factor = _as_float(cfg.get("stream_scale_factor")) or 1.0
        first_hop_override = _as_positive_int(cfg.get("stream_first_token_hop_len"))

        if token_overlap_len < 0:
            raise ValueError(f"Invalid token_overlap_len={token_overlap_len}")
        if token_min_hop_len <= 0 or token_max_hop_len <= 0:
            raise ValueError(
                f"Invalid hop lengths: token_min_hop_len={token_min_hop_len}, token_max_hop_len={token_max_hop_len}"
            )
        if token_max_hop_len < token_min_hop_len:
            token_max_hop_len = token_min_hop_len
        if stream_scale_factor < 1.0:
            raise ValueError(f"Invalid stream_scale_factor={stream_scale_factor}, must be >= 1.0")

    request_state = transfer_manager.request_payload.get(request_id)
    if not isinstance(request_state, dict) or "_cosyvoice3_async_state" not in request_state:
        info = _decode_additional_information(getattr(request, "additional_information", None))
        prompt_payload = {}
        for key in ("speech_token", "speech_feat", "embedding"):
            value = _to_cpu_tensor(info.get(key))
            if value is not None:
                prompt_payload[key] = value
        if isinstance(pooling_output, dict):
            for key in ("speech_token", "speech_feat", "embedding"):
                if key in prompt_payload:
                    continue
                value = _to_cpu_tensor(pooling_output.get(key))
                if value is not None:
                    prompt_payload[key] = value
        request_state = {
            "_cosyvoice3_async_state": {
                "seen_len": 0,
                "sent_prompt": False,
                "emitted_chunks": 0,
                "hop_len": int(first_hop_override or token_min_hop_len),
                "prompt_payload": prompt_payload,
            }
        }
        transfer_manager.request_payload[request_id] = request_state

    state = request_state["_cosyvoice3_async_state"]
    output_token_ids = _ensure_list(getattr(request, "output_token_ids", []))
    seen_len = int(state.get("seen_len", 0))
    new_tokens = output_token_ids[seen_len:] if seen_len < len(output_token_ids) else []
    state["seen_len"] = len(output_token_ids)

    if not hasattr(transfer_manager, "code_prompt_token_ids"):
        transfer_manager.code_prompt_token_ids = defaultdict(list)
    token_frames = transfer_manager.code_prompt_token_ids[request_id]
    for tok in new_tokens:
        tok_int = int(tok)
        if 0 <= tok_int < code_vocab_size:
            token_frames.append([tok_int])

    length = len(token_frames)
    if length <= 0:
        if not finished:
            return None
        payload: dict[str, Any] = {
            "code_predictor_codes": [],
            "finished": torch.tensor(True, dtype=torch.bool),
        }
        if not state.get("sent_prompt", False):
            payload.update(state.get("prompt_payload", {}))
            state["sent_prompt"] = True
        return payload

    if not use_hop_policy:
        chunk_length = length % chunk_size
        if chunk_length != 0 and not finished:
            return None

        context_length = chunk_length if chunk_length != 0 else chunk_size
        end_index = min(length, left_context_size_cfg + context_length)
        left_context_size = max(0, int(end_index - context_length))
        window_frames = token_frames[-end_index:]
        code_predictor_codes = [int(frame[0]) for frame in window_frames]
    else:
        # Upstream-aligned hop cadence (default-off):
        # - wait for hop_len + overlap tokens
        # - emit window = [overlap_from_prev] + [new_hop]
        # - progressive hop growth via stream_scale_factor up to token_max_hop_len
        hop_len = int(state.get("hop_len") or token_min_hop_len)
        hop_len = max(1, min(hop_len, token_max_hop_len))
        overlap_len = max(0, int(token_overlap_len))
        need = hop_len + overlap_len
        if not finished and length < need:
            return None

        window_frames = token_frames if finished else token_frames[:need]
        code_predictor_codes = [int(frame[0]) for frame in window_frames]

        chunk_index = int(state.get("emitted_chunks", 0))
        left_context_size = 0 if chunk_index == 0 else min(overlap_len, len(code_predictor_codes))

        if not finished:
            del token_frames[:hop_len]
            next_hop = int(hop_len * stream_scale_factor)
            next_hop = max(1, min(next_hop, token_max_hop_len))
            state["hop_len"] = next_hop

    payload = {
        "code_predictor_codes": code_predictor_codes,
        "left_context_size": left_context_size,
        "finished": torch.tensor(finished, dtype=torch.bool),
    }
    chunk_index = int(state.get("emitted_chunks", 0))
    n_timesteps = flow_tail_n_timesteps if chunk_index > 0 and flow_tail_n_timesteps is not None else flow_first_n_timesteps
    if n_timesteps is not None:
        payload["n_timesteps"] = n_timesteps
    if not state.get("sent_prompt", False):
        payload.update(state.get("prompt_payload", {}))
        state["sent_prompt"] = True
    state["emitted_chunks"] = int(state.get("emitted_chunks", 0)) + 1
    return payload
