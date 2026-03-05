# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from types import SimpleNamespace

import torch

from vllm_omni.model_executor.stage_input_processors.cosyvoice3 import talker2code2wav_async_chunk, text2flow


def _source_output(request_id: str, prompt_ids: list[int], out_ids: list[int], mm: dict):
    return SimpleNamespace(
        request_id=request_id,
        prompt_token_ids=prompt_ids,
        outputs=[SimpleNamespace(token_ids=out_ids, multimodal_output=mm)],
    )


def test_text2flow_supports_batched_source_outputs():
    stage_list = [
        SimpleNamespace(
            engine_outputs=[
                _source_output("req-0", [10, 11], [1, 2, 3], {"speech_token": torch.tensor([[1, 2]])}),
                _source_output("req-1", [20, 21], [4, 5], {"speech_token": torch.tensor([[3, 4]])}),
            ]
        )
    ]

    outputs = text2flow(stage_list=stage_list, engine_input_source=[0], prompt=None)

    assert len(outputs) == 2
    assert outputs[0]["prompt_token_ids"] == [1, 2, 3]
    assert outputs[1]["prompt_token_ids"] == [4, 5]
    assert outputs[0]["additional_information"]["prefix_ids"] == [10, 11]
    assert outputs[1]["additional_information"]["prefix_ids"] == [20, 21]


def test_talker2code2wav_async_chunk_filters_special_tokens_and_emits_payload():
    transfer_manager = SimpleNamespace(
        code_prompt_token_ids=defaultdict(list),
        request_payload={},
        connector=SimpleNamespace(
            config={
                "extra": {
                    "codec_chunk_frames": 2,
                    "codec_left_context_frames": 2,
                    "codec_vocab_size": 6561,
                }
            }
        ),
    )
    request = SimpleNamespace(
        external_req_id="rid-0",
        output_token_ids=[1, 2, 6562, 3],
        additional_information={
            "speech_token": [torch.tensor([[11, 12, 13]])],
            "speech_feat": [torch.tensor([[[0.1, 0.2], [0.3, 0.4]]])],
            "embedding": [torch.tensor([[0.5, 0.6]])],
        },
        is_finished=lambda: True,
    )

    payload = talker2code2wav_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output=None,
        request=request,
        is_finished=True,
    )

    assert payload is not None
    assert payload["finished"].item() is True
    assert payload["code_predictor_codes"] == [1, 2, 3]
    assert payload["left_context_size"] == 2
    assert "speech_token" in payload
    assert "speech_feat" in payload
    assert "embedding" in payload


def test_talker2code2wav_async_chunk_emits_eof_when_finished_without_valid_codes():
    transfer_manager = SimpleNamespace(
        code_prompt_token_ids=defaultdict(list),
        request_payload={},
        connector=SimpleNamespace(
            config={
                "extra": {
                    "codec_chunk_frames": 25,
                    "codec_left_context_frames": 25,
                    "codec_vocab_size": 6561,
                }
            }
        ),
    )
    request = SimpleNamespace(
        external_req_id="rid-eof",
        output_token_ids=[6561, 6562],  # all filtered out
        additional_information={},
        is_finished=lambda: True,
    )

    payload = talker2code2wav_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output=None,
        request=request,
        is_finished=True,
    )

    assert payload is not None
    assert payload["code_predictor_codes"] == []
    assert payload["finished"].item() is True


def test_talker2code2wav_async_chunk_flow_first_tail_timesteps():
    transfer_manager = SimpleNamespace(
        code_prompt_token_ids=defaultdict(list),
        request_payload={},
        connector=SimpleNamespace(
            config={
                "extra": {
                    "codec_chunk_frames": 2,
                    "codec_left_context_frames": 2,
                    "codec_vocab_size": 6561,
                    "flow_first_n_timesteps": 10,
                    "flow_tail_n_timesteps": 6,
                }
            }
        ),
    )
    request = SimpleNamespace(
        external_req_id="rid-flow",
        output_token_ids=[1, 2, 3, 4],
        additional_information={
            "speech_token": [torch.tensor([[11, 12, 13]])],
            "speech_feat": [torch.tensor([[[0.1, 0.2], [0.3, 0.4]]])],
            "embedding": [torch.tensor([[0.5, 0.6]])],
        },
        is_finished=lambda: False,
    )

    payload0 = talker2code2wav_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output=None,
        request=request,
        is_finished=False,
    )
    assert payload0 is not None
    assert payload0["n_timesteps"] == 10

    payload1 = talker2code2wav_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output=None,
        request=request,
        is_finished=True,
    )
    assert payload1 is not None
    assert payload1["n_timesteps"] == 6


def test_talker2code2wav_async_chunk_hop_policy_progressive_hop_and_overlap():
    transfer_manager = SimpleNamespace(
        code_prompt_token_ids=defaultdict(list),
        request_payload={},
        connector=SimpleNamespace(
            config={
                "extra": {
                    "codec_use_hop_policy": True,
                    "token_overlap_len": 2,
                    "token_min_hop_len": 3,
                    "token_max_hop_len": 6,
                    "stream_scale_factor": 2,
                    "codec_vocab_size": 6561,
                }
            }
        ),
    )
    request = SimpleNamespace(
        external_req_id="rid-hop",
        output_token_ids=list(range(1, 12)),
        additional_information={
            "speech_token": [torch.tensor([[11, 12, 13]])],
            "speech_feat": [torch.tensor([[[0.1, 0.2], [0.3, 0.4]]])],
            "embedding": [torch.tensor([[0.5, 0.6]])],
        },
        is_finished=lambda: False,
    )

    payload0 = talker2code2wav_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output=None,
        request=request,
        is_finished=False,
    )
    assert payload0 is not None
    assert payload0["code_predictor_codes"] == [1, 2, 3, 4, 5]
    assert payload0["left_context_size"] == 0
    assert payload0["finished"].item() is False

    payload1 = talker2code2wav_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output=None,
        request=request,
        is_finished=False,
    )
    assert payload1 is not None
    assert payload1["code_predictor_codes"] == [4, 5, 6, 7, 8, 9, 10, 11]
    assert payload1["left_context_size"] == 2
    assert payload1["finished"].item() is False


def test_talker2code2wav_async_chunk_string_false_does_not_enable_hop_policy():
    transfer_manager = SimpleNamespace(
        code_prompt_token_ids=defaultdict(list),
        request_payload={},
        connector=SimpleNamespace(
            config={
                "extra": {
                    "codec_use_hop_policy": "false",
                    "codec_chunk_frames": 2,
                    "codec_left_context_frames": 2,
                    "codec_vocab_size": 6561,
                }
            }
        ),
    )
    request = SimpleNamespace(
        external_req_id="rid-hop-string-false",
        output_token_ids=[1, 2, 3, 4],
        additional_information={
            "speech_token": [torch.tensor([[11, 12, 13]])],
            "speech_feat": [torch.tensor([[[0.1, 0.2], [0.3, 0.4]]])],
            "embedding": [torch.tensor([[0.5, 0.6]])],
        },
        is_finished=lambda: False,
    )

    payload = talker2code2wav_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output=None,
        request=request,
        is_finished=False,
    )
    assert payload is not None
    assert payload["code_predictor_codes"] == [1, 2, 3, 4]
    assert payload["left_context_size"] == 2


def test_talker2code2wav_async_chunk_ignores_hop_knob_parse_when_policy_disabled():
    transfer_manager = SimpleNamespace(
        code_prompt_token_ids=defaultdict(list),
        request_payload={},
        connector=SimpleNamespace(
            config={
                "extra": {
                    "codec_use_hop_policy": False,
                    "codec_chunk_frames": 2,
                    "codec_left_context_frames": 2,
                    "codec_vocab_size": 6561,
                    "token_min_hop_len": "",
                    "token_max_hop_len": "",
                    "token_overlap_len": "",
                }
            }
        ),
    )
    request = SimpleNamespace(
        external_req_id="rid-hop-knob-disabled",
        output_token_ids=[1, 2],
        additional_information={
            "speech_token": [torch.tensor([[11, 12, 13]])],
            "speech_feat": [torch.tensor([[[0.1, 0.2], [0.3, 0.4]]])],
            "embedding": [torch.tensor([[0.5, 0.6]])],
        },
        is_finished=lambda: False,
    )

    payload = talker2code2wav_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output=None,
        request=request,
        is_finished=False,
    )
    assert payload is not None
    assert payload["code_predictor_codes"] == [1, 2]
    assert payload["left_context_size"] == 0
