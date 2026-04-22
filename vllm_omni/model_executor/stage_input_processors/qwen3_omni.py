# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The Qwen team.
"""Stage input processor for Qwen3 Omni MoE: Thinker → Talker transition."""

from typing import Any

import torch
from vllm.inputs import TextPrompt
from vllm.logger import init_logger
from vllm.platforms import current_platform

from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.stage_input_processors.tts_utils import (
    extract_language_from_prompt,
    extract_language_from_request,
    extract_speaker_from_prompt,
    extract_speaker_from_request,
)

logger = init_logger(__name__)


def _compute_talker_prompt_ids_length(info, device: torch.device | str = "cuda") -> int:
    im_start_token_id = 151644
    system_token_id = 8948
    user_token_id = 872
    assistant_token_id = 77091

    thinker_sequences = torch.tensor(info["thinker_sequences"], dtype=torch.long, device=device).unsqueeze(0)  # [1, T]

    input_ids = torch.tensor(info["thinker_input_ids"], dtype=torch.long, device=device).unsqueeze(0)  # [1, T]

    im_start_indexes = torch.cat(
        [
            torch.nonzero(input_ids[0] == im_start_token_id).squeeze(1),
            torch.tensor([thinker_sequences.shape[-1]], device=input_ids.device, dtype=input_ids.dtype),
        ],
        dim=0,
    )

    sum_user_len = 0
    assistant_len = 0
    for i in range(len(im_start_indexes) - 1):
        s = int(im_start_indexes[i].item())
        e = int(im_start_indexes[i + 1].item())
        role = int(input_ids[0, s + 1].item())
        if role == system_token_id:
            continue
        elif role == user_token_id:
            sum_user_len += e - s
        elif role == assistant_token_id and i == len(im_start_indexes) - 2:
            assistant_len += 9  # 3 + 4 + 1 + 1
        else:
            pass

    return sum_user_len + assistant_len


# =========================
# Common helpers
# =========================


def _ensure_list(x):
    """Convert ConstantList / tensor-like to Python list."""
    if hasattr(x, "_x"):
        return list(x._x)
    elif not isinstance(x, list):
        return x
    return list(x)


def _validate_stage_inputs(stage_list, engine_input_source):
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")

    stage_id = engine_input_source[0]
    if stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {stage_id}")

    stage = stage_list[stage_id]
    if stage.engine_outputs is None:
        raise RuntimeError(f"Stage {stage_id} has no outputs yet")

    return stage.engine_outputs


# =========================
# Thinker -> Talker
# =========================


def _find_assistant_boundary(prompt_token_ids: list[int]) -> int:
    """Return the index of the last <|im_start|> token that starts an assistant role segment.

    Returns -1 if no assistant segment found.
    """
    IM_START = 151644
    ASSISTANT = 77091
    for i in range(len(prompt_token_ids) - 1, -1, -1):
        if prompt_token_ids[i] == IM_START:
            if i + 1 < len(prompt_token_ids) and prompt_token_ids[i + 1] == ASSISTANT:
                return i
            break  # Last <|im_start|> is not assistant role — stop
    return -1


def _assistant_parts_complete(
    prompt_token_ids: list[int],
    output_token_ids: list[int],
    has_decode_embed: bool = False,
) -> bool:
    """Return True only when ALL of the following hold:
    1. prompt ends with <|im_start|> assistant \\n (>=3 tokens after last im_start)
    2. at least one output token ID is available
    3. has_decode_embed == True (embedding for output_token_ids[0] is ready)
    """
    last_im = _find_assistant_boundary(prompt_token_ids)
    if last_im == -1:
        return True  # No assistant segment — nothing to wait for
    assistant_tokens_in_prompt = len(prompt_token_ids) - last_im
    if assistant_tokens_in_prompt < 3:
        return False  # \n not yet in prefill
    return len(output_token_ids) > 0 and has_decode_embed


def _merge_prefill_payloads(old: dict, new: dict) -> dict:
    """Merge two chunk_id==0 payloads by concatenating prefill embeddings."""
    merged = {**new}
    merged["thinker_prefill_embeddings"] = torch.cat(
        (old["thinker_prefill_embeddings"], new["thinker_prefill_embeddings"]), dim=0
    )
    merged["thinker_hidden_states"] = torch.cat(
        (old["thinker_hidden_states"], new["thinker_hidden_states"]), dim=0
    )
    for k in ("tts_bos_embed", "tts_eos_embed", "tts_pad_embed", "speaker", "language"):
        if k in old and k not in new:
            merged[k] = old[k]
    return merged


def thinker2talker_async_chunk(
    transfer_manager: Any,
    pooling_output: dict[str, Any],
    request: OmniEngineCoreRequest,
    is_finished: bool = False,
) -> list[dict[str, Any]]:
    """
    Process thinker outputs to create talker inputs.
    1. thinker's text generation outputs (token IDs + hidden states)
    2. Split hidden states into: prompt embeddings + generated embeddings
    3. Package for talker with additional information
    """
    # Lazily initialise per-request pending dicts on the transfer_manager
    if not hasattr(transfer_manager, "_pending_assistant"):
        transfer_manager._pending_assistant = {}
    if not hasattr(transfer_manager, "_ready_pre_payload"):
        transfer_manager._ready_pre_payload = {}

    request_id = request.external_req_id

    # Highest priority: return the pre-assistant part that was split off in the
    # previous chunk_id==0 call (when the boundary fell inside the batch).
    if request_id in transfer_manager._ready_pre_payload:
        return transfer_manager._ready_pre_payload.pop(request_id)

    finished = is_finished
    chunk_id = transfer_manager.put_req_chunk[request_id]
    output_token_ids = request.output_token_ids
    # Convert ConstantList to regular list for OmniSerializer serialization
    output_token_ids = _ensure_list(output_token_ids)
    if chunk_id == 0:
        all_token_ids = request.all_token_ids  # prefill + decode
        prompt_token_ids = request.prompt_token_ids
        # Convert ConstantList to regular list for OmniSerializer serialization
        all_token_ids = _ensure_list(all_token_ids)
        prompt_token_ids = _ensure_list(prompt_token_ids)

        embeds_cpu = pooling_output.get("0").detach().cpu()
        hidden_cpu = pooling_output.get("24").detach().cpu()
        batch_size = embeds_cpu.shape[0]
        full_prompt_len = len(prompt_token_ids)
        current_batch_start = full_prompt_len - batch_size

        talker_additional_info = {
            "request_id": request_id,
            "thinker_sequences": all_token_ids,
            "thinker_input_ids": prompt_token_ids,
            # Provide thinker-side TTS token embeddings for talker projection
            "tts_bos_embed": pooling_output.get("tts_bos_embed").detach().cpu(),
            "tts_eos_embed": pooling_output.get("tts_eos_embed").detach().cpu(),
            "tts_pad_embed": pooling_output.get("tts_pad_embed").detach().cpu(),
            "finished": torch.tensor(finished, dtype=torch.bool),
        }
        speaker = extract_speaker_from_request(request)
        if speaker is not None:
            talker_additional_info["speaker"] = speaker
        language = extract_language_from_request(request)
        if language is not None:
            talker_additional_info["language"] = language

        # === Decide based on assistant boundary position ===
        last_im = _find_assistant_boundary(prompt_token_ids)

        if last_im == -1:
            # No assistant segment in this batch — send immediately.
            talker_additional_info["thinker_prefill_embeddings"] = embeds_cpu
            talker_additional_info["thinker_hidden_states"] = hidden_cpu
            return talker_additional_info

        assistant_complete = (full_prompt_len - last_im) >= 3  # im_start + assistant + \n (unused — kept for clarity)
        local_im = last_im - current_batch_start  # offset of boundary within this batch

        # Retrieve any previously accumulated pending payload for this request
        pending = transfer_manager._pending_assistant.pop(request_id, None)

        if local_im > 0:
            # Boundary is inside the current batch: the tokens before it can be
            # sent immediately; the tokens from the boundary onward must wait.
            pre_payload = dict(talker_additional_info)
            pre_payload["thinker_prefill_embeddings"] = embeds_cpu[:local_im]
            pre_payload["thinker_hidden_states"] = hidden_cpu[:local_im]
            pre_payload["finished"] = torch.tensor(False, dtype=torch.bool)
            transfer_manager._ready_pre_payload[request_id] = pre_payload

            asst_payload = dict(talker_additional_info)
            asst_payload["thinker_prefill_embeddings"] = embeds_cpu[local_im:]
            asst_payload["thinker_hidden_states"] = hidden_cpu[local_im:]
            if pending is not None:
                asst_payload = _merge_prefill_payloads(pending, asst_payload)
            transfer_manager._pending_assistant[request_id] = asst_payload
            return None

        # local_im <= 0: entire batch is within the assistant region
        cur_payload = dict(talker_additional_info)
        cur_payload["thinker_prefill_embeddings"] = embeds_cpu
        cur_payload["thinker_hidden_states"] = hidden_cpu
        if pending is not None:
            cur_payload = _merge_prefill_payloads(pending, cur_payload)
        # Whether complete or not, keep accumulating — wait for decode embed
        transfer_manager._pending_assistant[request_id] = cur_payload
        return None
    else:
        embeds = pooling_output.get("0")
        hidden_states = pooling_output.get("24")
        chunk_total = int(embeds.shape[0]) if isinstance(embeds, torch.Tensor) else 0
        num_decode_tokens = len(output_token_ids) if output_token_ids else 0
        # num_prefill_tokens: tokens in this batch that are prompt inputs (not yet generated).
        # In a pure decode step, chunk_total == 1 and num_decode_tokens grows cumulatively,
        # so num_prefill_tokens clamps to 0.
        # In a transition step (last prefill batch → first decode token sampled),
        # chunk_total > num_decode_tokens because the sampled token is the OUTPUT, not an
        # input in this batch.  ALL embeddings in this batch are therefore prefill embeddings.
        num_prefill_tokens = max(0, chunk_total - num_decode_tokens)
        is_transition_or_pure_prefill = num_prefill_tokens > 0
        is_pure_decode = not is_transition_or_pure_prefill

        # Check if there is a pending assistant bootstrap to flush.
        # Only flush on a pure-decode step when the decode embed is available.
        pending = transfer_manager._pending_assistant.get(request_id)
        if pending is not None:
            prompt_token_ids_now = _ensure_list(request.prompt_token_ids)
            has_decode_embed = (
                is_pure_decode
                and isinstance(embeds, torch.Tensor)
                and embeds.numel() > 0
            )
            if _assistant_parts_complete(prompt_token_ids_now, output_token_ids, has_decode_embed):
                # Bootstrap complete — flush the pending payload with first_text embed.
                transfer_manager._pending_assistant.pop(request_id)
                flushed = dict(pending)
                flushed["finished"] = torch.tensor(finished, dtype=torch.bool)
                # Pass ALL available decode embeds so _get_talker_assistant_parts can fill
                # assistant_hidden[3:4] (first_text) and put any extras into trailing_text_hidden.
                # At flush time (first pure-decode step) embeds.shape[0] == 1, so this is
                # equivalent to embeds[0:1], but the intent is clearer.
                flushed["thinker_decode_embeddings"] = embeds.detach().cpu()
                if isinstance(hidden_states, torch.Tensor):
                    flushed["thinker_decode_hidden_states"] = hidden_states.detach().cpu()
                flushed["override_keys"] = [
                    "thinker_output_token_ids",
                    "thinker_decode_hidden_states",
                ]
                flushed["thinker_output_token_ids"] = output_token_ids
                return flushed
            else:
                # Not yet complete — keep waiting.
                pending["finished"] = torch.tensor(finished, dtype=torch.bool)
                return None

        talker_additional_info = {
            "request_id": request_id,
            "finished": torch.tensor(finished, dtype=torch.bool),
        }

        if is_transition_or_pure_prefill and isinstance(embeds, torch.Tensor) and isinstance(hidden_states, torch.Tensor):
            # ALL tokens in this batch are prefill inputs.
            # The newly sampled decode tokens (output_token_ids) are outputs only;
            # their embeddings will appear as inputs in the next pure-decode step.
            talker_additional_info["thinker_prefill_embeddings"] = embeds.detach().cpu()
            talker_additional_info["thinker_hidden_states"] = hidden_states.detach().cpu()
            if output_token_ids:
                # Track token IDs for talker decode ordering, but no decode embeds yet.
                talker_additional_info["override_keys"] = [
                    "thinker_output_token_ids",
                    "thinker_decode_hidden_states",
                ]
                talker_additional_info["thinker_output_token_ids"] = output_token_ids
        elif output_token_ids:
            # Pure decode step: all tokens in this batch are previously-generated tokens
            # being fed back as inputs.  Their embeddings are the decode embeddings.
            # NOTE: thinker_decode_embeddings is intentionally NOT in override_keys so
            # that ChunkTransferAdapter._update_request_payload accumulates it via
            # torch.cat instead of replacing, preserving embeds from earlier steps.
            talker_additional_info["override_keys"] = [
                "thinker_output_token_ids",
                "thinker_decode_hidden_states",
            ]
            if isinstance(embeds, torch.Tensor):
                talker_additional_info["thinker_decode_embeddings"] = embeds.detach().cpu()
                if isinstance(hidden_states, torch.Tensor):
                    talker_additional_info["thinker_decode_hidden_states"] = hidden_states.detach().cpu()
            talker_additional_info["thinker_output_token_ids"] = output_token_ids
    return talker_additional_info


def thinker2talker(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """
    Process thinker outputs to create talker inputs.

    Workflow:
    1. Extract thinker's text generation outputs (token IDs + hidden states)
    2. Split hidden states into: prompt embeddings + generated embeddings
    3. Package for talker with additional information

    Args:
        stage_list: List of stage objects
        engine_input_source: Source stage IDs (typically [0] for thinker)
        prompt: Original prompt data
        requires_multimodal_data: Whether multimodal data is required

    Returns:
        List of OmniTokensPrompt for talker stage
    """
    thinker_outputs = _validate_stage_inputs(stage_list, engine_input_source)
    talker_inputs: list[OmniTokensPrompt] = []

    device = torch.device(current_platform.device_type)

    # Process each thinker output
    for i, thinker_output in enumerate(thinker_outputs):
        output = thinker_output.outputs[0]

        info = {
            "thinker_prefill_embeddings": output.multimodal_output["0"].detach().to(device=device, dtype=torch.float),
            "thinker_hidden_states": output.multimodal_output["24"].detach().to(device=device, dtype=torch.float),
            "thinker_sequences": (
                thinker_output.prompt_token_ids + output.token_ids
            ),  # the thinker_sequences is the whole ids
            "thinker_input_ids": thinker_output.prompt_token_ids,
            # Provide thinker-side TTS token embeddings for talker projection
            "tts_bos_embed": output.multimodal_output["tts_bos_embed"].detach().to(device=device, dtype=torch.float),
            "tts_eos_embed": output.multimodal_output["tts_eos_embed"].detach().to(device=device, dtype=torch.float),
            "tts_pad_embed": output.multimodal_output["tts_pad_embed"].detach().to(device=device, dtype=torch.float),
        }
        speaker = extract_speaker_from_prompt(prompt, index=i)
        if speaker is not None:
            info["speaker"] = speaker
        language = extract_language_from_prompt(prompt, index=i)
        if language is not None:
            info["language"] = language

        prompt_len = _compute_talker_prompt_ids_length(info, device=device)

        talker_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0] * prompt_len,
                additional_information=info,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return talker_inputs


# =========================
# Talker -> Code2Wav
# =========================


def talker2code2wav_async_chunk(
    transfer_manager: Any,
    pooling_output: dict[str, Any],
    request: OmniEngineCoreRequest,
    is_finished: bool = False,
):
    """
    Pooling version.
    """
    request_id = getattr(request, "external_req_id", None)

    if "code_predictor_codes" not in pooling_output:
        logger.info(f"[CODE2WAV_DIAG] req={request_id[-12:] if request_id else 'N/A'} DROP=no_key keys={list(pooling_output.keys())[:5]}")
        return None
    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_size_config = int(cfg.get("codec_chunk_frames", 25))
    left_context_size_config = int(cfg.get("codec_left_context_frames", 25))

    if not isinstance(pooling_output, dict) or "code_predictor_codes" not in pooling_output:
        return None

    code_predictor_codes = pooling_output["code_predictor_codes"]

    if code_predictor_codes is None:
        logger.warning("code_predictor_codes is None")
        return None
    if isinstance(code_predictor_codes, torch.Tensor):
        if code_predictor_codes.numel() == 0:
            logger.warning("code_predictor_codes is empty 0")
            return None
    elif hasattr(code_predictor_codes, "__len__"):
        if len(code_predictor_codes) == 0:
            logger.warning("code_predictor_codes is empty 1")
            return None
    
    if isinstance(code_predictor_codes, torch.Tensor):
        # TODO: high concurrency issue here, need to fix it
        if not code_predictor_codes.any():
            logger.warning(f"[CODE2WAV_DIAG] req={request_id[-12:] if request_id else 'N/A'} DROP=all_zero shape={code_predictor_codes.shape}")
            return None
    else:
        code_tensor = torch.tensor(code_predictor_codes, dtype=torch.long)
        if not code_tensor.any():
            return None

    codec_codes = code_predictor_codes.to(torch.long).transpose(0, 1).cpu().to(torch.long).reshape(-1).tolist()
    if sum(codec_codes) == 0:
        logger.info(f"[CODE2WAV_DIAG] req={request_id[-12:] if request_id else 'N/A'} DROP=sum_zero")
        return None

    transfer_manager.code_prompt_token_ids[request_id].append(codec_codes)
    length = len(transfer_manager.code_prompt_token_ids[request_id])
    chunk_length = length % chunk_size_config
    if chunk_length != 0 and not is_finished:
        return None

    context_length = chunk_length if chunk_length != 0 else chunk_size_config
    # ensure left context does not exceed available length
    left_context_size = max(0, min(length - context_length, left_context_size_config))
    end_index = min(length, left_context_size + context_length)

    codes = (
        torch.tensor(transfer_manager.code_prompt_token_ids[request_id][-end_index:])
        .transpose(0, 1)
        .reshape(-1)
        .tolist()
    )
    info = {
        "code_predictor_codes": codes,
        "left_context_size": left_context_size,
        "finished": torch.tensor(is_finished, dtype=torch.bool),
    }
    return info


def talker2code2wav(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """
    Process talker outputs to create code2wav inputs.

    Workflow:
    1. Extract talker's codec code outputs (8-layer RVQ codes)
    2. Flatten codes for code2wav input
    3. Package for code2wav stage

    Args:
        stage_list: List of stage objects
        engine_input_source: Source stage IDs (typically [1] for talker)
        prompt: Original prompt data
        requires_multimodal_data: Whether multimodal data is required

    Returns:
        List of OmniTokensPrompt for code2wav stage
    """
    talker_outputs = _validate_stage_inputs(stage_list, engine_input_source)
    code2wav_inputs: list[OmniTokensPrompt] = []
    # Process each talker output
    for talker_output in talker_outputs:
        output = talker_output.outputs[0]
        seq_len = len(output.token_ids) - 1
        # Extract codec codes from talker output
        # Expected shape: [8, seq_len] (8-layer RVQ codes)
        codec_codes = (
            output.multimodal_output["code_predictor_codes"][-seq_len:]
            .to(torch.long)
            .transpose(0, 1)
            .cpu()
            .to(torch.long)
            .reshape(-1)
            .tolist()
        )  # 16, seq_len
        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=codec_codes,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return code2wav_inputs
