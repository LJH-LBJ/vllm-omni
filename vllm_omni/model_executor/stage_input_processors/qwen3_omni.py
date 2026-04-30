# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The Qwen team.
"""Stage input processor for Qwen3 Omni MoE: Thinker → Talker transition."""

import logging
from dataclasses import dataclass, field
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

# Pooling output layer keys: "0" = word embedding, "24" = accept_hidden_layer
_EMBED_LAYER_KEY = "0"
_HIDDEN_LAYER_KEY = "24"

# ChatML special token IDs
_IM_START_TOKEN_ID = 151644
_SYSTEM_TOKEN_ID = 8948


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
# PD disaggregation helpers
# =========================


def _get_prefill_stage(stage_list: list[Any], source_stage_id: int) -> Any | None:
    if source_stage_id <= 0:
        return None
    source_stage = stage_list[source_stage_id]
    if not getattr(source_stage, "is_decode_only", False):
        return None
    prev_stage = stage_list[source_stage_id - 1]
    if getattr(prev_stage, "is_prefill_only", False) and prev_stage.engine_outputs is not None:
        return prev_stage
    return None


def _merge_pd_embeddings(
    decode_emb: torch.Tensor,
    decode_hid: torch.Tensor,
    prefill_mm: dict[str, Any],
    device: torch.device,
    expected_total: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge prefill prompt embeddings with decode generated embeddings.

    In PD mode the prefill engine processes the prompt and the decode engine
    generates tokens starting from position 1.  This function concatenates
    them, removing the overlapping token(s):

        merged = prefill[:P] + decode[overlap:]

    where overlap = P + D - expected_total.
    """
    try:
        p_emb = prefill_mm[_EMBED_LAYER_KEY].detach().to(device=device, dtype=torch.float)
        p_hid = prefill_mm[_HIDDEN_LAYER_KEY].detach().to(device=device, dtype=torch.float)
    except (KeyError, AttributeError, TypeError) as exc:
        available_keys = list(prefill_mm.keys()) if isinstance(prefill_mm, dict) else type(prefill_mm).__name__
        logger.error(
            "_merge_pd_embeddings: failed to extract prefill embeddings (%s). "
            "Expected keys %r and %r, got: %s. "
            "Falling back to decode-only embeddings – talker user-segment will be degraded.",
            exc,
            _EMBED_LAYER_KEY,
            _HIDDEN_LAYER_KEY,
            available_keys,
        )
        return decode_emb, decode_hid

    if p_emb.shape[0] == 0 or decode_emb.shape[0] == 0:
        return decode_emb, decode_hid

    raw_total = p_emb.shape[0] + decode_emb.shape[0]
    overlap = max(0, raw_total - expected_total) if expected_total is not None else 0

    merged_emb = torch.cat([p_emb, decode_emb[overlap:]], dim=0)
    merged_hid = torch.cat([p_hid, decode_hid[overlap:]], dim=0)
    return merged_emb, merged_hid


def _get_prefill_multimodal_output(prefill_stage: Any, output_index: int) -> dict[str, Any] | None:
    """Return multimodal_output dict from the PD prefill stage for a given batch index."""
    try:
        prefill_eos = prefill_stage.engine_outputs
        prefill_eo = prefill_eos[min(output_index, len(prefill_eos) - 1)]
        return prefill_eo.outputs[0].multimodal_output
    except Exception:
        return None


def _resolve_tts_token_embedding(
    key: str,
    *,
    thinker_mm: dict[str, Any],
    prefill_mm: dict[str, Any] | None,
    device: torch.device,
) -> torch.Tensor | None:
    """Return TTS BOS/EOS/PAD embedding tensors for the talker projection path.

    Values are taken from the current thinker (decode) ``multimodal_output``; in
    PD mode, missing keys may be filled from the paired prefill stage output.
    """
    val = thinker_mm.get(key)
    if val is None and prefill_mm is not None:
        val = prefill_mm.get(key)
    return val.detach().to(device=device, dtype=torch.float) if val is not None else None


# =========================
# Streaming input helpers
# =========================


@dataclass
class _Thinker2TalkerStreamingState:
    last_prompt_len: int = 0
    last_output_len: int = 0
    merged_sequences: list[int] = field(default_factory=list)


@dataclass
class _Qwen3OmniStreamingState:
    thinker2talker: _Thinker2TalkerStreamingState = field(default_factory=_Thinker2TalkerStreamingState)
    talker2code2wav_last_seq_len: int = 0


def _get_qwen3_streaming_state(
    request_id: str,
    streaming_context: Any | None,
) -> _Qwen3OmniStreamingState:
    bridge_states = getattr(streaming_context, "bridge_states", None)
    per_model_state = bridge_states.setdefault("qwen3_omni", {})
    state = per_model_state.get(request_id)
    if state is None:
        state = _Qwen3OmniStreamingState()
        per_model_state[request_id] = state
    return state


def _get_streaming_talker_tokens(
    request_id: str,
    prompt_token_ids: list[int],
    output_token_ids: list[int],
    new_prompt_len_snapshot: int | None = None,
    streaming_context: Any | None = None,
    *,
    clear_state: bool = False,
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Return streaming token slices and merged token views for thinker->talker.
       e.g. For the second streaming input request:
       merged_sequences: [input_prompt 1, output_tokens 1[:-1], input_prompt 2, output_tokens 2]
      thinker_input_ids: [input_prompt 1, output_tokens 1[:-1], input_prompt 2]
    Returns:
        inc_prompt: prompt token delta for this segment.
        inc_output: output token delta for this segment.
        merged_sequences: full thinker_sequences to send downstream.
        thinker_input_ids: full thinker_input_ids paired with merged_sequences.
    """
    state = _get_qwen3_streaming_state(request_id, streaming_context).thinker2talker
    if new_prompt_len_snapshot:
        prompt_token_ids = prompt_token_ids[:-new_prompt_len_snapshot]
    cur_prompt_len = len(prompt_token_ids)
    cur_output_len = len(output_token_ids)

    inc_prompt = prompt_token_ids[state.last_prompt_len :]
    inc_output = output_token_ids[state.last_output_len :]
    delta_sequences = inc_prompt + inc_output
    cached_sequences = state.merged_sequences

    merged_sequences = cached_sequences + delta_sequences
    thinker_input_ids = cached_sequences + inc_prompt

    # Persist history for next segment. Drop the latest sampled token to keep
    # thinker_input_ids / thinker_sequences alignment with next-step append.
    cached_sequences.extend(delta_sequences[:-1])

    state.last_prompt_len = cur_prompt_len
    state.last_output_len = cur_output_len

    if clear_state:
        state.last_prompt_len = 0
        state.last_output_len = 0
        state.merged_sequences.clear()

    return inc_prompt, inc_output, merged_sequences, thinker_input_ids


def _get_streaming_codec_delta_len(
    cur_seq_len: int,
    request_id: str,
    talker_output: Any,
    streaming_context: Any | None = None,
) -> int:
    """Return newly added seq_len for talker->code2wav in streaming mode."""
    state = _get_qwen3_streaming_state(request_id, streaming_context)
    prev_seq_len = state.talker2code2wav_last_seq_len
    seq_len = cur_seq_len - prev_seq_len
    state.talker2code2wav_last_seq_len = cur_seq_len + 1
    if bool(getattr(talker_output, "finished", False)):
        # Final segment: clear history to avoid cross-session carry-over.
        state.talker2code2wav_last_seq_len = 0
    return seq_len


# =========================
# Thinker -> Talker
# =========================

def _get_prefill_part_state(transfer_manager: Any, request_id: str) -> dict[str, Any]:
    state_map = getattr(transfer_manager, "_prefill_part_state", None)
    if state_map is None:
        state_map = {}
        setattr(transfer_manager, "_prefill_part_state", state_map)
    if request_id not in state_map:
        state_map[request_id] = {"sent_prompt_tokens": 0}
    return state_map[request_id]


def _get_system_segment_ranges(prompt_token_ids: list[int]) -> list[tuple[int, int]]:
    """Return half-open (start, end) index ranges for system segments in the full prompt."""
    im_starts = [i for i in range(len(prompt_token_ids)) if prompt_token_ids[i] == _IM_START_TOKEN_ID]
    im_starts.append(len(prompt_token_ids))
    ranges: list[tuple[int, int]] = []
    for idx in range(len(im_starts) - 1):
        s, e = im_starts[idx], im_starts[idx + 1]
        if s + 1 < len(prompt_token_ids) and prompt_token_ids[s + 1] == _SYSTEM_TOKEN_ID:
            ranges.append((s, e))
    return ranges


def _filter_system_rows(
    prompt_token_ids: list[int],
    chunk_start: int,
    embeds: torch.Tensor,
    hidden: torch.Tensor,
) -> tuple[list[int], torch.Tensor, torch.Tensor]:
    """Remove system-segment rows from a chunk's embeddings.

    Args:
        prompt_token_ids: Full prompt token IDs for the request.
        chunk_start: Position of the first token of this chunk in the full prompt.
        embeds: [chunk_size, D] embedding tensor for the current chunk.
        hidden: [chunk_size, D] hidden state tensor for the current chunk.

    Returns:
        (filtered_full_ids, filtered_embeds, filtered_hidden):
          - filtered_full_ids: full prompt IDs with all system tokens removed.
            Used as thinker_input_ids / thinker_sequences sent to the talker.
          - filtered_embeds / filtered_hidden: current-chunk rows with system
            positions removed.
    """
    system_ranges = _get_system_segment_ranges(prompt_token_ids)
    if not system_ranges:
        return prompt_token_ids, embeds, hidden

    def _in_system(pos: int) -> bool:
        return any(s <= pos < e for s, e in system_ranges)

    # Full non-system IDs (constant across chunks, sent as override each step)
    filtered_full_ids = [prompt_token_ids[p] for p in range(len(prompt_token_ids)) if not _in_system(p)]

    # Per-chunk keep mask
    chunk_size = embeds.shape[0]
    keep_mask = torch.tensor(
        [not _in_system(chunk_start + i) for i in range(chunk_size)],
        dtype=torch.bool,
    )
    return filtered_full_ids, embeds[keep_mask], hidden[keep_mask]


def _fill_optional_fields(d: dict, request: OmniEngineCoreRequest) -> None:
    """Add speaker/language to an info dict if present on the request."""
    for key, fn in (("speaker", extract_speaker_from_request), ("language", extract_language_from_request)):
        val = fn(request)
        if val is not None:
            d[key] = val


def thinker2talker_async_chunk(  # noqa: C901
    transfer_manager: Any,
    pooling_output: dict[str, Any],
    request: OmniEngineCoreRequest,
    is_finished: bool = False,
) -> list[dict[str, Any]]:
    """
    Connector from Thinker → Talker supporting chunked prefill.
    Two phases:

      if n_decoded == 0 and not is_finished  (prefill):
        Emit all tokens immediately (including assistant part).
        Bootstrap pos 8 (tok0) will be zero-padded by the talker.
      else  (decode):
        Ordinary decode step.
    """
    request_id = request.external_req_id
    output_token_ids = _ensure_list(request.output_token_ids)
    logger.info(
        "[THINKER2TALKER] req=%s output_token_ids_len=%s is_finished=%s len_pooling_output_0=%s all_tokens_ids=%s prompt_token_ids=%s",
        request_id[-16:] if request_id else "N/A",
        len(output_token_ids) if output_token_ids is not None else 0,
        is_finished,
        len(pooling_output.get("0")) if pooling_output.get("0") is not None else 0,
        _ensure_list(request.all_token_ids)        if request.all_token_ids is not None else "N/A",
        _ensure_list(request.prompt_token_ids) if request.prompt_token_ids is not None else "N/A",
    )
    n_decoded = len(output_token_ids)

    prompt_token_ids = _ensure_list(request.prompt_token_ids)
    state = _get_prefill_part_state(transfer_manager, request_id)
    chunk_start = state["sent_prompt_tokens"]
    prefill_still_pending = chunk_start < len(prompt_token_ids)

    if prefill_still_pending and not (is_finished and n_decoded == 0):
        # ----- Prefill -----
        embeds_cpu = pooling_output.get(_EMBED_LAYER_KEY).detach().cpu()
        hidden_cpu = pooling_output.get(_HIDDEN_LAYER_KEY).detach().cpu()

        # Track the position of this chunk in the full prompt so we can
        # identify which rows belong to a system segment and drop them.
        # The talker's _compute_talker_prompt_ids_length already excludes
        # system tokens from the placeholder length, so embeddings and IDs
        # must be consistent: system rows must never reach the talker.
        state["sent_prompt_tokens"] = chunk_start + embeds_cpu.shape[0]

        filtered_ids, embeds_cpu, hidden_cpu = _filter_system_rows(
            prompt_token_ids, chunk_start, embeds_cpu, hidden_cpu
        )

        # Entire chunk was system tokens — nothing to forward this step.
        if embeds_cpu.shape[0] == 0:
            return None

        emit_info = {
            "thinker_prefill_embeddings": embeds_cpu,
            "thinker_hidden_states": hidden_cpu,
            "thinker_sequences": filtered_ids,
            "thinker_input_ids": filtered_ids,
            "override_keys": ["thinker_sequences", "thinker_input_ids"],
            "tts_bos_embed": pooling_output.get("tts_bos_embed").detach().cpu(),
            "tts_eos_embed": pooling_output.get("tts_eos_embed").detach().cpu(),
            "tts_pad_embed": pooling_output.get("tts_pad_embed").detach().cpu(),
            "finished": torch.tensor(False, dtype=torch.bool),
        }
        # Last prefill chunk: thinker already emitted the first output token
        # in this same scheduler step.  Forward the token IDs now so the talker
        # has the correct sequence length; the decode *embedding* is absent
        # here and will arrive in the very next decode step's pooling_output.
        if n_decoded >= 1 and output_token_ids:
            emit_info["thinker_output_token_ids"] = output_token_ids
            emit_info["override_keys"] = [
                "thinker_sequences", "thinker_input_ids", "thinker_output_token_ids"
            ]
        _fill_optional_fields(emit_info, request)
        return emit_info

    else:
        # ----- Decode -----
        talker_additional_info = {"finished": torch.tensor(is_finished, dtype=torch.bool)}
        _fill_optional_fields(talker_additional_info, request)
        if output_token_ids:
            talker_additional_info["override_keys"] = ["thinker_decode_embeddings", "thinker_output_token_ids"]
            talker_additional_info["thinker_decode_embeddings"] = pooling_output.get(_EMBED_LAYER_KEY).detach().cpu()
            talker_additional_info["thinker_output_token_ids"] = output_token_ids
        else:
            # Edge case: is_finished with no output tokens (request aborted mid-prefill).
            talker_additional_info["thinker_prefill_embeddings"] = pooling_output.get(_EMBED_LAYER_KEY).detach().cpu()
            talker_additional_info["thinker_hidden_states"] = pooling_output.get(_HIDDEN_LAYER_KEY).detach().cpu()
        return talker_additional_info


def thinker2talker(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any | None = None,
) -> list[OmniTokensPrompt]:
    """
    Process thinker outputs to create talker inputs.

    Workflow:
    1. Extract thinker's text generation outputs (token IDs + hidden states)
    2. Split hidden states into: prompt embeddings + generated embeddings
    3. Package for talker with additional information

    In PD disaggregation mode, merges prefill-stage prompt embeddings with
    decode-stage generated embeddings before handing off to the talker.

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

    # PD disaggregation: look up the preceding prefill stage (if any)
    source_stage_id = engine_input_source[0]
    prefill_stage = _get_prefill_stage(stage_list, source_stage_id)

    # Process each thinker output
    for i, thinker_output in enumerate(thinker_outputs):
        output = thinker_output.outputs[0]
        req_id = str(getattr(thinker_output, "request_id", f"idx-{i}"))
        prompt_token_ids = _ensure_list(thinker_output.prompt_token_ids)
        output_ids = _ensure_list(output.token_ids)
        is_streaming_session = bool(getattr(streaming_context, "enabled", False))
        if is_streaming_session:
            prompt_token_ids, output_ids, thinker_sequences, thinker_input_ids = _get_streaming_talker_tokens(
                req_id,
                prompt_token_ids,
                output_ids,
                getattr(streaming_context, "new_prompt_len_snapshot", None),
                streaming_context,
                clear_state=bool(getattr(thinker_output, "finished", False)),
            )
        else:
            thinker_sequences = prompt_token_ids + output_ids
            thinker_input_ids = prompt_token_ids
        # For streaming input, just send incremental prefill and hidden states tensor to talker
        # Equally applicable to non-streaming cases.
        new_seq_length = len(prompt_token_ids + output_ids) - 1
        thinker_mm = output.multimodal_output
        # Full thinker embedding sequence for the talker: single thinker engine in the
        # non-PD path; after optional merge with prefill-side tensors in PD mode.
        thinker_emb = thinker_mm[_EMBED_LAYER_KEY].detach().to(device=device, dtype=torch.float)[-new_seq_length:]
        thinker_hid = thinker_mm[_HIDDEN_LAYER_KEY].detach().to(device=device, dtype=torch.float)[-new_seq_length:]

        prefill_mm: dict[str, Any] | None = None
        if prefill_stage is not None:
            prefill_mm = _get_prefill_multimodal_output(prefill_stage, i)

        if prefill_mm is not None:
            expected_total = len(prompt_token_ids) + len(output_ids)
            try:
                thinker_emb, thinker_hid = _merge_pd_embeddings(
                    thinker_emb, thinker_hid, prefill_mm, device, expected_total=expected_total
                )
            except Exception as exc:
                logger.warning("[PD] Could not merge prefill embeddings: %s", exc)

        info = {
            "thinker_prefill_embeddings": thinker_emb,
            "thinker_hidden_states": thinker_hid,
            "thinker_sequences": thinker_sequences,  # the thinker_sequences is the whole ids
            "thinker_input_ids": thinker_input_ids,
            # Provide thinker-side TTS token embeddings for talker projection
            "tts_bos_embed": _resolve_tts_token_embedding(
                "tts_bos_embed", thinker_mm=thinker_mm, prefill_mm=prefill_mm, device=device
            ),
            "tts_eos_embed": _resolve_tts_token_embedding(
                "tts_eos_embed", thinker_mm=thinker_mm, prefill_mm=prefill_mm, device=device
            ),
            "tts_pad_embed": _resolve_tts_token_embedding(
                "tts_pad_embed", thinker_mm=thinker_mm, prefill_mm=prefill_mm, device=device
            ),
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
        logger.info(f"[CODE2WAV_DIAG] req={request_id[-16:] if request_id else 'N/A'} DROP=no_key keys={list(pooling_output.keys())[:5]}")
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
        logger.warning(f"req={request_id[-16:] if request_id else 'N/A'} DROP=none")
        return None
    if isinstance(code_predictor_codes, torch.Tensor):
        if code_predictor_codes.numel() == 0:
            logger.warning(f"req={request_id[-16:] if request_id else 'N/A'} code_predictor_codes is empty 0")
            return None
    elif hasattr(code_predictor_codes, "__len__"):
        if len(code_predictor_codes) == 0:
            logger.warning(f"req={request_id[-16:] if request_id else 'N/A'} code_predictor_codes is empty 1")
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
    streaming_context: Any | None = None,
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
    for i, talker_output in enumerate(talker_outputs):
        output = talker_output.outputs[0]
        req_id = str(getattr(talker_output, "request_id", f"idx-{i}"))
        cur_seq_len = len(output.token_ids) - 1
        seq_len = cur_seq_len
        is_streaming_session = bool(getattr(streaming_context, "enabled", False))
        if is_streaming_session:
            seq_len = _get_streaming_codec_delta_len(cur_seq_len, req_id, talker_output, streaming_context)
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
