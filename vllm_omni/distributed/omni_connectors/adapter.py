# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from collections.abc import Callable
from typing import Any

from vllm_omni.metrics import OrchestratorAggregator

from .utils.logging import get_connector_logger

logger = get_connector_logger(__name__)


def try_send_via_connector(
    connector: Any,
    stage_id: int,
    next_stage_id: int,
    req_id: str,
    next_inputs: Any,
    sampling_params: Any,
    original_prompt: Any,
    next_stage_queue_submit_fn: Callable[[dict[str, Any]], None],
    metrics: OrchestratorAggregator,
) -> bool:
    """
    Attempts to send data via OmniConnector.
    Returns True if successful, False otherwise.
    Encapsulates the logic of preparing payload, sending via connector,
    sending notification, and recording metrics.
    """
    try:
        t0 = time.time()

        # Strip non-serializable multimodal feature fields from original_prompt
        # before including it in metadata.  After stage-0 runs, the TokPrompt
        # returned by render_chat_async may carry processed multimodal features
        # (mm_kwargs, mm_placeholders, mm_hashes) that contain MultiModalKwargsItems
        # objects, which are not supported by OmniMsgpackEncoder.  The receiving
        # side (try_recv_via_connector) only extracts "engine_inputs" from the
        # payload and never uses "original_prompt", so stripping these fields
        # only affects debug metadata and is safe.
        _MM_FEATURE_KEYS = frozenset({"mm_kwargs", "mm_placeholders", "mm_hashes"})
        if isinstance(original_prompt, dict) and any(k in original_prompt for k in _MM_FEATURE_KEYS):
            safe_prompt = {k: v for k, v in original_prompt.items() if k not in _MM_FEATURE_KEYS}
        else:
            safe_prompt = original_prompt

        # Prepare data for connector
        payload_data = {
            "engine_inputs": next_inputs,
            "sampling_params": sampling_params,
            "metadata": {
                "original_prompt": safe_prompt,
                "stage_transition": f"{stage_id}->{next_stage_id}",
                "timestamp": time.time(),
            },
        }

        # Send data via connector
        success, serialized_size, metadata = connector.put(str(stage_id), str(next_stage_id), str(req_id), payload_data)

        if success:
            # Send lightweight notification via queue
            notify_payload = {
                "type": "generate",
                "request_id": req_id,
                "sampling_params": sampling_params,
                "from_connector": True,
                "from_stage": str(stage_id),
                "to_stage": str(next_stage_id),
                "sent_ts": time.time(),
            }
            # Merge connector metadata (e.g. shm handle or inline data) into queue payload
            if metadata:
                notify_payload["connector_metadata"] = metadata

            next_stage_queue_submit_fn(notify_payload)

            t1 = time.time()
            tx_ms = (t1 - t0) * 1000.0

            metrics.on_forward(
                stage_id,
                next_stage_id,
                req_id,
                serialized_size,  # Use size from connector
                float(tx_ms),
                True,  # Mark as using connector
            )
            return True
        else:
            # If put returned False, we let the caller handle fallback
            return False

    except Exception as e:
        logger.warning(
            "[Orchestrator] OmniConnector failed for req %s: %s; falling back to queue",
            req_id,
            e,
        )
        return False


def try_recv_via_connector(
    task: dict[str, Any],
    connectors: dict[Any, Any],
    stage_id: int,
) -> tuple[Any, dict[str, Any] | None]:
    """
    Attempts to resolve input data from either connector or IPC.
    Returns (engine_inputs, rx_metrics) or (None, None) if failed/skipped.
    """
    rid = task["request_id"]

    if task.get("from_connector"):
        from_stage = task.get("from_stage")
        to_stage = str(stage_id)

        if not from_stage:
            logger.error(
                "[Stage-%s] 'from_connector' is true but 'from_stage' is missing for request %s", stage_id, rid
            )
            return None, None

        # Get connector for this edge
        connector_key = (from_stage, to_stage)
        connector = connectors.get(connector_key)

        if connector:
            try:
                # Get data from connector with timeout
                _t_start = time.time()
                connector_metadata = task.get("connector_metadata")
                payload = connector.get(from_stage, to_stage, str(rid), metadata=connector_metadata)
                _t_end = time.time()

                if payload:
                    if isinstance(payload, tuple):
                        payload_data, serialized_size = payload
                    else:
                        payload_data = payload
                        serialized_size = len(connector.serialize_obj(payload_data))
                else:
                    payload_data = None
                    serialized_size = 0

                if payload_data and isinstance(payload_data, dict):
                    ein = payload_data.get("engine_inputs")
                    decode_ms = (_t_end - _t_start) * 1000.0

                    rx_metrics = {"rx_decode_time_ms": decode_ms, "rx_transfer_bytes": serialized_size}
                    return ein, rx_metrics
                else:
                    logger.error(
                        "[Stage-%s] Failed to get data from connector for request %s or payload is empty", stage_id, rid
                    )
                    return None, None
            except Exception as e:
                logger.error("[Stage-%s] Error retrieving data from connector for request %s: %s", stage_id, rid, e)
                return None, None
        else:
            logger.error(
                "[Stage-%s] No connector found for edge %s -> %s for request %s", stage_id, from_stage, to_stage, rid
            )
            return None, None
    else:
        # Data comes from queue as usual (e.g. seed request for Stage-0)
        # Since fallback logic is deprecated, we assume this is a direct inputs payload.
        # We still need to decode it if it used SHM (via legacy stage_utils logic, or new shm_connector format)
        # For Stage-0 specifically, 'engine_inputs' is often directly in the task dict.

        # Try to use the new stage_utils which uses OmniSerializer
        from vllm_omni.entrypoints.stage_utils import maybe_load_from_ipc_with_metrics

        try:
            ein, metrics = maybe_load_from_ipc_with_metrics(task, "engine_inputs", "engine_inputs_shm")
            # If metrics are empty or zero, we might want to populate dummy metrics
            return ein, metrics
        except Exception:
            # If engine_inputs is missing, it might be a different kind of payload,
            # but for Stage-0 seed it should be there.
            # We'll return None to let caller handle error if strictly required.
            return None, None


def compute_talker_prompt_ids_length(prompt_ids: list[int]) -> int:
    """Compute the length of the talker prompt ids.

    Args:
        prompt_ids: The prompt ids tensor.

    Returns:
        The length of the talker prompt ids.
    """
    im_start_token_id = 151644
    system_token_id = 8948
    user_token_id = 872
    assistant_token_id = 77091
    im_start_indexes = [i for i in range(len(prompt_ids)) if prompt_ids[i] == im_start_token_id]
    im_start_indexes.append(len(prompt_ids))
    sum_user_len = 0
    assistant_len = 0
    for i in range(len(im_start_indexes) - 1):
        s = im_start_indexes[i]
        e = im_start_indexes[i + 1]
        role = prompt_ids[s + 1]
        if role == system_token_id:
            continue
        elif role == user_token_id:
            sum_user_len += e - s
        elif role == assistant_token_id and i == len(im_start_indexes) - 2:
            assistant_len += 9  # 3 + 4 + 1 + 1
        else:
            pass

    return sum_user_len + assistant_len


def compute_first_chunk_talker_len(
    prompt_ids: list[int],
    thinker_max_batched_tokens: int,
) -> tuple[int, int]:
    """Compute (first_chunk_talker_len, total_talker_placeholder_len).

    In async_chunk mode the thinker sends embeddings in batches of
    ``thinker_max_batched_tokens``.  The first batch covers tokens
    [0 .. min(max_batched, total)-1].  After filtering out the system
    segment, the remaining non-system tokens correspond to the talker
    placeholder tokens that will be available from the very first chunk.

    Pre-warming the talker with only ``first_chunk_talker_len`` tokens
    (instead of the full ``total_talker_placeholder_len``) ensures the
    talker scheduler never schedules more tokens than there are valid
    thinker embeddings available for that step, avoiding KV-cache
    corruption from garbage placeholder embeddings.

    Args:
        prompt_ids: Thinker prompt token IDs.
        thinker_max_batched_tokens: ``max_num_batched_tokens`` of the
            thinker's scheduler config.

    Returns:
        ``(first_chunk_talker_len, total_talker_placeholder_len)``
    """
    total_talker_len = compute_talker_prompt_ids_length(prompt_ids)

    im_start_token_id = 151644
    system_token_id = 8948

    # Locate system segment(s) and sum their lengths.  The system turn is
    # always first in Qwen3-Omni, so we stop counting once we hit a non-system
    # segment.
    im_start_indexes = [i for i in range(len(prompt_ids)) if prompt_ids[i] == im_start_token_id]
    im_start_indexes.append(len(prompt_ids))
    system_prefix_len = 0
    for i in range(len(im_start_indexes) - 1):
        s = im_start_indexes[i]
        e = im_start_indexes[i + 1]
        if len(prompt_ids) > s + 1 and prompt_ids[s + 1] == system_token_id:
            system_prefix_len += e - s
        else:
            break  # system is always first

    total_prompt_len = len(prompt_ids)
    if thinker_max_batched_tokens <= 0:
        return max(1, total_talker_len), total_talker_len

    # Find the first thinker batch that contains at least one non-system token.
    # Each batch spans [batch_start .. batch_start + max_batched - 1].
    # batch_index of the first non-system token = floor(system_prefix_len / max_batched)
    first_non_sys_batch_start = (system_prefix_len // thinker_max_batched_tokens) * thinker_max_batched_tokens
    first_non_sys_batch_end = min(first_non_sys_batch_start + thinker_max_batched_tokens, total_prompt_len)
    first_chunk_non_system = max(0, first_non_sys_batch_end - max(system_prefix_len, first_non_sys_batch_start))

    # Non-system thinker tokens map 1:1 to talker placeholder tokens (up to
    # the bootstrap boundary).  Clip to total_talker_len to be safe.
    first_chunk_talker_len = min(first_chunk_non_system, total_talker_len)
    first_chunk_talker_len = max(1, first_chunk_talker_len)

    return first_chunk_talker_len, total_talker_len
