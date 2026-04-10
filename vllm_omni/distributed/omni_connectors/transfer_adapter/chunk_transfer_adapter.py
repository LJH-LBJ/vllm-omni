# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from collections import defaultdict, deque
from typing import Any

import torch
from vllm.v1.request import Request, RequestStatus

from ..factory import OmniConnectorFactory
from ..utils.config import ConnectorSpec
from ..utils.logging import get_connector_logger
from .base import OmniTransferAdapterBase

logger = get_connector_logger(__name__)


class OmniChunkTransferAdapter(OmniTransferAdapterBase):
    """Chunk-level transfer adapter for Omni connector pipelines.

    This class coordinates per-request chunk exchange between adjacent stages,
    and implements asynchronous get/put of chunks via background threads.
    It tracks per-request chunk indices for put/get, and accumulates
    payloads across chunks (concatenating tensors/lists in AR mode). It also
    caches prompt token ids and additional information for scheduler use.

    Scheduler integration is handled via WAITING_FOR_CHUNK transitions:
    requests are moved to waiting for chunk deque while polling, then restored
    to waiting/running queues once a chunk arrives. The requests will finish
    loading chunk util detecting the payload "finished" flag.

    The base class owns background recv/save loops; load/save only enqueue
    work and return immediately.
    """

    def __init__(self, vllm_config: Any):
        model_config = vllm_config.model_config
        self.scheduler_max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self.connector = self.create_connector(model_config)
        super().__init__(model_config)
        self.model_mode = getattr(model_config, "worker_type", None) or "ar"
        # State specific to Chunk management
        self.custom_process_next_stage_input_func = None
        custom_process_next_stage_input_func = getattr(model_config, "custom_process_next_stage_input_func", None)
        if custom_process_next_stage_input_func:
            module_path, func_name = custom_process_next_stage_input_func.rsplit(".", 1)
            module = importlib.import_module(module_path)
            self.custom_process_next_stage_input_func = getattr(module, func_name)
        # mapping for request id and chunk id
        self.put_req_chunk: dict[str, int] = defaultdict(int)
        self.get_req_chunk: dict[str, int] = defaultdict(int)
        self.finished_requests: set[str] = set()
        self.request_payload = {}
        self.code_prompt_token_ids: dict[str, list[torch.Tensor]] = defaultdict(list)
        self.request_ids_mapping: dict[str, str] = {}

        self.waiting_for_chunk_waiting_requests: deque[Any] = deque()
        self.waiting_for_chunk_running_requests: deque[Any] = deque()
        self.requests_with_ready_chunks = set()
        # Requests currently in chunked-prefill mode (thinker_prefill_complete
        # is False).  Used by process_pending_chunks to suppress spurious
        # output_token_ids that `_update_request_with_output` adds between
        # chunk polls — without this the scheduler schedules 1 extra token
        # that the model cannot serve (prefill embedding OOB).
        self._chunked_prefill_reqs: set[str] = set()

    @classmethod
    def create_connector(cls, model_config: Any):
        connector_config = getattr(model_config, "stage_connector_config", None)
        if connector_config is None:
            connector_config = {}
        elif not isinstance(connector_config, dict):
            connector_config = {
                "name": getattr(connector_config, "name", None),
                "extra": getattr(connector_config, "extra", {}),
            }

        connector_specs = ConnectorSpec(
            name=connector_config.get("name", "SharedMemoryConnector"),
            extra=connector_config.get("extra", {}),
        )
        return OmniConnectorFactory.create_connector(connector_specs)

    def load_async(self, request: Request):
        """Register a request for asynchronous chunk retrieval.

        This method does not read from the connector directly. It records
        request metadata and enqueues the request id for the background
        receive loop to poll.

        Stage-0 has no upstream producer, so this call is a no-op there.

        Args:
            request: The request object needing data.
        """
        stage_id = self.connector.stage_id

        if stage_id == 0:
            return
        if not hasattr(request, "additional_information"):
            request.additional_information = None
        self._cancelled_load_reqs.discard(request.request_id)
        self._pending_load_reqs.append(request)
        with self._recv_cond:
            self._recv_cond.notify()

    def save_async(
        self,
        pooling_output: torch.Tensor | None = None,
        request: Request | None = None,
    ):
        """Build and enqueue one chunk for asynchronous sending.

        Payload extraction happens in ``_send_single_request`` on the
        background save_loop thread.

        Args:
            pooling_output: Partial pooling output dictionary
            request: Request object
        """
        task = {
            "pooling_output": pooling_output,
            "request": request,
            "is_finished": request.is_finished(),
        }
        self._pending_save_reqs.append(task)
        with self._save_cond:
            self._save_cond.notify()

    def _poll_single_request(self, request: Request):
        stage_id = self.connector.stage_id
        target_stage_id = stage_id - 1
        req_id = request.request_id
        chunk_id = self.get_req_chunk[req_id]
        external_req_id = self.request_ids_mapping.get(req_id, req_id)
        connector_get_key = f"{external_req_id}_{target_stage_id}_{chunk_id}"

        # Use timeout=0 for non-blocking poll
        try:
            result = self.connector.get(
                str(target_stage_id),
                str(stage_id),
                connector_get_key,
            )
        except Exception as e:
            logger.error(f"SharedMemoryConnector get failed for req {connector_get_key}: {e}")
            return False

        if result is None:
            return False
        payload_data, size = result

        if payload_data:
            # Update connector state
            self.get_req_chunk[req_id] += 1

            if self.model_mode == "ar":
                merged_payload = self._update_request_payload(external_req_id, payload_data)
                request.additional_information = merged_payload

                if bool(merged_payload.get("finished", False)):
                    self.finished_requests.add(req_id)
                # Derive how many talker prefill tokens are currently buildable
                # from accumulated thinker prefill chunks, and expose only that
                # amount to the scheduler each round.
                # When partial_len >= full_len, thinker prefill has fully covered
                # the talker prompt and we can mark prefill as complete.
                thinker_input_ids = merged_payload.get("thinker_input_ids")
                thinker_prefill_embeddings = merged_payload.get(
                    "thinker_prefill_embeddings"
                )
                if (
                    thinker_input_ids is not None
                    and isinstance(thinker_prefill_embeddings, torch.Tensor)
                ):
                    try:
                        from vllm_omni.model_executor.stage_input_processors.qwen3_omni import (
                            _compute_partial_talker_prompt_ids_length,
                        )

                        partial_len = _compute_partial_talker_prompt_ids_length(
                            thinker_input_ids,
                            int(thinker_prefill_embeddings.shape[0]),
                        )
                        full_len = _compute_partial_talker_prompt_ids_length(
                            thinker_input_ids,
                            len(thinker_input_ids),
                        )
                        merged_payload["thinker_prefill_complete"] = (
                            partial_len >= full_len
                        )

                        # Reset prefill_done when thinker has not fully covered
                        # the talker prompt.  The model may have set it to True
                        # on a previous step because its cached prefill tensors
                        # were exhausted; now that more thinker data has arrived
                        # we must re-enter the prefill path.
                        if not merged_payload["thinker_prefill_complete"]:
                            merged_payload["prefill_done"] = False
                        else:
                            # Thinker prefill is complete: remove any stale
                            # prefill_done=False that accumulated from earlier
                            # partial chunks.  The model's own update_dict sets
                            # prefill_done=True when it finishes consuming the
                            # prefill cache; that value must not be overwritten
                            # by a lingering False from the connector payload.
                            merged_payload.pop("prefill_done", None)

                        # Track prefill/decode state for per-step output
                        # suppression in process_pending_chunks.
                        thinker_prefill_complete = merged_payload.get(
                            "thinker_prefill_complete", False
                        )
                        # Only leave _chunked_prefill_reqs when the
                        # *talker* has actually consumed ALL prefill
                        # tokens (num_computed >= full_len).  The
                        # thinker may signal prefill_complete while
                        # the talker still has an outstanding last
                        # prefill chunk; during that window
                        # _update_request_with_output still appends
                        # tokens that must be suppressed.
                        num_computed = getattr(
                            request, "num_computed_tokens", 0
                        )
                        if (
                            not thinker_prefill_complete
                            or num_computed < full_len
                        ):
                            self._chunked_prefill_reqs.add(req_id)
                        else:
                            self._chunked_prefill_reqs.discard(req_id)

                        # During prefill, clear stray output tokens so
                        # the scheduler does not count them as decode
                        # progress; after prefill, let the AR scheduler
                        # own output_token_ids growth.
                        if not thinker_prefill_complete:
                            out_ids = getattr(
                                request, "_output_token_ids", None
                            )
                            if out_ids is not None and len(out_ids) > 0:
                                out_ids.clear()

                        logger.info(
                            "[ChunkPrefillState] req=%s chunk=%d "
                            "partial=%d full=%d computed=%d "
                            "output_len=%d prompt_len=%d "
                            "prefill_complete=%s",
                            req_id,
                            chunk_id,
                            partial_len,
                            full_len,
                            int(num_computed),
                            len(
                                getattr(
                                    request, "_output_token_ids", []
                                )
                            ),
                            partial_len,
                            merged_payload["thinker_prefill_complete"],
                        )

                        is_finished_flag = bool(
                            merged_payload.get("finished", False)
                        )
                        if partial_len == 0 and not is_finished_flag:
                            # Keep polling until we have at least
                            # one schedulable talker token.
                            # Return False so recv_loop re-enqueues the
                            # request to keep polling.
                            return False
                        if partial_len == 0 and is_finished_flag:
                            partial_len = 1

                        # If partial_len has not grown beyond what the
                        # scheduler already consumed and thinker is not
                        # finished, this chunk carries no new prefill
                        # progress for talker.  Stay in WAITING_FOR_CHUNK
                        # to avoid the scheduler spinning on +1 padding
                        # tokens.
                        if (
                            not is_finished_flag
                            and not merged_payload["thinker_prefill_complete"]
                            and num_computed >= partial_len
                        ):
                            logger.debug(
                                "[ChunkPrefillStall] req=%s partial=%d "
                                "computed=%d — no new prefill data, "
                                "staying in WAITING_FOR_CHUNK",
                                req_id,
                                partial_len,
                                num_computed,
                            )
                            # Return False so recv_loop re-enqueues the
                            # request into _pending_load_reqs and keeps
                            # polling for the next thinker chunk.
                            return False

                        # Scheduler dispatches only the "currently
                        # processable" amount of prefill to Talker.
                        # After talker finishes prefill (num_computed >=
                        # full_len), do NOT overwrite prompt/all_token_ids:
                        # the AR scheduler owns scheduling via
                        # output_token_ids, and resetting _all_token_ids
                        # would destroy accumulated decode outputs.
                        talker_prefill_done = (
                            thinker_prefill_complete
                            and num_computed >= full_len
                        )
                        if not talker_prefill_done:
                            new_prompt = [0] * partial_len
                            request.prompt_token_ids = new_prompt
                            # Single assignment avoids intermediate
                            # empty state from two-step clear+extend.
                            request._all_token_ids = list(new_prompt)
                            # Also clear output_token_ids so a racing
                            # append_output_token_ids cannot grow
                            # _all_token_ids beyond the prompt length.
                            out = getattr(
                                request, "_output_token_ids", None
                            )
                            if out is not None:
                                out.clear()
                    except Exception as e:
                        logger.debug(
                            "Failed to update dynamic talker "
                            "prompt length for req %s: %s",
                            req_id,
                            e,
                        )
            else:
                if payload_data.get("finished"):
                    self.finished_requests.add(req_id)

                new_ids = payload_data.get("code_predictor_codes", [])
                request.prompt_token_ids = new_ids
                # Preserve previously attached request metadata (e.g. prompt
                # conditioning tensors) and update only per-chunk fields.
                prev_info = getattr(request, "additional_information", None)
                info = dict(prev_info) if isinstance(prev_info, dict) else {}
                for key, value in payload_data.items():
                    if key in {"code_predictor_codes", "finished"}:
                        continue
                    info[key] = value
                request.additional_information = info
                request.num_computed_tokens = 0

                # Empty chunk with more data expected: keep polling.
                if not new_ids and not payload_data.get("finished"):
                    return True

            # Mark as finished for consumption
            self._finished_load_reqs.add(req_id)
            logger.debug(f"[Stage-{stage_id}] Received one chunk for key {connector_get_key}")
            return True

        return False

    def _update_request_payload(self, req_id: str, payload_data: dict[str, Any]) -> dict[str, Any]:
        """Update the payload data for a request in the connector.

        Args:
            connector: OmniConnectorBase instance
            req_id: Request ID to update
            payload_data: New payload data to store
        """
        if req_id not in self.request_payload:
            self.request_payload[req_id] = payload_data
            return payload_data
        origin_payload = self.request_payload[req_id]
        merged_payload = dict(origin_payload)
        # Use .get() to avoid mutating the caller's dict.
        override_keys = payload_data.get("override_keys", [])
        for key, value in payload_data.items():
            if key in ("finished", "override_keys"):
                continue
            elif key in override_keys:
                merged_payload[key] = value
            elif isinstance(value, torch.Tensor) and key in origin_payload:
                merged_payload[key] = torch.cat([origin_payload[key], value], dim=0)
            elif isinstance(value, list) and key in origin_payload:
                merged_payload[key] = origin_payload[key] + value
            else:
                merged_payload[key] = value

        # finished is terminal metadata and should always reflect latest chunk.
        if "finished" in payload_data:
            merged_payload["finished"] = payload_data["finished"]

        self.request_payload[req_id] = merged_payload
        return merged_payload

    def _send_single_request(self, task: dict):
        pooling_output = task["pooling_output"]
        request = task["request"]
        is_finished = task["is_finished"]
        stage_id = self.connector.stage_id
        next_stage_id = stage_id + 1
        external_req_id = request.external_req_id
        chunk_id = self.put_req_chunk[external_req_id]
        connector_put_key = f"{external_req_id}_{stage_id}_{chunk_id}"
        # Process payload in save_loop thread
        payload_data = None
        if self.custom_process_next_stage_input_func:
            try:
                payload_data = self.custom_process_next_stage_input_func(
                    transfer_manager=self,
                    pooling_output=pooling_output,
                    request=request,
                    is_finished=is_finished,
                )

            except Exception as e:
                logger.error(f"Failed to use custom_process_input_func for payload extraction: {e}")

        if not payload_data:
            return

        success, size, metadata = self.connector.put(
            from_stage=str(stage_id),
            to_stage=str(next_stage_id),
            put_key=connector_put_key,
            data=payload_data,
        )

        if success:
            self.put_req_chunk[external_req_id] += 1
            logger.debug(f"[Stage-{stage_id}] Sent {connector_put_key}")
            finished_flag = payload_data.get("finished")
            is_payload_finished = False
            if isinstance(finished_flag, torch.Tensor):
                is_payload_finished = finished_flag.numel() == 1 and bool(finished_flag.item())
            elif finished_flag is not None:
                is_payload_finished = bool(finished_flag)

            # Reclaim per-request async state only after the terminal payload
            # has been sent successfully. This avoids cleanup->save races.
            if is_payload_finished:
                self.cleanup(request.request_id, external_req_id)

        if is_finished:
            self.code_prompt_token_ids.pop(external_req_id, None)
            cached_ic = getattr(self, "_cached_ic", None)
            if cached_ic is not None:
                cached_ic.pop(external_req_id, None)

    ########################################################################
    # Cleanup
    ########################################################################

    def cleanup_receiver(self, request_id: str) -> None:
        """Reclaim receiver-side per-request state (keyed by internal id).

        Safe to call from the scheduler even when ``save_async()`` has
        enqueued work that the background thread has not yet processed,
        because it only touches receiver-side dictionaries.

        Idempotent: calling with an already-cleaned or unknown id is safe.
        """
        self.finished_requests.discard(request_id)
        self.get_req_chunk.pop(request_id, None)
        self.requests_with_ready_chunks.discard(request_id)
        self.request_ids_mapping.pop(request_id, None)
        self._chunked_prefill_reqs.discard(request_id)
        self._cancelled_load_reqs.add(request_id)
        self._finished_load_reqs.discard(request_id)

    def cleanup_sender(self, external_req_id: str) -> None:
        """Reclaim sender-side per-request state (keyed by external id).

        Must only be called after the terminal chunk has actually been
        sent (i.e. from ``_send_single_request``), not before.

        Idempotent: calling with an already-cleaned or unknown id is safe.
        """
        self.put_req_chunk.pop(external_req_id, None)
        self.request_payload.pop(external_req_id, None)
        self.code_prompt_token_ids.pop(external_req_id, None)

        cached_ic = getattr(self, "_cached_ic", None)
        if cached_ic is not None:
            cached_ic.pop(external_req_id, None)

    def cleanup(
        self,
        request_id: str,
        external_req_id: str | None = None,
    ) -> None:
        """Reclaim all per-request state after a request finishes.

        Idempotent: calling with an already-cleaned or unknown id is safe.

        Args:
            request_id: Internal request id (receive / scheduler side key).
            external_req_id: External request id (send / payload side key).
                When *None*, looked up from ``request_ids_mapping``.
        """
        if external_req_id is None:
            external_req_id = self.request_ids_mapping.get(request_id, request_id)

        self.cleanup_receiver(request_id)
        self.cleanup_sender(external_req_id)

    ########################################################################
    # Schedule Helper
    ########################################################################

    def process_pending_chunks(
        self,
        waiting_queue: Any,
        running_queue: list[Request],
        requests: dict[str, Request] | None = None,
    ) -> None:
        """
        Process pending chunks for waiting and running queues.
        """
        if self.connector.stage_id == 0:
            return

        # Suppress spurious output tokens for requests still in
        # chunked-prefill.  Between chunk polls the AR scheduler's
        # _update_request_with_output appends sampled tokens which
        # make num_tokens > num_computed, tricking the scheduler
        # into scheduling a decode-like step the model cannot serve
        # (prefill embeddings at that index don't exist yet).
        if requests is not None:
            for req_id in list(self._chunked_prefill_reqs):
                request = requests.get(req_id)
                if request is None:
                    self._chunked_prefill_reqs.discard(req_id)
                    continue
                # Proactively remove from set once talker has
                # consumed all prefill tokens — avoids lingering
                # output suppression during early decode steps.
                num_comp = getattr(request, "num_computed_tokens", 0)
                num_prompt = len(
                    getattr(request, "prompt_token_ids", None) or []
                )
                if num_comp >= num_prompt > 0:
                    self._chunked_prefill_reqs.discard(req_id)
                    continue
                out_ids = getattr(request, "_output_token_ids", None)
                if out_ids is not None and len(out_ids) > 0:
                    out_ids.clear()
                # After clearing spurious output tokens, sync
                # _all_token_ids back to prompt length.  A race
                # between the recv_loop (which resets _all_token_ids
                # to the new prompt) and update_from_output (which
                # appends a sampled token) can leave _all_token_ids
                # longer than the prompt — causing the scheduler to
                # over-schedule and the talker to OOB.
                prompt = getattr(request, "prompt_token_ids", None)
                all_ids = getattr(request, "_all_token_ids", None)
                if (
                    prompt is not None
                    and all_ids is not None
                    and len(all_ids) != len(prompt)
                ):
                    request._all_token_ids = list(prompt)

        self._process_chunk_queue(
            waiting_queue, self.waiting_for_chunk_waiting_requests, RequestStatus.WAITING, self._finished_load_reqs
        )
        self._process_chunk_queue(
            running_queue, self.waiting_for_chunk_running_requests, RequestStatus.RUNNING, self._finished_load_reqs
        )
        while len(running_queue) > self.scheduler_max_num_seqs:
            request = running_queue.pop()
            request.status = RequestStatus.PREEMPTED
            waiting_queue.prepend_requests([request])

    def restore_queues(self, waiting_queue: Any, running_queue: list[Request]) -> None:
        """
        Restore requests waiting for chunk to the waiting and running queues.
        """
        # Add request waiting for chunk to the waiting and running queue
        for request in self.waiting_for_chunk_waiting_requests:
            waiting_queue.add_request(request)
        self.waiting_for_chunk_waiting_requests = deque()

        if self.waiting_for_chunk_running_requests:
            running_queue.extend(self.waiting_for_chunk_running_requests)
        self.waiting_for_chunk_running_requests = deque()

    def postprocess_scheduler_output(
        self,
        scheduler_output: Any,
        requests: dict[str, Request] | None = None,
    ) -> None:
        """
        Add additional info for cached requests and
        clean up ready chunks from scheduler output.
        """
        if requests is not None:
            self.attach_cached_additional_information(scheduler_output, requests)

        # --- DIAG: log what the scheduler is sending ---
        new_reqs = getattr(scheduler_output, "scheduled_new_reqs", [])
        for nr in (new_reqs or []):
            rid = getattr(nr, "req_id", None)
            nc = getattr(nr, "num_computed_tokens", "?")
            plen = len(getattr(nr, "prompt_token_ids", []) or [])
            logger.info(
                "[DIAG-SCHED] NEW req=%s num_computed=%s prompt_len=%d",
                rid, nc, plen,
            )
        cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
        if cached and getattr(cached, "req_ids", None):
            nc_list = getattr(cached, "num_computed_tokens", [])
            for idx, rid in enumerate(cached.req_ids):
                nc_val = nc_list[idx] if idx < len(nc_list) else "?"
                req = (requests or {}).get(rid)
                req_nc = getattr(req, "num_computed_tokens", "?") if req else "?"
                logger.info(
                    "[DIAG-SCHED] CACHED req=%s snapshot_num_computed=%s "
                    "live_req.num_computed=%s prompt_len=%d",
                    rid, nc_val, req_nc,
                    len(getattr(req, "prompt_token_ids", []) or []) if req else 0,
                )
        # --- END DIAG ---

        self._clear_chunk_ready(scheduler_output)

    @staticmethod
    def attach_cached_additional_information(scheduler_output: Any, requests: dict[str, Request]) -> None:
        cached_reqs = getattr(scheduler_output, "scheduled_cached_reqs", None)
        if not cached_reqs:
            return
        if not hasattr(cached_reqs, "additional_information"):
            cached_reqs.additional_information = {}
        # Propagate current prompt_token_ids into the cached-request data so
        # that the model runner's _update_states can sync req_state and trigger
        # zero-filling for async-chunk talker requests.
        #
        # Background: OmniARScheduler wraps scheduled_new_reqs in
        # OmniNewRequestData but leaves scheduled_cached_reqs as the base
        # vLLM CachedRequestData, which has no prompt_token_ids field.
        # When a thinker chunk arrives and the talker's prompt grows from
        # partial_len to a larger partial_len (e.g. 200→400), the chunk
        # adapter sets request.prompt_token_ids = [0]*new_len on the live
        # Request object.  Without this patch that new value never reaches
        # the model runner's CachedRequestState, so the zero-fill guard
        # (len(req_state.prompt_token_ids) > num_tokens_no_spec) stays False
        # and the codec_embedding sees a -1 placeholder → OOB.
        if not hasattr(cached_reqs, "prompt_token_ids"):
            cached_reqs.prompt_token_ids = {}
        for req_id in cached_reqs.req_ids:
            request = requests.get(req_id) if req_id else None
            additional_info = getattr(request, "additional_information", None) if request else None
            cached_reqs.additional_information[req_id] = additional_info
            if request is not None:
                cached_reqs.prompt_token_ids[req_id] = request.prompt_token_ids

    def _process_chunk_queue(
        self,
        queue: Any,
        waiting_for_chunk_list: deque[Any],
        target_status: RequestStatus,
        finished_load_reqs: set[str],
    ) -> None:
        queue_snapshot = list(queue)
        for request in queue_snapshot:
            if request.status != RequestStatus.WAITING_FOR_CHUNK:
                if request.request_id in self.requests_with_ready_chunks:
                    # Requests that have loaded chunk from last round
                    # of schedule, but have not scheduled
                    continue
                if request.request_id in self.finished_requests:
                    continue
                # Requests that waiting for chunk
                self.load_async(request)
                request.status = RequestStatus.WAITING_FOR_CHUNK
            else:
                if request.request_id in finished_load_reqs:
                    request.status = target_status
                    finished_load_reqs.remove(request.request_id)
                    self.requests_with_ready_chunks.add(request.request_id)
                    continue
            queue.remove(request)
            waiting_for_chunk_list.append(request)

    def _clear_chunk_ready(self, scheduler_output: Any) -> None:
        if scheduler_output.scheduled_new_reqs:
            for req_data in scheduler_output.scheduled_new_reqs:
                if req_data.req_id in self.requests_with_ready_chunks:
                    self.requests_with_ready_chunks.remove(req_data.req_id)

        if scheduler_output.scheduled_cached_reqs:
            for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
                if req_id in self.requests_with_ready_chunks:
                    self.requests_with_ready_chunks.remove(req_id)
