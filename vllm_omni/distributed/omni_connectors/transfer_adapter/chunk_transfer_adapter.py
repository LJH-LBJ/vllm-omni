# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from collections import defaultdict, deque
from collections.abc import Callable
from typing import Any

import torch
from vllm.v1.request import Request, RequestStatus

from vllm_omni.data_entry_keys import OmniPayloadStruct, unflatten_payload

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
        self.scheduler_max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.connector = self.create_connector(model_config)
        super().__init__(model_config)
        self.model_mode = getattr(model_config, "worker_type", None) or "ar"
        # State specific to Chunk management
        self.custom_process_next_stage_input_func: Callable[..., OmniPayloadStruct | None] | None = None
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
        self.requests_origin_status = {}
        self.requests_num_chunks_sent: dict[str, int] = defaultdict(int)

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

        # If the request is preempted, skip the already saved chunks.
        if request.num_computed_tokens < self.requests_num_chunks_sent.get(request.external_req_id, 0):
            logger.warning(
                f"Enqueue save_async for request {request.external_req_id}, "
                f"request.num_computed_tokens={request.num_computed_tokens}, "
                f"previous_chunks_sent={self.requests_num_chunks_sent.get(request.external_req_id, 0)}"
            )
            return

        self.requests_num_chunks_sent[request.external_req_id] = request.num_computed_tokens
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

            meta = payload_data.get("meta", {})
            if self.model_mode == "ar":
                merged_payload = self._update_request_payload(external_req_id, payload_data)
                request.additional_information = merged_payload
                finished_flag = meta.get("finished", False)
                is_chunk_finished = (
                    bool(finished_flag.item()) if isinstance(finished_flag, torch.Tensor) else bool(finished_flag)
                )
                if is_chunk_finished:
                    self.finished_requests.add(req_id)
                embed_data = payload_data.get("embed", {})
                has_prefill_embeds = isinstance(embed_data.get("prefill"), torch.Tensor)
                # A chunk is considered a "prefill boundary" if it has the "finished" flag or
                # contains decode/cached_decode embeds.
                prefill_boundary = (
                    is_chunk_finished
                    or isinstance(embed_data.get("decode"), torch.Tensor)
                    or isinstance(embed_data.get("cached_decode"), torch.Tensor)
                )
                if has_prefill_embeds and not prefill_boundary:
                    if self._gate_chunked_prefill_chunk(
                        request, payload_data, external_req_id, stage_id, connector_get_key
                    ):
                        return True

                # When entering decode phase for the first time, proactively replace
                # request_payload with a copy that excludes the large prefill tensors.
                # request.additional_information still references the full merged_payload
                # so the CURRENT model step (last talker-prefill) can still read
                # embed.prefill.  Starting from the next _update_request_payload call,
                # the cleaned payload is used as the base, so subsequent decode steps
                # serialise only small per-token data.
                if prefill_boundary and not is_chunk_finished:
                    self._evict_prefill_tensors(external_req_id, stage_id)
            else:
                if meta.get("finished"):
                    self.finished_requests.add(req_id)

                new_ids = payload_data.get("codes", {}).get("audio")
                if isinstance(new_ids, torch.Tensor):
                    new_ids = new_ids.tolist()
                elif new_ids is None:
                    new_ids = []
                request.prompt_token_ids = new_ids
                prev_info = getattr(request, "additional_information", None)
                info = dict(prev_info) if isinstance(prev_info, dict) else {}
                for key, value in payload_data.items():
                    if key == "codes":
                        continue
                    if isinstance(value, dict):
                        existing_sub = info.get(key)
                        merged_sub = dict(existing_sub) if isinstance(existing_sub, dict) else {}
                        for sk, sv in value.items():
                            if key == "meta" and sk == "finished":
                                continue
                            merged_sub[sk] = sv
                        info[key] = merged_sub
                        continue
                    info[key] = value
                request.additional_information = info
                request.num_computed_tokens = 0

                # Empty chunk with more data expected: keep polling.
                if not new_ids and not meta.get("finished"):
                    return True

            # Mark as finished for consumption
            self._finished_load_reqs.add(req_id)
            logger.debug(f"[Stage-{stage_id}] Received one chunk for key {connector_get_key}")
            return True

        return False

    def _gate_chunked_prefill_chunk(
        self,
        request: Request,
        payload_data: dict,
        external_req_id: str,
        stage_id: int,
        connector_get_key: str,
    ) -> bool:
        """Chunked-prefill readiness gate for AR-mode Thinker→Talker transfer.

        Returns True if the chunk must be buffered (not enough tokens have
        accumulated for the next scheduler slice yet), False if it can be
        released to the scheduler.
        """
        # Hold the final prefill chunk until the tok0 decode embed arrives in
        # the next step, so that the talker receives a complete bootstrap sequence.
        if payload_data.get("ids", {}).get("output"):
            self._pending_load_reqs.append(request)
            with self._recv_cond:
                self._recv_cond.notify()
            logger.debug(
                "[Stage-%s] Buffering final prefill chunk for key %s until tok0 embed arrives",
                stage_id,
                connector_get_key,
            )
            return True

        accumulated = self.request_payload.get(external_req_id, payload_data)
        cumulative_embeds = accumulated.get("embed", {}).get("prefill")
        available_tokens = cumulative_embeds.shape[0] if isinstance(cumulative_embeds, torch.Tensor) else 0
        remaining_prompt_tokens = max(request.num_prompt_tokens - request.num_computed_tokens, 0)
        next_scheduler_slice = min(
            self.scheduler_max_num_batched_tokens,
            remaining_prompt_tokens,
        )
        ready_tokens = available_tokens - request.num_computed_tokens
        if ready_tokens >= next_scheduler_slice:
            logger.debug(
                "[Stage-%s] Releasing prefill chunk for key %s: available=%d computed=%d next=%d",
                stage_id,
                connector_get_key,
                available_tokens,
                request.num_computed_tokens,
                next_scheduler_slice,
            )
            return False
        else:
            # wait for more prefill tokens
            self._pending_load_reqs.append(request)
            with self._recv_cond:
                self._recv_cond.notify()
            logger.debug(
                "[Stage-%s] Buffering prefill chunk for key %s: available=%d computed=%d next=%d",
                stage_id,
                connector_get_key,
                available_tokens,
                request.num_computed_tokens,
                next_scheduler_slice,
            )
            return True

    def _evict_prefill_tensors(self, external_req_id: str, stage_id: int) -> None:
        """At the Talker prefill→decode boundary, strip large prefill tensors
        (embed.prefill/tts_*, hidden_states.output, ids.all/prompt) from the
        cached payload so subsequent decode steps don't serialize them.
        """
        acc = self.request_payload.get(external_req_id)
        if acc is None or not isinstance(acc.get("embed", {}).get("prefill"), torch.Tensor):
            return
        cleaned_embed = {
            k: v for k, v in acc.get("embed", {}).items() if k not in ("prefill", "tts_bos", "tts_eos", "tts_pad")
        }
        cleaned_hs = {k: v for k, v in acc.get("hidden_states", {}).items() if k != "output"}
        cleaned_ids = {k: v for k, v in acc.get("ids", {}).items() if k not in ("all", "prompt")}
        cleaned = {k: v for k, v in acc.items() if k not in ("embed", "hidden_states", "ids")}
        cleaned["embed"] = cleaned_embed
        cleaned["hidden_states"] = cleaned_hs
        cleaned["ids"] = cleaned_ids
        self.request_payload[external_req_id] = cleaned
        logger.debug(
            "[Stage-%s] Cleared prefill tensors from request_payload for req %s",
            stage_id,
            external_req_id,
        )

    def _update_request_payload(self, req_id: str, payload_data: dict[str, Any]) -> dict[str, Any]:
        """Update the stored payload for *req_id* with the latest chunk."""
        if req_id not in self.request_payload:
            self.request_payload[req_id] = payload_data
            return payload_data
        origin_payload = self.request_payload[req_id]
        raw_ok = payload_data.get("meta", {}).pop("override_keys", [])
        override_keys = {tuple(k) if isinstance(k, list) else k for k in raw_ok}
        merged_payload = dict(origin_payload)
        # Merge non-dict top-level keys from new payload
        merged_payload.update({k: v for k, v in payload_data.items() if not isinstance(v, dict)})
        # Merge nested dicts with concat for tensors/lists, respecting override_keys
        for type_key, new_val in payload_data.items():
            if not isinstance(new_val, dict):
                continue
            origin_sub = origin_payload.get(type_key, {})
            if not isinstance(origin_sub, dict):
                merged_payload[type_key] = new_val
                continue
            merged_sub = dict(origin_sub)
            for qual, value in new_val.items():
                if type_key == "meta" and qual == "finished":
                    merged_sub[qual] = value
                elif (type_key, qual) in override_keys:
                    merged_sub[qual] = value
                elif isinstance(value, torch.Tensor) and qual in origin_sub:
                    merged_sub[qual] = torch.cat([origin_sub[qual], value], dim=0)
                elif isinstance(value, list) and qual in origin_sub:
                    merged_sub[qual] = origin_sub[qual] + value
                else:
                    merged_sub[qual] = value
            merged_payload[type_key] = merged_sub

        self.request_payload[req_id] = merged_payload
        return merged_payload

    def _send_single_request(self, task: dict):
        raw_po = task["pooling_output"]
        pooling_output = unflatten_payload(raw_po) if isinstance(raw_po, dict) else raw_po
        request = task["request"]
        is_finished = task["is_finished"]
        stage_id = self.connector.stage_id
        next_stage_id = stage_id + 1
        external_req_id = request.external_req_id
        chunk_id = self.put_req_chunk[external_req_id]
        connector_put_key = f"{external_req_id}_{stage_id}_{chunk_id}"
        # Process payload in save_loop thread
        payload_data: OmniPayloadStruct | None = None
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

        if payload_data is None:
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
            # payload_data is a dict (returned by thinker2talker_async_chunk).
            # On the receiver side, the msgpack wire round-trip also produces dicts.
            meta = payload_data.get("meta", {}) if isinstance(payload_data, dict) else {}
            finished_flag = meta.get(
                "finished", payload_data.get("finished") if isinstance(payload_data, dict) else None
            )
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
            self.requests_num_chunks_sent.pop(external_req_id, None)
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
        self.requests_origin_status.pop(request_id, None)

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
        self.requests_num_chunks_sent.pop(external_req_id, None)
        prefill_part_state = getattr(self, "_prefill_part_state", None)
        if isinstance(prefill_part_state, dict):
            prefill_part_state.pop(external_req_id, None)

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
    ) -> None:
        """
        Process pending chunks for waiting and running queues.
        """
        if self.connector.stage_id == 0:
            return
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
        stage_id = self.connector.stage_id

        if stage_id == 0:
            return

        if requests is not None:
            self.attach_cached_additional_information(scheduler_output, requests)
        self._clear_chunk_ready(scheduler_output)

    @staticmethod
    def attach_cached_additional_information(scheduler_output: Any, requests: dict[str, Request]) -> None:
        cached_reqs = getattr(scheduler_output, "scheduled_cached_reqs", None)
        if not cached_reqs:
            return
        if not hasattr(cached_reqs, "additional_information"):
            cached_reqs.additional_information = {}
        for req_id in cached_reqs.req_ids:
            request = requests.get(req_id) if req_id else None
            additional_info = getattr(request, "additional_information", None) if request else None
            cached_reqs.additional_information[req_id] = additional_info
            if request and additional_info:
                request.additional_information = None

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
            self.requests_origin_status[request.request_id] = target_status
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

    def finish_requests(
        self, request_ids: Any, finished_status: RequestStatus, requests: dict[str, Request] | None = None
    ) -> list[tuple[str, int]]:
        assert RequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids,)
        elif request_ids is not None:
            request_ids = set(request_ids)
        else:
            request_ids = requests.keys()

        # First pass: collect requests to remove from queues
        for req_id in request_ids:
            request = requests.get(req_id) if requests else None
            if request is None or request.is_finished():
                # Invalid request ID.
                continue
            if req_id in self.requests_origin_status:
                request.status = self.requests_origin_status.pop(req_id)
