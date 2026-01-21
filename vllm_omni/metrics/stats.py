from __future__ import annotations

import time
from dataclasses import dataclass
from pprint import pformat
from collections import defaultdict
from typing import Any, Callable, Optional, Union

from vllm.logger import init_logger
from vllm_omni.metrics.utils import _build_field_defs, _build_row, _format_table, _get_field_names

logger = init_logger(__name__)


@dataclass
class StageStats:
    total_token: int = 0
    total_gen_time_ms: float = 0.0

    @property
    def avg_tokens_per_s(self) -> float:
        return (self.total_token * 1000.0 / self.total_gen_time_ms) if self.total_gen_time_ms > 0 else 0.0

@dataclass
class StageRequestStats:
    batch_id: int
    batch_size: int
    num_tokens_in: int
    num_tokens_out: int
    stage_gen_time_ms: float
    rx_transfer_bytes: int
    rx_decode_time_ms: float
    rx_in_flight_time_ms: float
    stage_stats: StageStats
    stage_id: Optional[int] = None
    request_id: Optional[str] = None
    preprocess_time_ms: float = 0.0

    @property
    def rx_mbps(self) -> float:
        return (
            (float(self.rx_transfer_bytes) * 8.0) / (max(float(self.rx_decode_time_ms), 1e-6) * 1000.0)
            if self.rx_transfer_bytes > 0
            else 0.0
        )

    @property
    def tokens_per_s(self) -> float:
        return (self.num_tokens_out * 1000.0 / self.stage_gen_time_ms) if (self.stage_gen_time_ms > 0) else 0.0


@dataclass
class TransferEdgeStats:
    from_stage: int
    to_stage: int
    request_id: str
    size_bytes: int
    tx_time_ms: float
    used_shm: bool = False
    rx_decode_time_ms: float = 0.0
    in_flight_time_ms: float = 0.0

    @property
    def total_time_ms(self) -> float:
        return float(self.tx_time_ms) + float(self.rx_decode_time_ms) + float(self.in_flight_time_ms)


@dataclass
class RequestE2EStats:
    request_id: str
    e2e_total_ms: float
    e2e_total_tokens: int
    transfers_total_time_ms: float
    transfers_total_bytes: int

    @property
    def e2e_tpt(self) -> float:
        return (self.e2e_total_ms / self.e2e_total_tokens) if self.e2e_total_tokens > 0 else 0.0

# === Field Configuration ===
# Fields requiring unit conversion:  original_field_name -> (display_name, transform_fn)
FIELD_TRANSFORMS: dict[str, tuple[str, Callable[[Any], Any]]] = {
    "rx_transfer_bytes":  ("rx_transfer_kbytes", lambda v: v / 1024.0),
    "size_bytes": ("size_kbytes", lambda v: v / 1024.0),
    "transfers_total_bytes": ("transfers_total_kbytes", lambda v: v / 1024.0),
}

# Fields to exclude from table display for each event type
STAGE_EXCLUDE = {"stage_stats", "stage_id", "request_id"}
TRANSFER_EXCLUDE = {"from_stage", "to_stage", "request_id", "used_shm"}
E2E_EXCLUDE = {"request_id"}

# Overall summary fields (manually defined since it's not a dataclass)
OVERALL_FIELDS = [
    "e2e_requests",
    "e2e_wall_time_ms",
    "e2e_total_tokens",
    "e2e_avg_time_per_request_ms",
    "e2e_avg_tokens_per_s",
]
STAGE_FIELDS = _build_field_defs(StageRequestStats, STAGE_EXCLUDE, FIELD_TRANSFORMS)
TRANSFER_FIELDS = _build_field_defs(TransferEdgeStats, TRANSFER_EXCLUDE, FIELD_TRANSFORMS)
E2E_FIELDS = _build_field_defs(RequestE2EStats, E2E_EXCLUDE, FIELD_TRANSFORMS)

def _get_or_create_transfer_event(
    transfer_events: dict[tuple[int, int, str], TransferEdgeStats],
    from_stage: int,
    to_stage: int,
    request_id: str,
) -> TransferEdgeStats:
    key = (from_stage, to_stage, request_id)
    evt = transfer_events.get(key)
    if evt is None:
        evt = TransferEdgeStats(
            from_stage=from_stage,
            to_stage=to_stage,
            request_id=request_id,
            size_bytes=0,
            tx_time_ms=0.0,
            used_shm=False,
            rx_decode_time_ms=0.0,
            in_flight_time_ms=0.0,
        )
        transfer_events[key] = evt
    return evt


def record_transfer_tx(
    transfer_events: dict[tuple[int, int, str], TransferEdgeStats],
    from_stage: int,
    to_stage: int,
    request_id: Any,
    size_bytes: int,
    tx_time_ms: float,
    used_shm: bool,
) -> TransferEdgeStats | None:
    try:
        evt = _get_or_create_transfer_event(
            transfer_events,
            int(from_stage),
            int(to_stage),
            str(request_id),
        )
        # Accumulate tx metrics
        evt.size_bytes += int(size_bytes)
        evt.tx_time_ms += float(tx_time_ms)
        evt.used_shm = evt.used_shm or bool(used_shm)
        return evt
    except Exception:
        return None


def record_transfer_rx(
    transfer_events: dict[tuple[int, int, str], TransferEdgeStats],
    stats: StageRequestStats,
) -> TransferEdgeStats | None:
    try:
        if stats.stage_id is None or stats.stage_id <= 0:
            return None
        from_stage = int(stats.stage_id) - 1
        to_stage = int(stats.stage_id)
        rid_key = str(stats.request_id)
        evt = _get_or_create_transfer_event(transfer_events, from_stage, to_stage, rid_key)
        # Accumulate rx metrics
        if evt.size_bytes == 0:
            # size_bytes has been recorded in tx phase
            evt.size_bytes = int(stats.rx_transfer_bytes)
        evt.rx_decode_time_ms += float(stats.rx_decode_time_ms)
        evt.in_flight_time_ms += float(stats.rx_in_flight_time_ms)
        return evt
    except Exception:
        return None

def log_request_stats(stats: Union[StageRequestStats, TransferEdgeStats, RequestE2EStats], stats_type: str, **kwargs) -> None:
    if stats_type == "stage_stats":
        logger.info(
            pformat(
                {
                    "type": stats_type,
                    "stage_id": stats.stage_id,
                    "request_id": stats.request_id,
                    "batch_size": int(stats.batch_size),
                    "num_tokens_out": int(stats.num_tokens_out),
                    "stage_gen_time_ms": float(stats.stage_gen_time_ms),
                    "tokens_per_s": float(stats.tokens_per_s),
                    "rx_transfer_bytes": int(stats.rx_transfer_bytes),
                    "rx_decode_time_ms": float(stats.rx_decode_time_ms),
                    "rx_mbps": float(stats.rx_mbps),
                },
                sort_dicts=False,
            )
        )
    if stats_type == "transfer_tx_stats":
        logger.info(
            pformat(
                {
                    "type": stats_type,
                    "from_stage": stats.from_stage,
                    "to_stage": stats.to_stage,
                    "request_id": stats.request_id,
                    "size_bytes": int(stats.size_bytes),
                    "tx_time_ms": float(stats.tx_time_ms),
                    "tx_mbps": (float(stats.size_bytes) * 8.0) / (max(stats.tx_time_ms, 1e-6) * 1000.0),
                    "used_shm": bool(stats.used_shm),
                },
                sort_dicts=False,
            )
        )
    if stats_type == "transfer_rx_stats":
        logger.info(
            pformat(
                {
                    "type": stats_type,
                    "from_stage": stats.from_stage,
                    "to_stage": stats.to_stage,
                    "request_id": stats.request_id,
                    "rx_bytes": int(stats.rx_transfer_bytes),
                    "rx_decode_time_ms": float(stats.rx_decode_time_ms),
                    "in_flight_time_ms": float(stats.rx_in_flight_time_ms),
                    "rx_time_per_kb_ms": (
                        (float(stats.rx_decode_time_ms) / max(float(stats.rx_transfer_bytes) / 1024.0, 1e-6)) if stats.rx_transfer_bytes > 0 else 0.0
                    ),
                },
                sort_dicts=False,
            )
        )
    if stats_type == "transfer_total_stats":
        logger.info(
            pformat(
                {
                    "type": stats_type,
                    "from_stage": stats.stage_id - 1,
                    "to_stage": stats.stage_id,
                    "request_id": stats.request_id,
                    "size_bytes": int(kwargs.get("size_bytes", 0)),
                    "tx_time_ms": float(kwargs.get("tx_time_ms", 0.0)),
                    "in_flight_time_ms": stats.rx_in_flight_time_ms,
                    "rx_decode_time_ms": stats.rx_decode_time_ms,
                    "total_time_ms": float(kwargs.get("total_time_ms", 0.0)),
                    "total_time_per_kb_ms": (
                        float(kwargs.get("total_time_ms", 0.0)) / max(float(kwargs.get("size_bytes", 0)) / 1024.0, 1e-6) if kwargs.get("size_bytes", 0) > 0 else 0.0
                    ),
                },
                sort_dicts=False,
            )
        )
    if stats_type == "request_level_metrics":
        logger.info(
            pformat(
                {
                    "type": stats_type,
                    "request_id": stats.request_id,
                    "e2e_time_ms": stats.e2e_total_ms,
                    "e2e_tpt": (stats.e2e_total_ms / stats.e2e_total_tokens) if stats.e2e_total_tokens > 0 else 0.0,
                    "e2e_total_tokens": stats.e2e_total_tokens,
                    "transfers_total_time_ms": float(stats.transfers_total_time_ms),
                    "transfers_total_bytes": int(stats.transfers_total_bytes),
                },
                sort_dicts=False,
            )
        )
    if stats_type == "stage_running_avg":
        logger.info(
            pformat(
                {
                    "type": stats_type,
                    "stage_id": stats.stage_id,
                    "total_tokens": stats.stage_stats.total_token,
                    "total_gen_time_ms": stats.stage_stats.total_gen_time_ms,
                    "avg_tokens_per_s": stats.stage_stats.avg_tokens_per_s,
                },
                sort_dicts=False,
            )
        )


class OrchestratorAggregator:
    def __init__(
        self,
        num_stages: int,
        enable_stats: bool,
        wall_start_ts: float,
    ) -> None:
        self.num_stages = int(num_stages)
        self.enable_stats = bool(enable_stats)
        self.init_run_state(wall_start_ts)
        self.stage_events: dict[str, list[StageRequestStats]] = {}
        self.transfer_events: dict[tuple[int, int, str], TransferEdgeStats] = {} # Key: (from_stage, to_stage, request_id)
        self.e2e_events: list[RequestE2EStats] = []
    
    def init_run_state(self, wall_start_ts: float) -> None:
        # Per-run aggregates and timing state
        self.stage_total_tokens = [0 for _ in range(self.num_stages)]
        self.e2e_total_ms = 0.0
        self.e2e_total_tokens = 0
        self.e2e_count = 0
        self.e2e_done = set()
        self.wall_start_ts = float(wall_start_ts)
        self.last_finish_ts = float(wall_start_ts)
        self.stage_first_ts = [None for _ in range(self.num_stages)]
        self.stage_last_ts = [None for _ in range(self.num_stages)]
        self.accumulated_gen_time_ms: defaultdict[str, float] = defaultdict(float) # {request_id: accumulated_gen_time_ms}

    @staticmethod
    def _as_stage_request_stats(stage_id: int, req_id: str, metrics: StageRequestStats | dict[str, Any]) -> StageRequestStats:
        'Convert dict to StageRequestStats if needed.'
        if isinstance(metrics, StageRequestStats):
            stats = metrics
            stats.stage_id = stage_id
            stats.request_id = req_id
            return stats
        else:
            stage_stats = None
            if "stage_stats" in metrics:
                ss = metrics["stage_stats"]
                stage_stats = StageStats(
                    total_token=int(ss.get("total_token", 0)),
                    total_gen_time_ms=float(ss.get("total_gen_time_ms", 0.0)),
                )
            return StageRequestStats(
                stage_id=stage_id,
                request_id=req_id,
                num_tokens_in=int(metrics.get("num_tokens_in", 0)),
                num_tokens_out=int(metrics.get("num_tokens_out", 0)),
                batch_id=metrics.get("batch_id", -1),
                batch_size=metrics.get("batch_size"),
                stage_gen_time_ms=self.accumulated_gen_time_ms.pop(req_id, 0.0),
                rx_transfer_bytes=int(metrics.get("rx_transfer_bytes")),
                rx_decode_time_ms=metrics.get("rx_decode_time_ms"),
                rx_in_flight_time_ms=metrics.get("rx_in_flight_time_ms", 0.0),
                stage_stats=stage_stats,
            )

    def on_stage_metrics(self, stage_id: int, req_id: Any, metrics: StageRequestStats | dict[str, Any]) -> None:
        stats = self._as_stage_request_stats(stage_id, req_id, metrics)
        self.stage_total_tokens[stats.stage_id] += int(stats.num_tokens_out)
        if stats.stage_id == 0:
            self.stage_total_tokens[stats.stage_id] += int(stats.num_tokens_in)
        self.stage_events.setdefault(str(stats.request_id), []).append(stats)
        if self.enable_stats:
            log_request_stats(stats, "stage_stats")
            if stats.stage_stats is not None:
                log_request_stats(stats, "stage_running_avg")

        evt = record_transfer_rx(self.transfer_events, stats)
        if self.enable_stats and stats.stage_id is not None and stats.stage_id > 0:
            log_request_stats(stats, "transfer_rx_stats")
            if evt is not None and (evt.tx_time_ms > 0.0 or evt.size_bytes > 0):
                log_request_stats(
                    stats,
                    "transfer_total_stats",
                    size_bytes=int(evt.size_bytes),
                    tx_time_ms=float(evt.tx_time_ms),
                    total_time_ms=float(evt.total_time_ms),
                )

    def record_stage_preprocess_time(self, stage_id: int, req_id: Any, prep_time_ms: float) -> None:
        if req_id in self.stage_events:
            for stats in self.stage_events[req_id]:
                if stats.stage_id == stage_id:
                    stats.preprocess_time_ms = float(prep_time_ms)
                    break
        else:
            logger.warning(
                "Failed to record preprocess time for request %s at stage %s: no stage event found",
                req_id,
                stage_id,
            )

    def on_forward(
        self,
        from_stage: int,
        to_stage: int,
        req_id: Any,
        size_bytes: int,
        tx_ms: float,
        used_shm: bool,
    ) -> None:
        # Mark first input time for the destination stage if not set
        if self.stage_first_ts[to_stage] is None:
            self.stage_first_ts[to_stage] = time.time()
        evt = record_transfer_tx(
            self.transfer_events,
            from_stage=from_stage,
            to_stage=to_stage,
            request_id=req_id,
            size_bytes=size_bytes,
            tx_time_ms=tx_ms,
            used_shm=used_shm,
        )
        if self.enable_stats and evt is not None:
            log_request_stats(evt, "transfer_tx_stats")

    def on_finalize_request(
        self,
        stage_id: int,
        req_id: Any,
        req_start_ts: float,
    ) -> None:
        rid_key = str(req_id)
        _t0 = float(req_start_ts)
        _t1 = time.time()
        # Update last output time for this stage
        prev_last = self.stage_last_ts[stage_id]
        self.stage_last_ts[stage_id] = _t1 if prev_last is None else max(prev_last, _t1)
        self.last_finish_ts = max(self.last_finish_ts, _t1)
        e2e_ms = (_t1 - _t0) * 1000.0

        # Sum tokens from all stages for this request
        # Include input tokens from stage 0 + output tokens from all stages
        total_tokens = 0
        if rid_key in self.stage_events:
            for evt in self.stage_events[rid_key]:
                if evt.stage_id == 0:
                    total_tokens += int(evt.num_tokens_in)
                total_tokens += int(evt.num_tokens_out)

        self.e2e_total_ms += e2e_ms
        self.e2e_total_tokens += total_tokens
        self.e2e_count += 1
        self.e2e_done.add(rid_key)
        per_req_record = RequestE2EStats(
            request_id=rid_key,
            e2e_total_ms=e2e_ms,
            e2e_total_tokens=total_tokens,
            transfers_total_time_ms=float(
                sum(
                    evt.total_time_ms
                    for evt in self.transfer_events.values()
                    if evt.request_id == rid_key
                )
            ),
            transfers_total_bytes=int(
                sum(
                    evt.size_bytes
                    for evt in self.transfer_events.values()
                    if evt.request_id == rid_key
                )
            ),
        )
        self.e2e_events.append(per_req_record)
        if self.enable_stats:
            log_request_stats(per_req_record, "request_level_metrics")

    def build_and_log_summary(self, final_stage_id_to_prompt: Union[dict[str, int], int]) -> dict[str, Any]:
        wall_time_ms = max(0.0, (self.last_finish_ts - self.wall_start_ts) * 1000.0)
        e2e_avg_req = (wall_time_ms / self.e2e_count) if self.e2e_count > 0 else 0.0
        e2e_avg_tok = (self.e2e_total_tokens * 1000.0 / wall_time_ms) if wall_time_ms > 0 else 0.0

        if isinstance(final_stage_id_to_prompt, int):
            final_stage_id_map:  dict[str, int] = {"*": int(final_stage_id_to_prompt)}
        else:
            final_stage_id_map = final_stage_id_to_prompt

        stage_wall_time_ms = [
            ((self.stage_last_ts[i] - self.stage_first_ts[i]) * 1000.0)
            if (self.stage_first_ts[i] is not None and self.stage_last_ts[i] is not None)
            else 0.0
            for i in range(self.num_stages)
        ]

        overall_summary = {
            "e2e_requests": int(self.e2e_count),
            "e2e_wall_time_ms":  float(wall_time_ms),
            "e2e_total_tokens": int(self.e2e_total_tokens),
            "e2e_avg_time_per_request_ms":  float(e2e_avg_req),
            "e2e_avg_tokens_per_s": float(e2e_avg_tok),
            "stage_wall_time_ms": stage_wall_time_ms,
        }

        # Print overall summary
        logger.info(
            "\n%s",
            _format_table("Overall Summary", overall_summary, OVERALL_FIELDS),
        )

        all_request_ids = sorted(set(self.stage_events.keys()) | {e.request_id for e in self.e2e_events})

        result_stage_table = []
        result_trans_table = []
        result_e2e_table = []

        for rid in all_request_ids: 
            # === E2E table (single column) ===
            e2e_evt = next((e for e in self.e2e_events if e.request_id == rid), None)
            if e2e_evt:
                e2e_data = _build_row(e2e_evt, E2E_FIELDS)
                result_e2e_table.append({"request_id": rid, **e2e_data})

                logger.info(
                    "\n%s",
                    _format_table(
                        f"RequestE2EStats [request_id={rid}]",
                        e2e_data,
                        _get_field_names(E2E_FIELDS),
                    ),
                )

            # === Stage table (columns = stage_id) ===
            stage_evts = sorted(
                self.stage_events.get(rid, []),
                key=lambda e: e.stage_id if e.stage_id is not None else -1,
            )
            stage_rows = [
                {"stage_id": evt.stage_id, **_build_row(evt, STAGE_FIELDS)}
                for evt in stage_evts
            ]
            result_stage_table.append({"request_id": rid, "stages": stage_rows})

            if stage_rows:
                logger.info(
                    "\n%s",
                    _format_table(
                        f"StageRequestStats [request_id={rid}]",
                        stage_rows,
                        column_key="stage_id",
                        value_fields=_get_field_names(STAGE_FIELDS),
                    ),
                )

            # === Transfer table (columns = edge) ===
            transfer_evts = sorted(
                [e for e in self.transfer_events.values() if e.request_id == rid],
                key=lambda e: (e.from_stage, e.to_stage),
            )
            transfer_rows = [
                {"edge": f"{evt.from_stage}->{evt.to_stage}", **_build_row(evt, TRANSFER_FIELDS)}
                for evt in transfer_evts
            ]
            result_trans_table.append({"request_id": rid, "transfers": transfer_rows})

            if transfer_rows:
                logger.info(
                    "\n%s",
                    _format_table(
                        f"TransferEdgeStats [request_id={rid}]",
                        transfer_rows,
                        column_key="edge",
                        value_fields=_get_field_names(TRANSFER_FIELDS),
                    ),
                )

        return {
            "final_stage_id":  final_stage_id_map,
            "overall_summary": overall_summary,
            "stage_table": result_stage_table,
            "trans_table": result_trans_table,
            "e2e_table": result_e2e_table,
        }