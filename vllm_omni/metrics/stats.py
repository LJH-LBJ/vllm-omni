from __future__ import annotations

import time
from dataclasses import dataclass, fields
from pprint import pformat
from typing import Any, Optional, Union

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class StageStats:
    total_token: int = 0
    total_gen_time: float = 0.0

    @property
    def avg_tokens_per_s(self) -> float:
        return (self.total_token * 1000.0 / self.total_gen_time) if self.total_gen_time > 0 else 0.0


@dataclass
class StageRequestStats:
    stage_id: Optional[int] = None
    request_id: Optional[str] = None
    num_tokens_in: int
    num_tokens_out: int
    batch_id: int
    batch_size: int
    num_engine_outputs: Optional[int] = None
    stage_gen_time_ms: float
    rx_transfer_bytes: int
    rx_decode_time_ms: float
    rx_in_flight_time_ms: float
    stage_stats: StageStats

    @property
    def rx_mbps(self) -> float:
        return (
            (float(self.rx_transfer_bytes) * 8.0) / (max(float(self.rx_decode_time_ms), 1e-6) * 1000.0)
            if self.rx_transfer_bytes > 0
            else 0.0
        )

    @property
    def tokens_per_s(self) -> float:
        return (self.num_engine_outputs * 1000.0 / self.stage_gen_time_ms) if (self.stage_gen_time_ms > 0) else 0.0


@dataclass
class TransferEdgeStats:
    from_stage: int
    to_stage: int
    request_id: str
    size_bytes: int
    tx_time_ms: float
    rx_decode_time_ms: float
    rx_in_flight_time_ms: float
    used_shm: bool = False

    @property
    def total_time_ms(self) -> float:
        return self.tx_time_ms + self.rx_in_flight_time_ms + self.rx_decode_time_ms


@dataclass
class RequestE2EStats:
    request_id: str
    e2e_total_ms: float
    e2e_total_tokens: int
    transfers_total_time_ms: float
    transfers_total_bytes: int
    stages: dict[str, Any]

    @property
    def e2e_tpt(self) -> float:
        return (self.e2e_total_ms / self.e2e_total_tokens) if self.e2e_total_tokens > 0 else 0.0


@dataclass
class OrchestratorStatsSummary:
    e2e_requests: int
    e2e_total_time_ms: float
    e2e_sum_time_ms: float
    e2e_total_tokens: int
    e2e_avg_time_per_request_ms: float
    e2e_avg_tokens_per_s: float
    wall_time_ms: float
    final_stage_id: dict[str, int]
    stages: list[dict[str, Any]]
    transfers: list[dict[str, Any]]


def record_stage_metrics(
    per_request: dict[str, dict[str, Any]],
    stage_req_counts: list[int],
    stage_total_time_ms: list[float],
    stage_total_tokens: list[int],
    stats: StageRequestStats,
) -> None:
    try:
        stage_req_counts[stats.stage_id] += 1
        stage_total_tokens[stats.stage_id] += int(stats.num_tokens_out)
        rid_key = str(stats.request_id)
        pr = per_request.setdefault(rid_key, {"stages": {}, "transfers_ms": 0.0, "transfers_bytes": 0})
        pr_stages = pr["stages"]  # type: ignore[index]
        stage_data: dict[str, Any] = {
            "stage_gen_time_ms": float(stats.stage_gen_time_ms),
            "num_tokens_out": int(stats.num_tokens_out),
        }
        # Only record num_tokens_in for stage 0 (initial prompt)
        if stats.stage_id == 0:
            stage_data["num_tokens_in"] = int(stats.num_tokens_in)
            stage_total_tokens[stats.stage_id] += int(stats.num_tokens_in)
        pr_stages[stats.stage_id] = stage_data
    except Exception:
        pass


def aggregate_rx_and_maybe_total(
    transfer_edge_req: dict[tuple[int, int, str], dict[str, float]],
    transfer_agg: dict[tuple[int, int], dict[str, float]],
    per_request: dict[str, dict[str, Any]],
    stats: StageRequestStats,
) -> tuple[int, float, float] | None:
    try:
        # Update RX aggregates for (stage_id-1 -> stage_id)
        if stats.stage_id > 0:
            key = (stats.stage_id - 1, stats.stage_id)
            agg = transfer_agg.get(key)
            if agg is None:
                agg = {
                    "sum_bytes": 0.0,
                    "sum_ms": 0.0,
                    "count": 0.0,
                    "sum_rx_bytes": 0.0,
                    "sum_rx_ms": 0.0,
                    "rx_count": 0.0,
                    "sum_total_ms": 0.0,
                    "total_count": 0.0,
                }
                transfer_agg[key] = agg
            agg["sum_rx_bytes"] += float(stats.rx_transfer_bytes)
            agg["sum_rx_ms"] += float(stats.rx_decode_time_ms)
            agg["rx_count"] += 1.0

            # Try combine with sender-side timing if present
            rid_key = str(stats.request_id)
            s = transfer_edge_req.get((stats.stage_id - 1, stats.stage_id, rid_key))
            if s is None:
                return None
            tx_ms = float(s.get("tx_ms", 0.0))
            size_b = float(s.get("size_bytes", stats.rx_transfer_bytes))
            total_ms = tx_ms + float(stats.rx_in_flight_time_ms) + float(stats.rx_decode_time_ms)
            agg["sum_total_ms"] += total_ms
            agg["total_count"] += 1.0
            # accumulate per-request transfer totals
            try:
                pr = per_request.setdefault(rid_key, {"stages": {}, "transfers_ms": 0.0, "transfers_bytes": 0})
                pr["transfers_ms"] = float(pr.get("transfers_ms", 0.0)) + total_ms  # type: ignore[index]
                pr["transfers_bytes"] = int(pr.get("transfers_bytes", 0)) + int(stats.rx_transfer_bytes)  # type: ignore[index]
            except Exception:
                pass
            return int(size_b), float(tx_ms), float(total_ms)
        return None
    except Exception:
        return None


def record_sender_transfer_agg(
    transfer_agg: dict[tuple[int, int], dict[str, float]],
    transfer_edge_req: dict[tuple[int, int, str], dict[str, float]],
    stats: TransferEdgeStats,
) -> None:
    try:
        key = (stats.from_stage, stats.to_stage)
        agg = transfer_agg.get(key)
        if agg is None:
            agg = {
                "sum_bytes": 0.0,
                "sum_ms": 0.0,
                "count": 0.0,
                "sum_rx_bytes": 0.0,
                "sum_rx_ms": 0.0,
                "rx_count": 0.0,
                "sum_total_ms": 0.0,
                "total_count": 0.0,
            }
            transfer_agg[key] = agg
        agg["sum_bytes"] += float(stats.size_bytes)
        agg["sum_ms"] += float(stats.tx_time_ms)
        agg["count"] += 1.0
        # Store sender-side timing for per-request combination
        rid_key = str(stats.request_id)
        transfer_edge_req[(stats.from_stage, stats.to_stage, rid_key)] = {
            "tx_ms": float(stats.tx_time_ms),
            "size_bytes": float(stats.size_bytes),
        }
    except Exception:
        pass


def count_tokens_from_outputs(engine_outputs: list[Any]) -> int:
    total = 0
    for _ro in engine_outputs:
        try:
            outs = getattr(_ro, "outputs", None)
            if outs and len(outs) > 0:
                tokens = getattr(outs[0], "token_ids", None)
                if tokens is not None:
                    total += len(tokens)
        except Exception:
            pass
    return total


def build_stage_summary(
    stage_req_counts: list[int],
    stage_total_tokens: list[int],
    stage_total_time_ms: list[float],
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for sid in range(len(stage_req_counts)):
        reqs = stage_req_counts[sid]
        tokens = stage_total_tokens[sid]
        total_ms = float(stage_total_time_ms[sid])
        avg_req = (total_ms / reqs) if reqs > 0 else 0.0
        avg_tok = (tokens * 1000.0 / total_ms) if total_ms > 0 else 0.0
        summary.append(
            {
                "stage_id": sid,
                "requests": int(reqs),
                "tokens": int(tokens),
                "total_time_ms": total_ms,
                "avg_time_per_request_ms": avg_req,
                "avg_tokens_per_s": avg_tok,
            }
        )
    return summary


def build_transfer_summary(
    transfer_agg: dict[tuple[int, int], dict[str, float]],
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for (src, dst), agg in transfer_agg.items():
        sum_bytes = float(agg.get("sum_bytes", 0.0))
        sum_ms = float(agg.get("sum_ms", 0.0))
        samples = int(agg.get("count", 0.0))
        tx_mbps = (sum_bytes * 8.0) / (max(sum_ms, 1e-6) * 1000.0) if sum_bytes > 0 else 0.0
        sum_rx_bytes = float(agg.get("sum_rx_bytes", 0.0))
        sum_rx_ms = float(agg.get("sum_rx_ms", 0.0))
        samples_rx = int(agg.get("rx_count", 0.0))
        rx_mbps = (sum_rx_bytes * 8.0) / (max(sum_rx_ms, 1e-6) * 1000.0) if sum_rx_bytes > 0 else 0.0
        sum_total_ms = float(agg.get("sum_total_ms", 0.0))
        samples_total = int(agg.get("total_count", 0.0))
        total_mbps = (sum_bytes * 8.0) / (max(sum_total_ms, 1e-6) * 1000.0) if sum_bytes > 0 else 0.0
        summary.append(
            {
                "from_stage": src,
                "to_stage": dst,
                "samples": samples,
                "total_bytes": int(sum_bytes),
                "total_time_ms": sum_ms,
                "tx_mbps": tx_mbps,
                "rx_samples": samples_rx,
                "rx_total_bytes": int(sum_rx_bytes),
                "rx_total_time_ms": sum_rx_ms,
                "rx_mbps": rx_mbps,
                "total_samples": samples_total,
                "total_transfer_time_ms": sum_total_ms,
                "total_mbps": total_mbps,
            }
        )
    return summary


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
    if stats_type == "transfer_stats":
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
                    "size_bytes": int(*kwargs.get("size_bytes", 0)),
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
                    "stages": stats.stages,
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
                    "total_gen_time_ms": stats.stage_stats.total_gen_time,
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
        self.stage_events = []
        self.transfer_events = []
        self.e2e_events = []
    
    def init_run_state(self, wall_start_ts: float) -> None:
        # Per-run aggregates and timing state
        self.stage_total_time_ms = [0.0 for _ in range(self.num_stages)]
        self.stage_total_tokens = [0 for _ in range(self.num_stages)]
        self.stage_req_counts = [0 for _ in range(self.num_stages)]
        self.transfer_agg = {}
        self.transfer_edge_req = {}
        self.e2e_total_ms = 0.0
        self.e2e_total_tokens = 0
        self.e2e_count = 0
        self.e2e_done = set()
        self.per_request = {}
        self.sum_per_request_transfer_ms = 0.0
        self.wall_start_ts = float(wall_start_ts)
        self.last_finish_ts = float(wall_start_ts)
        self.stage_seen_batches = {sid: set() for sid in range(self.num_stages)}
        self.stage_first_ts = [None for _ in range(self.num_stages)]
        self.stage_last_ts = [None for _ in range(self.num_stages)]

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
                    total_gen_time=float(ss.get("total_gen_time", 0.0)),
                )
            return StageRequestStats(
                stage_id=stage_id,
                request_id=req_id,
                num_tokens_in=int(metrics.get("num_tokens_in", 0)),
                num_tokens_out=int(metrics.get("num_tokens_out", 0)),
                batch_id=int(metrics.get("batch_id", -1)),
                batch_size=int(metrics.get("batch_size", 0)),
                num_engine_outputs=int(metrics.get("num_engine_outputs", 0)),
                stage_gen_time_ms=float(metrics.get("stage_gen_time_ms", 0.0)),
                tokens_per_s=float(metrics.get("tokens_per_s", 0.0)),
                rx_transfer_bytes=int(metrics.get("rx_transfer_bytes", 0)),
                rx_decode_time_ms=float(metrics.get("rx_decode_time_ms", 0.0)),
                rx_in_flight_time_ms=float(metrics.get("rx_in_flight_time_ms", 0.0)),
                stage_stats=stage_stats,
            )

    @staticmethod        
    def _as_transfer_edge_stats(from_stage: int, to_stage: int, req_id: str, metrics: TransferEdgeStats | dict[str, Any]) -> TransferEdgeStats:
        'Convert dict to TransferEdgeStats if needed.'
        if isinstance(metrics, TransferEdgeStats):
            stats = metrics
            stats.from_stage = from_stage
            stats.to_stage = to_stage
            stats.request_id = req_id
            return stats
        else:
            return TransferEdgeStats(
                from_stage=from_stage,
                to_stage=to_stage,
                request_id=req_id,
                size_bytes=int(metrics.get("size_bytes", 0)),
                tx_time_ms=float(metrics.get("tx_time_ms", 0.0)),
                rx_decode_time_ms=float(metrics.get("rx_decode_time_ms", 0.0)),
                rx_in_flight_time_ms=float(metrics.get("rx_in_flight_time_ms", 0.0)),
                used_shm=bool(metrics.get("used_shm", False)),
            )


    def on_stage_metrics(self, stage_id: int, req_id: Any, metrics: StageRequestStats | dict[str, Any]) -> None:
        stats = self._as_stage_request_stats(stage_id, req_id, metrics)
        record_stage_metrics(
            self.per_request,
            self.stage_req_counts,
            self.stage_total_time_ms,
            self.stage_total_tokens,
            stats,
        )
        self.stage_events.append(stats)
        if self.enable_stats:
            log_request_stats(stats, "stage_stats")
            if stats.stage_stats is not None:
                log_request_stats(stats, "stage_running_avg")
        try:
            batch_id_raw = metrics.get("batch_id", None)
            if batch_id_raw is not None:
                batch_id = int(batch_id_raw)
                if batch_id not in self.stage_seen_batches[stats.stage_id]:
                    self.stage_total_time_ms[stats.stage_id] += float(metrics.get("stage_gen_time_ms", 0.0))
                    self.stage_seen_batches[stats.stage_id].add(batch_id)
        except Exception:
            pass
        combined = aggregate_rx_and_maybe_total(
            self.transfer_edge_req,
            self.transfer_agg,
            self.per_request,
            stats,
        )
        if self.enable_stats and stats.stage_id > 0:
            log_request_stats(stats, "transfer_rx_stats")
            if combined is not None:
                size_b_c, tx_ms_c, total_ms_c = combined
                log_request_stats(
                    stats, "transfer_total_stats", size_bytes=int(size_b_c),
                    tx_time_ms=float(tx_ms_c), total_time_ms=float(total_ms_c)
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
        metrics = TransferEdgeStats(
            from_stage=from_stage,
            to_stage=to_stage,
            request_id=str(req_id),
            size_bytes=int(size_bytes),
            tx_time_ms=float(tx_ms),
            used_shm=bool(used_shm),
        ) 
        stats = self._as_transfer_edge_stats(from_stage, to_stage, req_id, metrics)
        if self.stage_first_ts[to_stage] is None:
            self.stage_first_ts[to_stage] = time.time()
        if self.enable_stats:
            log_request_stats(stats, "Transfer_edge_stats")
        record_sender_transfer_agg(
            self.transfer_agg,
            self.transfer_edge_req,
            stats,
        )

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
        pr = self.per_request.setdefault(rid_key, {"stages": {}, "transfers_ms": 0.0, "transfers_bytes": 0})
        total_tokens = 0
        stages_info = pr.get("stages", {})
        for sid, stage_data in stages_info.items():
            # Add input tokens only from stage 0 (initial prompt)
            if sid == 0:
                total_tokens += int(stage_data.get("num_tokens_in", 0))
            total_tokens += int(stage_data.get("num_tokens_out", 0))

        self.e2e_total_ms += e2e_ms
        self.e2e_total_tokens += total_tokens
        self.e2e_count += 1
        self.e2e_done.add(rid_key)
        per_req_record = RequestE2EStats(
            request_id=rid_key,
            e2e_total_ms=e2e_ms,
            e2e_total_tokens=total_tokens,
            transfers_total_time_ms=float(pr.get("transfers_ms", 0.0)),
            transfers_total_bytes=int(pr.get("transfers_bytes", 0)),
            stages=stages_info,
        )
        self.sum_per_request_transfer_ms += float(pr.get("transfers_ms", 0.0))
        self.e2e_events.append(per_req_record)
        if self.enable_stats:
            log_request_stats(per_req_record, "request_level_metrics")
    
    def merge(self, events: list[Any]) -> dict[str, Any]:
        """Compute grouped averages for StageRequestStats/TransferEdgeStats/RequestE2EStats.

        - StageRequestStats are grouped by stage_id
        - TransferEdgeStats are grouped by (from_stage, to_stage)
        - RequestE2EStats are aggregated as a single group
        """

        def _accumulate_numeric(sums: dict[str, float], evt: Any) -> None:
            for field in fields(type(evt)):
                value = getattr(evt, field.name, None)
                if value is None or isinstance(value, (bool, str)):
                    continue
                if isinstance(value, (int, float)):
                    sums[field.name] = sums.get(field.name, 0.0) + float(value)
                    continue
                if isinstance(value, dict):
                    for k, v in value.items():
                        if isinstance(v, (int, float)):
                            key_name = f"{field.name}_{k}"
                            sums[key_name] = sums.get(key_name, 0.0) + float(v)
                    continue
                if isinstance(value, StageStats):
                    for s_field in fields(StageStats):
                        s_value = getattr(value, s_field.name, None)
                        if isinstance(s_value, (int, float)):
                            key_name = f"{field.name}_{s_field.name}"
                            sums[key_name] = sums.get(key_name, 0.0) + float(s_value)

        stage_sums: dict[int, dict[str, float]] = {}
        stage_counts: dict[int, int] = {}
        transfer_sums: dict[tuple[int, int], dict[str, float]] = {}
        transfer_counts: dict[tuple[int, int], int] = {}
        e2e_sums: dict[str, float] = {}
        e2e_count = 0

        for event in events:
            if isinstance(event, StageRequestStats):
                if event.stage_id is None:
                    continue
                sid = int(event.stage_id)
                sums = stage_sums.setdefault(sid, {})
                _accumulate_numeric(sums, event)
                stage_counts[sid] = stage_counts.get(sid, 0) + 1
            elif isinstance(event, TransferEdgeStats):
                key = (int(event.from_stage), int(event.to_stage))
                sums = transfer_sums.setdefault(key, {})
                _accumulate_numeric(sums, event)
                transfer_counts[key] = transfer_counts.get(key, 0) + 1
            elif isinstance(event, RequestE2EStats):
                _accumulate_numeric(e2e_sums, event)
                e2e_count += 1

        def _avg_map(sums: dict[str, float], count: int) -> dict[str, float]:
            return {name: (sums[name] / count) for name in sums} if count > 0 else {}

        stage_avgs = {str(sid): _avg_map(sums, stage_counts.get(sid, 0)) for sid, sums in stage_sums.items()}
        transfer_avgs = {
            f"{src}->{dst}": _avg_map(sums, transfer_counts.get((src, dst), 0))
            for (src, dst), sums in transfer_sums.items()
        }

        return {
            "averages": {
                "StageRequestStats": {"by_stage": stage_avgs},
                "TransferEdgeStats": {"by_edge": transfer_avgs},
                "RequestE2EStats": _avg_map(e2e_sums, e2e_count),
            },
            "counts": {
                "e2e": int(e2e_count),
                "stage": {str(sid): int(cnt) for sid, cnt in stage_counts.items()},
                "transfer": {f"{src}->{dst}": int(cnt) for (src, dst), cnt in transfer_counts.items()},
            },
        }

    def reset_events(self) -> None:
        self.stage_events = []
        self.transfer_events = []
        self.e2e_events = []

    @staticmethod
    def default_for_type(data_type):
        try:
            return data_type()
        except Exception:
            return None

    def build_and_log_summary(self, final_stage_id_to_prompt: dict[str, int]) -> OrchestratorStatsSummary:
        # Compute stage summary using wall time between first input and last output per stage
        stage_summary: list[dict[str, Any]] = []
        for sid in range(self.num_stages):
            first_ts = self.stage_first_ts[sid]
            last_ts = self.stage_last_ts[sid]
            total_ms = (
                (max(0.0, (last_ts - first_ts)) * 1000.0) if (first_ts is not None and last_ts is not None) else 0.0
            )
            reqs = self.stage_req_counts[sid]
            tokens = self.stage_total_tokens[sid]
            avg_req = (total_ms / reqs) if reqs > 0 else 0.0
            avg_tok = (tokens * 1000.0 / total_ms) if total_ms > 0 else 0.0
            stage_summary.append(
                {
                    "stage_id": sid,
                    "requests": int(reqs),
                    "tokens": int(tokens),
                    "total_time_ms": float(total_ms),
                    "avg_time_per_request_ms": float(avg_req),
                    "avg_tokens_per_s": float(avg_tok),
                }
            )
        transfer_summary = build_transfer_summary(self.transfer_agg)
        e2e_avg_req = (self.e2e_total_ms / self.e2e_count) if self.e2e_count > 0 else 0.0
        e2e_avg_tok = (self.e2e_total_tokens * 1000.0 / self.e2e_total_ms) if self.e2e_total_ms > 0 else 0.0
        wall_time_ms = max(0.0, (self.last_finish_ts - self.wall_start_ts) * 1000.0)
        return OrchestratorStatsSummary(
            e2e_requests=int(self.e2e_count),
            e2e_total_time_ms=float(wall_time_ms),
            e2e_sum_time_ms=float(self.e2e_total_ms),
            e2e_total_tokens=int(self.e2e_total_tokens),
            e2e_avg_time_per_request_ms=e2e_avg_req,
            e2e_avg_tokens_per_s=e2e_avg_tok,
            wall_time_ms=wall_time_ms,
            final_stage_id=final_stage_id_to_prompt,
            stages=stage_summary,
            transfers=transfer_summary,
        )
