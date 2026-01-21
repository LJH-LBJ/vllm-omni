from __future__ import annotations

from vllm_omni.metrics.loggers import OmniStatLoggerBase, OmniStatLoggerManager
from vllm_omni.metrics.stats import OrchestratorAggregator


def test_orchestrator_aggregator_builds_summary() -> None:
    agg = OrchestratorAggregator(num_stages=2, enable_debug_events=False, wall_start_ts=0.0)
    agg.set_final_stage_map({"r1": 1})
    agg.stage_first_ts[0] = 0.0
    agg.stage_last_ts[0] = 0.03
    agg.stage_first_ts[1] = 0.05
    agg.stage_last_ts[1] = 0.07

    agg.on_forward(0, 1, "r1", size_bytes=1024, tx_ms=5.0, used_shm=False)
    agg.on_stage_metrics(
        0,
        "r1",
        {
            "num_tokens_in": 3,
            "num_tokens_out": 3,
            "stage_gen_time_ms": 30.0,
            "batch_id": 1,
            "batch_size": 1,
            "rx_transfer_bytes": 0,
            "rx_decode_time_ms": 0.0,
        },
    )
    agg.on_stage_metrics(
        1,
        "r1",
        {
            "num_tokens_out": 4,
            "stage_gen_time_ms": 20.0,
            "batch_id": 1,
            "batch_size": 1,
            "rx_transfer_bytes": 1024,
            "rx_decode_time_ms": 5.0,
            "rx_in_flight_time_ms": 2.0,
        },
    )
    agg.on_finalize_request(1, "r1", req_start_ts=0.0)

    summary = agg.build_run_summary()
    data = summary.to_dict()
    assert data["e2e_requests"] == 1
    assert len(data["stages"]) == 2
    assert data["stages"][0]["requests"] == 1
    assert data["transfers"][0]["samples"] == 1
    assert data["transfers"][0]["total_mbps"] >= 0.0


class _DummyLogger(OmniStatLoggerBase):
    def __init__(self, interval_s: float = 0.0) -> None:
        super().__init__(interval_s=interval_s)
        self.logged: list[dict] = []

    def log(self, summary) -> None:  # type: ignore[override]
        self.logged.append(summary.to_dict())


def test_logger_manager_triggers_logging_on_interval() -> None:
    agg = OrchestratorAggregator(num_stages=1, enable_debug_events=False, wall_start_ts=0.0)
    agg.set_final_stage_map({"r": 0})
    dummy_logger = _DummyLogger(interval_s=0.0)
    mgr = OmniStatLoggerManager(
        aggregator=agg,
        loggers=[dummy_logger],
        final_stage_map_provider=lambda: agg.final_stage_map,
    )
    agg.set_logger_manager(mgr)
    agg.stage_first_ts[0] = 0.0
    agg.stage_last_ts[0] = 0.01
    agg.on_stage_metrics(
        0,
        "r",
        {"num_tokens_out": 1, "stage_gen_time_ms": 1.0, "batch_id": 1, "batch_size": 1, "rx_transfer_bytes": 0},
    )
    agg.on_finalize_request(0, "r", req_start_ts=0.0)
    assert dummy_logger.logged, "logger manager should emit summary when interval is reached"
