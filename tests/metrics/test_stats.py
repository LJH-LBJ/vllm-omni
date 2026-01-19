from __future__ import annotations
from dataclasses import asdict

from vllm_omni.metrics import OmniLoggingStatLogger
from vllm_omni.metrics import OrchestratorAggregator
from vllm_omni.metrics.stats import RequestE2EStats, StageRequestStats, StageStats, TransferEdgeStats


def test_orchestrator_aggregator_builds_summary() -> None:
    agg = OrchestratorAggregator(num_stages=2, enable_stats=False, wall_start_ts=0.0)
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

    summary = agg.build_and_log_summary(final_stage_id_to_prompt={"r1": 1})
    data = asdict(summary)
    assert data["e2e_requests"] == 1
    assert len(data["stages"]) == 2
    assert data["stages"][0]["requests"] == 1
    assert data["transfers"][0]["samples"] == 1
    assert data["transfers"][0]["total_mbps"] >= 0.0


def test_logger_clears_events_when_disabled() -> None:
    agg = OrchestratorAggregator(num_stages=1, enable_stats=False, wall_start_ts=0.0)
    agg.stage_events.append(
        StageRequestStats(
            stage_id=0,
            request_id="r",
            num_tokens_in=1,
            num_tokens_out=1,
            batch_id=1,
            batch_size=1,
            num_engine_outputs=1,
            stage_gen_time_ms=1.0,
            rx_transfer_bytes=0,
            rx_decode_time_ms=0.0,
            rx_in_flight_time_ms=0.0,
            stage_stats=StageStats(total_token=1, total_gen_time=1.0),
        )
    )
    agg.e2e_events.append(
        RequestE2EStats(
            request_id="r",
            e2e_total_ms=10.0,
            e2e_total_tokens=1,
            transfers_total_time_ms=0.0,
            transfers_total_bytes=0,
            stages={},
        )
    )
    logger = OmniLoggingStatLogger(agg, enable_stats=False)
    logger.do_log_stats()

    assert not agg.stage_events
    assert not agg.transfer_events
    assert not agg.e2e_events


def test_merge_groups_by_stage_and_edge() -> None:
    agg = OrchestratorAggregator(num_stages=2, enable_stats=False, wall_start_ts=0.0)
    events = [
        StageRequestStats(
            stage_id=0,
            request_id="r1",
            num_tokens_in=2,
            num_tokens_out=4,
            batch_id=1,
            batch_size=1,
            num_engine_outputs=1,
            stage_gen_time_ms=10.0,
            rx_transfer_bytes=0,
            rx_decode_time_ms=0.0,
            rx_in_flight_time_ms=0.0,
            stage_stats=StageStats(total_token=4, total_gen_time=10.0),
        ),
        StageRequestStats(
            stage_id=0,
            request_id="r2",
            num_tokens_in=3,
            num_tokens_out=6,
            batch_id=2,
            batch_size=1,
            num_engine_outputs=1,
            stage_gen_time_ms=20.0,
            rx_transfer_bytes=0,
            rx_decode_time_ms=0.0,
            rx_in_flight_time_ms=0.0,
            stage_stats=StageStats(total_token=6, total_gen_time=20.0),
        ),
        StageRequestStats(
            stage_id=1,
            request_id="r1",
            num_tokens_in=0,
            num_tokens_out=10,
            batch_id=1,
            batch_size=1,
            num_engine_outputs=1,
            stage_gen_time_ms=15.0,
            rx_transfer_bytes=128,
            rx_decode_time_ms=1.0,
            rx_in_flight_time_ms=1.0,
            stage_stats=StageStats(total_token=10, total_gen_time=15.0),
        ),
        TransferEdgeStats(
            from_stage=0,
            to_stage=1,
            request_id="r1",
            size_bytes=100,
            tx_time_ms=2.0,
            rx_decode_time_ms=1.0,
            in_flight_time_ms=1.0,
            used_shm=False,
        ),
        RequestE2EStats(
            request_id="r1",
            e2e_total_ms=100.0,
            e2e_total_tokens=7,
            transfers_total_time_ms=5.0,
            transfers_total_bytes=100,
            stages={},
        ),
        RequestE2EStats(
            request_id="r2",
            e2e_total_ms=200.0,
            e2e_total_tokens=9,
            transfers_total_time_ms=6.0,
            transfers_total_bytes=300,
            stages={},
        ),
    ]

    merged = agg.merge(events)
    counts = merged["counts"]
    averages = merged["averages"]

    assert counts["e2e"] == 2
    assert counts["stage"]["0"] == 2
    assert counts["stage"]["1"] == 1
    assert counts["transfer"]["0->1"] == 1

    stage0_avg = averages["StageRequestStats"]["by_stage"]["0"]["num_tokens_out"]
    transfer_avg = averages["TransferEdgeStats"]["by_edge"]["0->1"]["size_bytes"]
    e2e_avg = averages["RequestE2EStats"]["e2e_total_tokens"]

    assert stage0_avg == 5.0
    assert transfer_avg == 100.0
    assert e2e_avg == 8.0
