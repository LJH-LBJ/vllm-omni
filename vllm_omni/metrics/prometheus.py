from __future__ import annotations

from typing import Any

from vllm.logger import init_logger

from .loggers import OmniStatLoggerBase
from .stats import RunSummary

logger = init_logger(__name__)


class OmniPrometheusStatLogger(OmniStatLoggerBase):
    """Prometheus exporter for Omni stats (best-effort, optional dependency)."""

    def __init__(self, interval_s: float = 10.0, enabled: bool = True, registry: Any = None, prefix: str = "omni"):
        super().__init__(interval_s=interval_s, enabled=enabled)
        try:
            from prometheus_client import CollectorRegistry, Gauge
        except Exception:
            logger.debug("prometheus_client not available; disabling Prometheus stat logger")
            self.enabled = False
            self._registry = None
            self._Gauge = None
            return
        self._registry = registry or CollectorRegistry(auto_describe=True)
        self._Gauge = Gauge
        self.prefix = prefix
        self.stage_requests = self._Gauge(
            f"{prefix}_stage_requests",
            "Number of requests per stage",
            ["stage_id"],
            registry=self._registry,
        )
        self.stage_tokens_per_s = self._Gauge(
            f"{prefix}_stage_avg_tokens_per_s",
            "Average tokens per second per stage",
            ["stage_id"],
            registry=self._registry,
        )
        self.transfer_mbps = self._Gauge(
            f"{prefix}_transfer_total_mbps",
            "Average transfer Mbps per edge",
            ["from_stage", "to_stage"],
            registry=self._registry,
        )
        self.e2e_avg_latency = self._Gauge(
            f"{prefix}_e2e_avg_latency_ms",
            "Average end-to-end latency per request (ms)",
            registry=self._registry,
        )
        self.e2e_throughput = self._Gauge(
            f"{prefix}_e2e_avg_tokens_per_s",
            "Average end-to-end tokens per second",
            registry=self._registry,
        )

    @property
    def registry(self) -> Any | None:
        return getattr(self, "_registry", None)

    def log(self, summary: RunSummary) -> None:
        if not self.enabled or self._Gauge is None or self._registry is None:
            return
        for stage in summary.stages:
            sid = str(stage.get("stage_id"))
            self.stage_requests.labels(stage_id=sid).set(stage.get("requests", 0))
            self.stage_tokens_per_s.labels(stage_id=sid).set(stage.get("avg_tokens_per_s", 0.0))
        for transfer in summary.transfers:
            self.transfer_mbps.labels(
                from_stage=str(transfer.get("from_stage")),
                to_stage=str(transfer.get("to_stage")),
            ).set(transfer.get("total_mbps", 0.0))
        self.e2e_avg_latency.set(summary.e2e_avg_time_per_request_ms)
        self.e2e_throughput.set(summary.e2e_avg_tokens_per_s)
