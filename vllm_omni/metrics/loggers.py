from pprint import pformat
from vllm.logger import init_logger
from vllm_omni.metrics.stats import OrchestratorAggregator

logger = init_logger(__name__)


class OmniLoggingStatLogger:
    """Console/file logger for aggregated Omni stats."""

    def __init__(self, aggregator: OrchestratorAggregator, enable_stats: bool) -> None:
        self.aggregator = aggregator
        self.enable_stats = enable_stats
        self.summary = None
        self.engine_is_idle = False
        self.last_e2e_total_tokens = 0
        self.last_e2e_total_ms = 0.0
    
    def do_log_stats(self) -> None:
        """Log the aggregated stats if logging is enabled and engine is not idle."""
        if not self.enable_stats:
            # Clear events to avoid memory growth when stats logging is disabled
            self.aggregator.reset_events()
            return
        log_fn = logger.debug if self.engine_is_idle else logger.info
        self._merge()  # Merge all events into a summary
        log_fn("[OmniStats Summary] %s", pformat(self.summary, sort_dicts=False))
        self._reset()  # Reset after logging

    def _reset(self) -> None:
        self.summary = None
        self.last_e2e_total_tokens = 0
        self.last_e2e_total_ms = 0.0
        self.aggregator.reset_events()

    def _merge(self) -> None:
        """Merge all events into a summary."""
        self.summary = self.aggregator.merge(self.aggregator.e2e_events+self.aggregator.stage_events+self.aggregator.transfer_events)
    
    def _update_engine_idle(self) -> None:
        delta_time = self.aggregator.e2e_total_ms - self.last_e2e_total_ms
        delta_token = self.aggregator.e2e_total_tokens - self.last_e2e_total_tokens
        generation_throughput = (
            delta_token / delta_time * 1000.0 if delta_time > 0 else 0.0
        )
        self.engine_is_idle = not any(
            (
                generation_throughput,
                self.last_e2e_total_ms,
                self.last_e2e_total_tokens,
            )
        )
        self.last_e2e_total_ms = self.aggregator.e2e_total_ms
        self.last_e2e_total_tokens = self.aggregator.e2e_total_tokens