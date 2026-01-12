from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pprint import pformat
from typing import Any, Callable

from vllm.logger import init_logger

from .stats import RunSummary

logger = init_logger(__name__)


class OmniStatLoggerBase(ABC):
    """Abstract logger for Omni stats."""

    def __init__(self, interval_s: float = 10.0, enabled: bool = True) -> None:
        self.interval_s = float(interval_s)
        self.enabled = enabled
        self._last_log_ts: float = 0.0

    def should_log(self, now: float) -> bool:
        if not self.enabled:
            return False
        return (now - self._last_log_ts) >= self.interval_s

    def update_ts(self, now: float) -> None:
        self._last_log_ts = now

    @abstractmethod
    def log(self, summary: RunSummary) -> None:
        raise NotImplementedError


class OmniLoggingStatLogger(OmniStatLoggerBase):
    """Console/file logger for aggregated Omni stats."""

    def __init__(self, interval_s: float = 10.0, enabled: bool = True) -> None:
        super().__init__(interval_s=interval_s, enabled=enabled)

    def log(self, summary: RunSummary) -> None:
        logger.info("[OmniStats] %s", pformat(summary.to_dict(), sort_dicts=False))


class OmniStatLoggerManager:
    """Manages multiple stat loggers over a shared aggregator."""

    def __init__(
        self,
        aggregator: Any,
        loggers: list[OmniStatLoggerBase] | None = None,
        final_stage_map_provider: Callable[[], dict[str, int] | int | None] | None = None,
    ) -> None:
        self.aggregator = aggregator
        self.loggers = loggers or []
        self.final_stage_map_provider = final_stage_map_provider
        if hasattr(self.aggregator, "set_logger_manager"):
            self.aggregator.set_logger_manager(self)

    def maybe_log(self) -> None:
        now = time.time()
        ready_loggers = [l for l in self.loggers if l.should_log(now)]
        if not ready_loggers:
            return
        final_stage = self.final_stage_map_provider() if self.final_stage_map_provider else None
        summary = self.aggregator.build_run_summary(final_stage)
        for lg in ready_loggers:
            lg.log(summary)
            lg.update_ts(now)

    def force_log(self) -> None:
        final_stage = self.final_stage_map_provider() if self.final_stage_map_provider else None
        summary = self.aggregator.build_run_summary(final_stage)
        for lg in self.loggers:
            if not lg.enabled:
                continue
            lg.log(summary)
            lg.update_ts(time.time())
