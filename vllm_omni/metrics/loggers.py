from abc import ABC, abstractmethod
from pprint import pformat
from vllm_omni.metrics.stats import OrchestratorAggregator
from vllm.logger import init_logger

logger = init_logger(__name__)


class OmniStatLoggerBase(ABC):
    """Abstract logger for Omni stats."""
    
    def __init__(self, log_enable: bool = True) -> None:
        self.log_enable = log_enable

    @abstractmethod
    def force_log(self) -> None:
        raise NotImplementedError
    
    def reset(self) -> None:
        pass

    def merge(self) -> None:
        pass

    
class OmniLoggingStatLogger(OmniStatLoggerBase):
    """Console/file logger for aggregated Omni stats."""

    def __init__(self, aggregator: OrchestratorAggregator, log_enable: bool = True) -> None:
        super().__init__(log_enable=log_enable)
        self.aggregator = aggregator
        self.summery = None
    
    def force_log(self) -> None:
        if not self.log_enable:
            return
        self.merge()
        logger.info("[OmniStats Summary] %s", pformat(self.summery, sort_dicts=False))

    def reset(self) -> None:
        self.summery = None
        self.aggregator.reset_events()

    def merge(self) -> None:
        # Merge all events into a summary
        self.summery = self.aggregator.merge(self.aggregator.e2e_events+self.aggregator.stage_events+self.aggregator.transfer_events)
            