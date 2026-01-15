from abc import ABC, abstractmethod
from pprint import pformat
from vllm.logger import init_logger

logger = init_logger(__name__)


class OmniStatLoggerBase(ABC):
    """Abstract logger for Omni stats."""
    
    def __init__(self, log_enable: bool = True) -> None:
        self.log_enable = log_enable

    @abstractmethod
    def log(self, summary: RunSummary) -> None:
        raise NotImplementedError
    
class OmniLoggingStatLogger(OmniStatLoggerBase):
    """Console/file logger for aggregated Omni stats."""

    def __init__(self, log_enable: bool = True) -> None:
        super().__init__(log_enable=log_enable)
    
    def log(self, summary: RunSummary) -> None:
        if not self.log_enable:
            return
        # Implement logging logic here
        logger.info("[OmniStats] %s", pformat(summary.to_dict(), sort_dicts=False))