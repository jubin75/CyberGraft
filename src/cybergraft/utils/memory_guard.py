"""RSS safety guard for the 16 GB target machine."""

import logging
from dataclasses import dataclass
from typing import Optional

import psutil


@dataclass
class MemoryGuard:
    """Check resident set size and refuse unsafe experiments."""

    warning_gb: float = 12.0
    abort_gb: float = 14.0

    def __post_init__(self) -> None:
        if self.warning_gb <= 0 or self.abort_gb <= self.warning_gb:
            raise ValueError("Require 0 < warning_gb < abort_gb")

    def rss_gb(self) -> float:
        return psutil.Process().memory_info().rss / (1024 ** 3)

    def check(self, stage: str, logger: Optional[logging.Logger] = None) -> float:
        rss = self.rss_gb()
        if rss > self.abort_gb:
            raise MemoryError(
                f"RSS {rss:.2f} GB exceeded {self.abort_gb:.2f} GB safety limit during {stage}"
            )
        if rss > self.warning_gb and logger is not None:
            logger.warning("RSS %.2f GB exceeded %.2f GB warning threshold during %s", rss, self.warning_gb, stage)
        return rss
