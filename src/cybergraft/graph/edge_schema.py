"""Schema for a directed sparse synaptic edge."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Edge:
    """Directed edge from ``source`` to ``target``.

    ``sign`` stays optional because M0's toy graph is explicitly an engineering
    model, not a claim about neurotransmitter identity.
    """

    source: int
    target: int
    weight: float
    provenance: str
    delay_ms: Optional[float] = None
    sign: Optional[str] = None
    sign_source: Optional[str] = None

    def __post_init__(self) -> None:
        if self.source < 0 or self.target < 0:
            raise ValueError("Edge endpoints must be non-negative")
        if not self.provenance:
            raise ValueError("Edge provenance must be non-empty")
        if self.delay_ms is not None and self.delay_ms < 0:
            raise ValueError("Edge delay_ms must be non-negative")
