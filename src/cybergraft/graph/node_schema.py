"""Schema for a graph node.

The fields are intentionally kept even for the toy graph so later data adapters
cannot silently discard species and provenance context.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class Node:
    """A neuron-like node used by the simulator."""

    id: int
    species: str
    cell_type: str
    subsystem: str
    input_class: str = "unknown"
    output_class: str = "unknown"
    position: Optional[Tuple[float, float, float]] = None

    def __post_init__(self) -> None:
        if self.id < 0:
            raise ValueError("Node id must be non-negative")
        for field_name in ("species", "cell_type", "subsystem"):
            if not getattr(self, field_name):
                raise ValueError(f"Node {field_name} must be non-empty")
