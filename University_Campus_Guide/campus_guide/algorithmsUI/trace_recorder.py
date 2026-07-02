from __future__ import annotations

from typing import Sequence

from map_data import GridPos

from .common import NeighborInfo, TraceStep


class TraceRecorder:
    """Small adapter between pure algorithm logic and the visualizer trace.

    Algorithms should decide what happened; this class turns that event into
    the UI-friendly TraceStep object.
    """

    def __init__(self, mode: str, limit: int | None = None):
        self.mode = mode
        self.limit = limit
        self.steps: list[TraceStep] = []

    def can_record(self) -> bool:
        return self.limit is None or len(self.steps) < self.limit

    def add(
        self,
        current: GridPos,
        frontier: Sequence[GridPos] | None = None,
        reached: Sequence[GridPos] | None = None,
        neighbors: Sequence[NeighborInfo] | None = None,
        cost: float = 0.0,
        heuristic: float = 0.0,
        note: str = "",
        iteration: int | None = None,
        mode: str | None = None,
        **attrs,
    ) -> TraceStep:
        step = TraceStep(
            iteration if iteration is not None else len(self.steps) + 1,
            current,
            list(frontier or []),
            list(reached or []),
            list(neighbors or []),
            mode or self.mode,
            cost,
            heuristic,
            note,
        )
        for name, value in attrs.items():
            setattr(step, name, value)
        self.steps.append(step)
        return step
