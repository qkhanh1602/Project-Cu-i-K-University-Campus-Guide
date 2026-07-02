from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from map_data import GridPos, Stage, manhattan, movement_cost


@dataclass
class PureResult:
    path: List[GridPos]
    expanded: int = 0
    assignment: Optional[dict[str, Any]] = None
    success: bool = False


def heuristic(pos: GridPos, goal: GridPos) -> int:
    return manhattan(pos, goal)


def path_cost(path: List[GridPos], stage: Stage) -> float:
    return sum(float(movement_cost(pos, stage)) for pos in path[1:])


def reconstruct(parent: Dict[GridPos, Optional[GridPos]], goal: GridPos) -> List[GridPos]:
    path: List[GridPos] = []
    cur: Optional[GridPos] = goal
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur)
    path.reverse()
    return path


def best_partial(parent: Dict[GridPos, Optional[GridPos]], stage: Stage) -> List[GridPos]:
    if not parent:
        return [stage.start]
    best = min(parent, key=lambda pos: heuristic(pos, stage.goal))
    return reconstruct(parent, best)
