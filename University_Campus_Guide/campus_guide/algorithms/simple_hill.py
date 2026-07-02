from __future__ import annotations

from map_data import Stage, neighbors

from .common import PureResult, heuristic


def simple_hill_climbing(stage: Stage, max_steps: int = 300) -> PureResult:
    current = stage.start
    path = [current]
    expanded = 0

    for _ in range(max_steps):
        expanded += 1
        if current == stage.goal:
            return PureResult(path, expanded, success=True)

        current_h = heuristic(current, stage.goal)
        next_state = None
        for nb in neighbors(current, stage):
            if next_state is None and heuristic(nb, stage.goal) < current_h:
                next_state = nb

        if next_state is None:
            return PureResult(path, expanded, success=False)

        current = next_state
        path.append(current)

    return PureResult(path, expanded, success=False)
