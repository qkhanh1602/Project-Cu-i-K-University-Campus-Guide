from __future__ import annotations

from typing import List

from map_data import GridPos, Stage, movement_cost, neighbors

from .common import PureResult, heuristic


FOUND = object()


def ida_star(stage: Stage) -> PureResult:
    path: List[GridPos] = [stage.start]
    threshold = heuristic(stage.start, stage.goal)
    expanded = 0

    def search(g: float, threshold: float):
        nonlocal expanded
        current = path[-1]
        f = g + heuristic(current, stage.goal)
        if f > threshold:
            return f
        expanded += 1
        if current == stage.goal:
            return FOUND

        next_threshold = float("inf")
        for nb in neighbors(current, stage):
            if nb in path:
                continue
            path.append(nb)
            result = search(g + float(movement_cost(nb, stage)), threshold)
            if result is FOUND:
                return FOUND
            next_threshold = min(next_threshold, float(result))
            path.pop()
        return next_threshold

    while True:
        result = search(0.0, threshold)
        if result is FOUND:
            return PureResult(path[:], expanded, success=True)
        if result == float("inf"):
            return PureResult(path[:], expanded, success=False)
        threshold = float(result)
