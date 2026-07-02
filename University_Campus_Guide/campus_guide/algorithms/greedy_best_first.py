from __future__ import annotations

import heapq

from map_data import Stage, neighbors

from .common import PureResult, best_partial, heuristic, reconstruct


def greedy_best_first(stage: Stage) -> PureResult:
    frontier = [(heuristic(stage.start, stage.goal), 0, stage.start)]
    parent = {stage.start: None}
    reached = set()
    order = 1
    expanded = 0

    while frontier:
        _, _, current = heapq.heappop(frontier)
        if current in reached:
            continue
        reached.add(current)
        expanded += 1

        if current == stage.goal:
            return PureResult(reconstruct(parent, current), expanded, success=True)

        for nb in neighbors(current, stage):
            if nb in reached or nb in parent:
                continue
            parent[nb] = current
            heapq.heappush(frontier, (heuristic(nb, stage.goal), order, nb))
            order += 1

    return PureResult(best_partial(parent, stage), expanded, success=False)
