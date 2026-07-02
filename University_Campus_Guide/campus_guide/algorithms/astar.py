from __future__ import annotations

import heapq

from map_data import Stage, movement_cost, neighbors

from .common import PureResult, best_partial, heuristic, reconstruct


def astar(stage: Stage) -> PureResult:
    frontier = [(heuristic(stage.start, stage.goal), 0, stage.start)]
    parent = {stage.start: None}
    g_score = {stage.start: 0.0}
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
            new_g = g_score[current] + float(movement_cost(nb, stage))
            if new_g >= g_score.get(nb, float("inf")):
                continue
            parent[nb] = current
            g_score[nb] = new_g
            f = new_g + heuristic(nb, stage.goal)
            heapq.heappush(frontier, (f, order, nb))
            order += 1

    return PureResult(best_partial(parent, stage), expanded, success=False)
