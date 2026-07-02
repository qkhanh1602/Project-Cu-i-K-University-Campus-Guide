from __future__ import annotations

from collections import deque

from map_data import Stage, neighbors

from .common import PureResult, best_partial, reconstruct


def bfs(stage: Stage) -> PureResult:
    if stage.start == stage.goal:
        return PureResult([stage.start], expanded=0, success=True)

    frontier = deque([stage.start])
    parent = {stage.start: None}
    explored = set()
    expanded = 0

    while frontier:
        node = frontier.popleft()
        explored.add(node)
        expanded += 1

        for child in neighbors(node, stage):
            if child in explored or child in parent:
                continue

            parent[child] = node

            if child == stage.goal:
                return PureResult(reconstruct(parent, child), expanded, success=True)

            frontier.append(child)

    return PureResult(best_partial(parent, stage), expanded, success=False)
