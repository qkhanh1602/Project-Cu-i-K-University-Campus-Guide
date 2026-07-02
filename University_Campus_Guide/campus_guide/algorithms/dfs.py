from __future__ import annotations

from map_data import Stage, neighbors

from .common import PureResult, best_partial, reconstruct


def dfs(stage: Stage) -> PureResult:
    if stage.start == stage.goal:
        return PureResult([stage.start], expanded=0, success=True)

    frontier = [stage.start]
    parent = {stage.start: None}
    reached = {stage.start}
    expanded = 0

    while frontier:
        node = frontier.pop()
        expanded += 1

        for child in reversed(neighbors(node, stage)):
            if child == stage.goal:
                parent[child] = node
                return PureResult(reconstruct(parent, child), expanded, success=True)

            if child not in reached and child not in frontier:
                reached.add(child)
                parent[child] = node
                frontier.append(child)

    return PureResult(best_partial(parent, stage), expanded, success=False)
