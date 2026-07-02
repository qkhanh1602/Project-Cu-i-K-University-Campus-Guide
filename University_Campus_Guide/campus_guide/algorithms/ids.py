from __future__ import annotations

from map_data import GridPos, Stage, neighbors

from .common import PureResult, best_partial, reconstruct


CUTOFF = "cutoff"
FAILURE = "failure"


def depth_limited_search(stage: Stage, limit: int) -> tuple[GridPos | str, dict[GridPos, GridPos | None], int]:
    parent: dict[GridPos, GridPos | None] = {stage.start: None}
    expanded = 0

    def recursive_dls(node: GridPos, depth: int, path_seen: set[GridPos]) -> GridPos | str:
        nonlocal expanded
        expanded += 1
        if node == stage.goal:
            return node
        if depth == limit:
            return CUTOFF

        result: GridPos | str = FAILURE
        for child in neighbors(node, stage):
            if child in path_seen:
                continue
            parent[child] = node
            child_result = recursive_dls(child, depth + 1, path_seen | {child})
            if child_result == CUTOFF:
                result = CUTOFF
            elif child_result != FAILURE:
                return child_result
        return result

    return recursive_dls(stage.start, 0, {stage.start}), parent, expanded


def ids(stage: Stage, max_depth: int = 200) -> PureResult:
    total_expanded = 0
    best_parent = {stage.start: None}

    for limit in range(max_depth + 1):
        result, parent, expanded = depth_limited_search(stage, limit)
        total_expanded += expanded
        best_parent = parent
        if result != CUTOFF and result != FAILURE:
            return PureResult(reconstruct(parent, result), total_expanded, success=True)
        if result == FAILURE:
            return PureResult(best_partial(parent, stage), total_expanded, success=False)

    return PureResult(best_partial(best_parent, stage), total_expanded, success=False)
