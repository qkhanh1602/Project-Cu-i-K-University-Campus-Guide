from __future__ import annotations

from map_data import Stage, neighbors

from .common import PureResult, heuristic


def local_beam_search(stage: Stage, beam_width: int = 3, max_steps: int = 300) -> PureResult:
    current_state_set = [(stage.start, [stage.start])]
    expanded = 0

    for _ in range(max_steps):
        neighbor_states = []
        for current, path in current_state_set:
            expanded += 1
            if current == stage.goal:
                return PureResult(path, expanded, success=True)
            for nb in neighbors(current, stage):
                neighbor_states.append((nb, path + [nb]))

        if not neighbor_states:
            break

        neighbor_states.sort(key=lambda item: heuristic(item[0], stage.goal))
        current_state_set = neighbor_states[:beam_width]

    best_pos, best_path = min(current_state_set, key=lambda item: heuristic(item[0], stage.goal))
    return PureResult(best_path, expanded, success=(best_pos == stage.goal))
