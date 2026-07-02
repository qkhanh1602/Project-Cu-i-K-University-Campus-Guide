from __future__ import annotations

from map_data import GridPos, Stage, neighbors
from algorithmsUI.game_common import (
    LOOKAHEAD_DEPTH,
    MAX_AGENT_BRANCH,
    MAX_MIN_BRANCH,
    _agent_successors,
    _game_static_value,
    _opponent_successors,
    opponent_positions_for_route,
)

from .common import PureResult


def legal_player_moves(current: GridPos, path: list[GridPos], enemies: list[GridPos], stage: Stage) -> list[GridPos]:
    return [nb for nb in neighbors(current, stage) if nb not in enemies and nb not in path]


def max_value(pos: GridPos, stage: Stage, depth: int, enemies: list[GridPos]) -> float:
    if depth <= 0 or pos == stage.goal:
        return _game_static_value(pos, stage, enemies=enemies)

    actions = _agent_successors(pos, stage, enemies, MAX_AGENT_BRANCH)
    if not actions:
        return _game_static_value(pos, stage, enemies=enemies)

    return max(min_value(action, stage, depth - 1, enemies) for action in actions)


def min_value(pos: GridPos, stage: Stage, depth: int, enemies: list[GridPos]) -> float:
    if depth <= 0 or pos == stage.goal:
        return _game_static_value(pos, stage, enemies=enemies)

    opponent_moves = _opponent_successors(pos, enemies, stage)[:MAX_MIN_BRANCH]
    if not opponent_moves:
        return _game_static_value(pos, stage, enemies=enemies)

    return min(max_value(pos, stage, depth - 1, next_enemies) for next_enemies in opponent_moves)


def minimax(stage: Stage, max_steps: int = 80) -> PureResult:
    current = stage.start
    path = [current]

    for step in range(max_steps):
        if current == stage.goal:
            return PureResult(path, step, success=True)

        enemies = opponent_positions_for_route(path, stage)
        legal = legal_player_moves(current, path, enemies, stage)
        if not legal:
            break

        current = max(
            legal,
            key=lambda nb: min_value(nb, stage, LOOKAHEAD_DEPTH, enemies),
        )
        path.append(current)

    return PureResult(path, len(path) - 1, success=(path[-1] == stage.goal))
