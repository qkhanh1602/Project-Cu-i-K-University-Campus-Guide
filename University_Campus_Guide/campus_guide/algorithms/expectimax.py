from __future__ import annotations

from map_data import GridPos, Stage, neighbors
from algorithmsUI.game_common import (
    EXPECTIMAX_DEPTH,
    MAX_AGENT_BRANCH,
    _agent_successors,
    _expectimax_chance_outcomes,
    _game_static_value,
    opponent_positions_for_route,
)

from .common import PureResult


def legal_player_moves(current: GridPos, path: list[GridPos], enemies: list[GridPos], stage: Stage) -> list[GridPos]:
    return [nb for nb in neighbors(current, stage) if nb not in enemies and nb not in path]


def max_value(pos: GridPos, stage: Stage, depth: int, enemies: list[GridPos]) -> float:
    if depth <= 0 or pos == stage.goal:
        return _game_static_value(pos, stage, "expectimax", enemies)

    actions = _agent_successors(pos, stage, enemies, MAX_AGENT_BRANCH)
    if not actions:
        return _game_static_value(pos, stage, "expectimax", enemies)

    return max(chance_value(action, stage, depth - 1, enemies) for action in actions)


def chance_value(pos: GridPos, stage: Stage, depth: int, enemies: list[GridPos]) -> float:
    if depth <= 0 or pos == stage.goal:
        return _game_static_value(pos, stage, "expectimax", enemies)

    outcomes = _expectimax_chance_outcomes(pos, stage, enemies)
    if not outcomes:
        return _game_static_value(pos, stage, "expectimax", enemies)

    expected_value = 0.0
    for probability, _label, next_enemies, extra_penalty in outcomes:
        child_value = max_value(pos, stage, depth - 1, next_enemies)
        expected_value += probability * (child_value - extra_penalty)
    return expected_value


def expectimax(stage: Stage, max_steps: int = 80) -> PureResult:
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
            key=lambda nb: chance_value(nb, stage, EXPECTIMAX_DEPTH, enemies),
        )
        path.append(current)

    return PureResult(path, len(path) - 1, success=(path[-1] == stage.goal))
