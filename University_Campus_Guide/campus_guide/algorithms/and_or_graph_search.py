from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from map_data import GridPos, Stage, is_walkable, neighbors
from algorithmsUI.common import move

from .common import PureResult, heuristic


Plan = List[object] | Tuple[str, Dict[GridPos, object]]
FAILURE = None
ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")


def action_results(state: GridPos, action: str, stage: Stage) -> List[GridPos]:
    intended = move(state, action)
    if not is_walkable(intended, stage):
        return []
    return [intended]


def and_or_graph_search(stage: Stage) -> PureResult:
    expanded = 0

    def or_search(state: GridPos, path: List[GridPos]) -> Optional[Plan]:
        nonlocal expanded
        expanded += 1

        if state == stage.goal:
            return []
        if state in path:
            return FAILURE

        ordered_actions = sorted(
            ACTIONS,
            key=lambda action: heuristic(move(state, action), stage.goal),
        )

        for action in ordered_actions:
            result_states = action_results(state, action, stage)
            if not result_states:
                continue
            plan = and_search(result_states, path + [state])
            if plan is not FAILURE:
                return (action, plan)
        return FAILURE

    def and_search(states: List[GridPos], path: List[GridPos]) -> Optional[Dict[GridPos, object]]:
        plans: Dict[GridPos, object] = {}
        for state in states:
            plan = or_search(state, path)
            if plan is FAILURE:
                return FAILURE
            plans[state] = plan
        return plans

    def flatten(state: GridPos, plan: object) -> List[GridPos]:
        if state == stage.goal or plan == [] or not isinstance(plan, tuple):
            return [state]
        action, subplans = plan
        result_states = action_results(state, action, stage)
        if not result_states:
            return [state]
        next_state = min(result_states, key=lambda pos: heuristic(pos, stage.goal))
        child = subplans.get(next_state) if isinstance(subplans, dict) else None
        return [state] + flatten(next_state, child)

    plan = or_search(stage.start, [])
    if plan is FAILURE:
        return PureResult([stage.start], expanded, success=False)
    path = flatten(stage.start, plan)
    return PureResult(path, expanded, success=(path[-1] == stage.goal))
