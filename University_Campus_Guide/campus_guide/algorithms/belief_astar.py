from __future__ import annotations

import heapq
import itertools

from map_data import Stage
from algorithmsUI.belief_common import (
    _apply_belief_action,
    _belief_goal_set,
    _belief_h,
    _belief_initial,
    _belief_rep_next,
    _belief_step_cost,
    _is_goal_belief,
    _representative_path_to_goal,
)

from .common import PureResult


ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")


def belief_astar(stage: Stage, max_expansions: int = 9000) -> PureResult:
    goals = _belief_goal_set(stage)
    start_belief = _belief_initial(stage)
    counter = itertools.count()
    frontier = [(_belief_h(start_belief, goals), next(counter), start_belief, [stage.start], 0.0)]
    best_g = {start_belief: 0.0}
    expanded = 0

    while frontier and expanded < max_expansions:
        _, _, belief, rep_path, g = heapq.heappop(frontier)
        expanded += 1

        if _is_goal_belief(belief, goals):
            return PureResult(_representative_path_to_goal(rep_path, stage, use_cost=True), expanded, success=True)

        for action in ACTIONS:
            next_belief = _apply_belief_action(belief, action, stage, goals)
            next_g = g + _belief_step_cost(stage, belief, next_belief)
            if next_g >= best_g.get(next_belief, float("inf")):
                continue
            best_g[next_belief] = next_g
            rep_next = _belief_rep_next(rep_path[-1], action, next_belief, stage)
            f = next_g + _belief_h(next_belief, goals)
            heapq.heappush(frontier, (f, next(counter), next_belief, rep_path + [rep_next], next_g))

    return PureResult([stage.start], expanded, success=False)
