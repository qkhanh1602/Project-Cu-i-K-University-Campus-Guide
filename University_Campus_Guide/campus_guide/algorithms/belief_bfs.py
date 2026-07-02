from __future__ import annotations

from collections import deque

from map_data import Stage
from algorithmsUI.belief_common import (
    _apply_belief_action,
    _belief_goal_set,
    _belief_initial,
    _belief_rep_next,
    _is_goal_belief,
    _representative_path_to_goal,
)

from .common import PureResult


ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")


def belief_bfs(stage: Stage, max_expansions: int = 9000) -> PureResult:
    goals = _belief_goal_set(stage)
    start_belief = _belief_initial(stage)
    frontier = deque([(start_belief, [stage.start])])
    reached = {start_belief}
    expanded = 0

    while frontier and expanded < max_expansions:
        belief, rep_path = frontier.popleft()
        expanded += 1

        if _is_goal_belief(belief, goals):
            return PureResult(_representative_path_to_goal(rep_path, stage), expanded, success=True)

        for action in ACTIONS:
            next_belief = _apply_belief_action(belief, action, stage, goals)
            if next_belief in reached:
                continue
            reached.add(next_belief)
            rep_next = _belief_rep_next(rep_path[-1], action, next_belief, stage)
            frontier.append((next_belief, rep_path + [rep_next]))

    return PureResult([stage.start], expanded, success=False)
