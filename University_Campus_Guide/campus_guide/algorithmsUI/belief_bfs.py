from __future__ import annotations

from collections import deque
import time
from typing import List

from map_data import GridPos, Stage

from .belief_common import (
    Belief,
    _apply_belief_action,
    _belief_frontier_view,
    _belief_goal_progress,
    _belief_goal_set,
    _belief_goal_status_text,
    _belief_h,
    _belief_initial,
    _belief_note,
    _belief_paths_from_actions,
    _belief_rep_next,
    _belief_text,
    _is_goal_belief,
    _representative_path_to_goal,
)
from .common import NeighborInfo, SearchResult, finish_search as _finish
from .trace_recorder import TraceRecorder


ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
MAX_EXPANSIONS = 9000
MAX_TRACE_STEPS = MAX_EXPANSIONS + 1


def belief_bfs(stage: Stage) -> SearchResult:
    name = "Belief State BFS"
    aggregate = "MAX"
    goal_size = 1
    start_time = time.perf_counter()
    start_b = _belief_initial(stage)
    goals = _belief_goal_set(stage, goal_size)

    frontier = deque([(start_b, [stage.start], [])])
    reached = {start_b}
    reached_order: List[Belief] = []
    recorder = TraceRecorder(name, MAX_TRACE_STEPS)
    trace = recorder.steps
    expanded = 0
    best_path: List[GridPos] = [stage.start]
    best_h = _belief_h(start_b, goals, aggregate)
    best_actions: List[str] = []
    best_belief: Belief = start_b

    while frontier and expanded < MAX_EXPANSIONS:
        belief, rep_path, actions = frontier.popleft()
        rep = rep_path[-1]
        h_b = _belief_h(belief, goals, aggregate)
        reached_order.append(belief)
        expanded += 1
        if h_b < best_h:
            best_h = h_b
            best_path = rep_path
            best_actions = actions
            best_belief = belief

        infos: List[NeighborInfo] = []
        reached_goal_count = _belief_goal_progress(belief, goals)
        if reached_goal_count and not _is_goal_belief(belief, goals):
            infos.append(
                NeighborInfo(
                    rep,
                    "GOAL-PARTIAL",
                    _belief_goal_status_text(belief, goals),
                    "INFO",
                    "M\u1ed9t ph\u1ea7n belief \u0111\u00e3 t\u1edbi Goal v\u00e0 \u0111\u1ee9ng y\u00ean; BFS v\u1eabn ph\u1ea3i ti\u1ebfp t\u1ee5c v\u00ec c\u00f2n tr\u1ea1ng th\u00e1i ch\u01b0a t\u1edbi Goal.",
                )
            )

        if _is_goal_belief(belief, goals):
            if recorder.can_record():
                recorder.add(
                    rep,
                    _belief_frontier_view(frontier_b for frontier_b, _, _ in frontier),
                    _belief_frontier_view(reached_order),
                    [NeighborInfo(rep, "GOAL-TEST", "belief_goal=True", "OK", "T\u1ea5t c\u1ea3 tr\u1ea1ng th\u00e1i trong belief \u0111\u00e3 t\u1edbi Goal.")],
                    len(actions),
                    h_b,
                    _belief_note(name, aggregate, goal_size, belief)
                    + "\nBFS d\u1eebng: m\u1ecdi tr\u1ea1ng th\u00e1i trong belief \u0111\u1ec1u \u1edf Goal. Thu\u1eadt to\u00e1n tr\u1ea3 v\u1ec1 l\u1eddi gi\u1ea3i ho\u00e0n ch\u1ec9nh.",
                )
            result_path = _representative_path_to_goal(rep_path, stage, use_cost=False)
            result = _finish(name, start_time, result_path, stage, expanded, trace)
            result.belief_paths = _belief_paths_from_actions(stage, actions, goals)
            return result

        for action in ACTIONS:
            next_belief = _apply_belief_action(belief, action, stage, goals)
            rep_next = _belief_rep_next(rep, action, next_belief, stage)
            value = f"B'={_belief_text(next_belief)}; {_belief_goal_status_text(next_belief, goals)}"
            if next_belief in reached:
                infos.append(NeighborInfo(rep_next, action, value, "SKIP", "Belief n\u00e0y \u0111\u00e3 c\u00f3 trong reached n\u00ean BFS kh\u00f4ng th\u00eam l\u1ea1i."))
                continue
            reached.add(next_belief)
            frontier.append((next_belief, rep_path + [rep_next], actions + [action]))
            status = "GOAL" if _is_goal_belief(next_belief, goals) else "ADD"
            reason = (
                "Action n\u00e0y t\u1ea1o belief to\u00e0n Goal; khi pop kh\u1ecfi Queue, BFS s\u1ebd d\u1eebng th\u00e0nh c\u00f4ng."
                if status == "GOAL"
                else "Th\u00eam belief m\u1edbi v\u00e0o cu\u1ed1i Queue FIFO."
            )
            infos.append(NeighborInfo(rep_next, action, value, status, reason))

        if recorder.can_record():
            recorder.add(
                rep,
                _belief_frontier_view(frontier_b for frontier_b, _, _ in frontier),
                _belief_frontier_view(reached_order),
                infos,
                len(actions),
                h_b,
                _belief_note(name, aggregate, goal_size, belief)
                + "\nBFS tr\u00ean belief state: pop Queue FIFO, \u00e1p d\u1ee5ng c\u00f9ng m\u1ed9t action cho t\u1ea5t c\u1ea3 START? trong belief.",
            )

    if recorder.can_record():
        recorder.add(
            best_path[-1],
            [],
            _belief_frontier_view(reached_order),
            [NeighborInfo(best_path[-1], "STOP", f"h(B) t\u1ed1t nh\u1ea5t={best_h:.1f}", "FAIL", "Kh\u00f4ng t\u00ecm \u0111\u01b0\u1ee3c chu\u1ed7i action chung \u0111\u01b0a to\u00e0n b\u1ed9 belief t\u1edbi Goal.")],
            len(best_actions),
            best_h,
            _belief_note(name, aggregate, goal_size, best_belief)
            + "\nBFS \u0111\u00e3 duy\u1ec7t h\u1ebft belief reachable ho\u1eb7c ch\u1ea1m gi\u1edbi h\u1ea1n m\u1edf r\u1ed9ng. Kh\u00f4ng c\u00f3 chu\u1ed7i action chung \u0111\u01b0a to\u00e0n b\u1ed9 START? v\u1ec1 \u0111\u00fang Goal, n\u00ean nh\u00e2n v\u1eadt \u0111\u1ee9ng t\u1ea1i belief t\u1ed1t nh\u1ea5t v\u00e0 kh\u00f4ng v\u1ebd \u0111\u01b0\u1eddng gi\u1ea3 t\u1edbi Goal.",
        )

    result = _finish(
        name,
        start_time,
        best_path,
        stage,
        expanded,
        trace,
        status="D\u1eebng - ch\u01b0a \u0111\u1ea1t Goal",
    )
    result.belief_paths = []
    return result
