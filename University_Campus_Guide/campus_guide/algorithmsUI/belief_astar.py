from __future__ import annotations

import heapq
import itertools
import time
from typing import Dict, List, Tuple

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
    _belief_step_cost,
    _belief_text,
    _is_goal_belief,
    _representative_path_to_goal,
)
from .common import NeighborInfo, SearchResult, finish_search as _finish
from .trace_recorder import TraceRecorder


ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
MAX_EXPANSIONS = 9000
MAX_TRACE_STEPS = MAX_EXPANSIONS + 1


def belief_astar(stage: Stage) -> SearchResult:
    name = "Belief State A*"
    aggregate = "MAX"
    goal_size = 1
    start_time = time.perf_counter()
    start_b = _belief_initial(stage)
    goals = _belief_goal_set(stage, goal_size)
    h0 = _belief_h(start_b, goals, aggregate)

    frontier: List[Tuple[float, int, Belief, List[GridPos], List[str], float]] = [(h0, 0, start_b, [stage.start], [], 0.0)]
    order_counter = itertools.count(1)
    best_g: Dict[Belief, float] = {start_b: 0.0}
    reached = set()
    reached_order: List[Belief] = []
    recorder = TraceRecorder(name, MAX_TRACE_STEPS)
    trace = recorder.steps
    expanded = 0
    best_path: List[GridPos] = [stage.start]
    best_h = h0
    best_actions: List[str] = []
    best_belief: Belief = start_b

    while frontier and expanded < MAX_EXPANSIONS:
        _, _, belief, rep_path, actions, g_b = heapq.heappop(frontier)
        if belief in reached and g_b > best_g.get(belief, float("inf")):
            continue
        rep = rep_path[-1]
        h_b = _belief_h(belief, goals, aggregate)
        reached.add(belief)
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
                    "M\u1ed9t ph\u1ea7n belief \u0111\u00e3 t\u1edbi Goal v\u00e0 \u0111\u1ee9ng y\u00ean; A* v\u1eabn ph\u1ea3i ti\u1ebfp t\u1ee5c v\u00ec c\u00f2n tr\u1ea1ng th\u00e1i ch\u01b0a t\u1edbi Goal.",
                )
            )

        if _is_goal_belief(belief, goals):
            if recorder.can_record():
                recorder.add(
                    rep,
                    _belief_frontier_view(b for _, _, b, _, _, _ in frontier),
                    _belief_frontier_view(reached_order),
                    [NeighborInfo(rep, "GOAL-TEST", "belief_goal=True", "OK", "T\u1ea5t c\u1ea3 tr\u1ea1ng th\u00e1i trong belief \u0111\u00e3 t\u1edbi Goal.")],
                    g_b,
                    h_b,
                    _belief_note(name, aggregate, goal_size, belief)
                    + f"\nA* d\u1eebng: f(B)=g(B)+h(B)={g_b:.1f}+{h_b:.1f}; m\u1ecdi tr\u1ea1ng th\u00e1i trong belief \u0111\u1ec1u \u1edf Goal. Thu\u1eadt to\u00e1n tr\u1ea3 v\u1ec1 l\u1eddi gi\u1ea3i ho\u00e0n ch\u1ec9nh.",
                )
            result_path = _representative_path_to_goal(rep_path, stage, use_cost=True)
            result = _finish(name, start_time, result_path, stage, expanded, trace)
            result.belief_paths = _belief_paths_from_actions(stage, actions, goals)
            return result

        for action in ACTIONS:
            next_belief = _apply_belief_action(belief, action, stage, goals)
            rep_next = _belief_rep_next(rep, action, next_belief, stage)
            step_cost = _belief_step_cost(stage, belief, next_belief)
            next_g = g_b + step_cost
            next_h = _belief_h(next_belief, goals, aggregate)
            next_f = next_g + next_h
            value = f"g={next_g:.1f}, h(B)={next_h:.1f}, f={next_f:.1f}; B'={_belief_text(next_belief)}; {_belief_goal_status_text(next_belief, goals)}"

            if next_belief in reached and next_g >= best_g.get(next_belief, float("inf")):
                infos.append(NeighborInfo(rep_next, action, value, "SKIP", "Reached \u0111\u00e3 c\u00f3 belief n\u00e0y v\u1edbi g(B) t\u1ed1t h\u01a1n."))
                continue

            if next_g < best_g.get(next_belief, float("inf")):
                best_g[next_belief] = next_g
                heapq.heappush(frontier, (next_f, next(order_counter), next_belief, rep_path + [rep_next], actions + [action], next_g))
                status = "GOAL" if _is_goal_belief(next_belief, goals) else "ADD/UPDATE"
                reason = (
                    "Action n\u00e0y t\u1ea1o belief to\u00e0n Goal; khi l\u1ea5y kh\u1ecfi Priority Queue, A* s\u1ebd d\u1eebng th\u00e0nh c\u00f4ng."
                    if status == "GOAL"
                    else "Th\u00eam/c\u1eadp nh\u1eadt Priority Queue theo f(B)=g(B)+h(B)."
                )
                infos.append(NeighborInfo(rep_next, action, value, status, reason))
            else:
                infos.append(NeighborInfo(rep_next, action, value, "SKIP", "Frontier \u0111\u00e3 c\u00f3 belief n\u00e0y v\u1edbi g(B) t\u1ed1t h\u01a1n."))

        if recorder.can_record():
            recorder.add(
                rep,
                _belief_frontier_view(b for _, _, b, _, _, _ in frontier),
                _belief_frontier_view(reached_order),
                infos,
                g_b,
                h_b,
                _belief_note(name, aggregate, goal_size, belief)
                + "\nA* tr\u00ean belief state: ch\u1ecdn belief c\u00f3 f(B)=g(B)+h(B) nh\u1ecf nh\u1ea5t, r\u1ed3i \u00e1p d\u1ee5ng c\u00f9ng action cho m\u1ecdi START?.",
            )

    if recorder.can_record():
        recorder.add(
            best_path[-1],
            [],
            _belief_frontier_view(reached_order),
            [NeighborInfo(best_path[-1], "STOP", f"h(B) t\u1ed1t nh\u1ea5t={best_h:.1f}", "FAIL", "Kh\u00f4ng t\u00ecm \u0111\u01b0\u1ee3c chu\u1ed7i action chung \u0111\u01b0a to\u00e0n b\u1ed9 belief t\u1edbi Goal.")],
            float(len(best_actions)),
            best_h,
            _belief_note(name, aggregate, goal_size, best_belief)
            + "\nA* \u0111\u00e3 duy\u1ec7t h\u1ebft belief reachable theo f(B)=g(B)+h(B) ho\u1eb7c ch\u1ea1m gi\u1edbi h\u1ea1n m\u1edf r\u1ed9ng. Kh\u00f4ng c\u00f3 chu\u1ed7i action chung \u0111\u01b0a to\u00e0n b\u1ed9 START? v\u1ec1 \u0111\u00fang Goal, n\u00ean nh\u00e2n v\u1eadt \u0111\u1ee9ng t\u1ea1i belief t\u1ed1t nh\u1ea5t v\u00e0 kh\u00f4ng v\u1ebd \u0111\u01b0\u1eddng gi\u1ea3 t\u1edbi Goal.",
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
