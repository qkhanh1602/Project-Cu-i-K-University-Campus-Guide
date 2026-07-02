from __future__ import annotations

import time
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from map_data import GridPos, Stage, is_walkable, neighbors

from .common import (
    NeighborInfo,
    action_name,
    finish_search as _finish,
    h,
    move,
    path_cost,
    stopped_search as _stopped,
)
from .trace_recorder import TraceRecorder

Plan = Tuple[str, Dict[GridPos, object]]
Failure = None

ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
MAX_OR_CALLS = 4500
MAX_TRACE_STEPS = 260
MAX_SECONDS = 1.5


def and_or_graph_search(stage: Stage):
    """AND-OR-GRAPH-SEARCH for a nondeterministic campus action model.

    The code follows the classroom pseudocode:
    AND_OR_GRAPH_SEARCH(problem) -> OR_SEARCH(initial_state, problem, [])
    OR_SEARCH tries actions; each action produces result_states; AND_SEARCH
    must recursively solve every result state.
    """

    start_time = time.perf_counter()
    recorder = TraceRecorder("AND-OR Graph Search", MAX_TRACE_STEPS)
    trace = recorder.steps
    expanded = 0
    best_path: List[GridPos] = [stage.start]
    best_h = h(stage.start, stage.goal)
    timed_out = False

    def build_goal_distances() -> Dict[GridPos, int]:
        dist: Dict[GridPos, int] = {stage.goal: 0}
        frontier = deque([stage.goal])
        while frontier:
            current = frontier.popleft()
            for nb in neighbors(current, stage):
                if nb in dist:
                    continue
                dist[nb] = dist[current] + 1
                frontier.append(nb)
        return dist

    goal_dist = build_goal_distances()

    def timed_out_now() -> bool:
        return (time.perf_counter() - start_time) > MAX_SECONDS

    def ordered_actions(state: GridPos) -> List[str]:
        ranked: List[Tuple[int, int, int, str]] = []
        for index, action in enumerate(ACTIONS):
            nxt = move(state, action)
            if is_walkable(nxt, stage):
                ranked.append((goal_dist.get(nxt, 99999), h(nxt, stage.goal), index, action))
        return [action for _, _, _, action in sorted(ranked)]

    def results(state: GridPos, action: str) -> List[GridPos]:
        """Return nondeterministic result states for one action.

        To keep the map readable, nondeterminism is shown clearly at the first
        action from START: the intended move plus one possible slip. Later states
        still run through the same OR_SEARCH/AND_SEARCH code, but usually have a
        single result state, so the tree does not explode or freeze the UI.
        """

        intended = move(state, action)
        if not is_walkable(intended, stage):
            return []

        out: List[GridPos] = [intended]
        if state == stage.start:
            slip_priority = {
                "RIGHT": ("UP", "DOWN", "LEFT"),
                "LEFT": ("UP", "DOWN", "RIGHT"),
                "UP": ("RIGHT", "LEFT", "DOWN"),
                "DOWN": ("RIGHT", "LEFT", "UP"),
            }
            for slip_action in slip_priority.get(action, ()):  # e.g. RIGHT may slip UP.
                slip = move(state, slip_action)
                if slip != intended and is_walkable(slip, stage):
                    out.append(slip)
                    break

        unique: List[GridPos] = []
        for pos in out:
            if pos not in unique:
                unique.append(pos)
        return unique

    def update_best(path: List[GridPos]) -> None:
        nonlocal best_path, best_h
        if not path:
            return
        current_h = h(path[-1], stage.goal)
        if current_h < best_h or (path[-1] == stage.goal and best_path[-1] != stage.goal):
            best_path = path[:]
            best_h = current_h

    def can_reach_goal_without_path(state: GridPos, path: List[GridPos]) -> bool:
        if state == stage.goal:
            return True
        forbidden = set(path)
        frontier = deque([state])
        seen = {state}
        while frontier:
            current = frontier.popleft()
            for nb in neighbors(current, stage):
                if nb in seen:
                    continue
                if nb in forbidden and nb != stage.goal:
                    continue
                if nb == stage.goal:
                    return True
                seen.add(nb)
                frontier.append(nb)
        return False

    def add_trace(
        current: GridPos,
        frontier: List[GridPos],
        reached: List[GridPos],
        infos: List[NeighborInfo],
        note: str,
    ) -> None:
        if not recorder.can_record():
            return
        recorder.add(
            current,
            frontier[:],
            reached[:],
            infos[:],
            path_cost(reached, stage),
            h(current, stage.goal),
            note,
        )

    def or_search(state: GridPos, path: List[GridPos]) -> Optional[Plan]:
        nonlocal expanded, timed_out
        if timed_out_now() or expanded >= MAX_OR_CALLS:
            timed_out = True
            add_trace(
                state,
                [],
                path + [state],
                [NeighborInfo(state, "LIMIT", "node/time budget", "FAIL", "D\u1eebng m\u1edf r\u1ed9ng \u0111\u1ec3 tr\u00e1nh treo UI.")],
                "OR_SEARCH d\u1eebng v\u00ec v\u01b0\u1ee3t gi\u1edbi h\u1ea1n node/time; tr\u1ea3 failure cho nh\u00e1nh n\u00e0y.",
            )
            return Failure

        expanded += 1
        reached = path + [state]
        update_best(reached)

        if state == stage.goal:
            add_trace(
                state,
                [],
                reached,
                [NeighborInfo(state, "GOAL-TEST", "state in goal_test", "OK", "State l\u00e0 GOAL n\u00ean OR_SEARCH tr\u1ea3 k\u1ebf ho\u1ea1ch r\u1ed7ng [].")],
                "OR_SEARCH(state, problem, path): state thu\u1ed9c goal_test n\u00ean return [].",
            )
            return ("GOAL", {})

        if state in path:
            add_trace(
                state,
                [],
                reached,
                [NeighborInfo(state, "CYCLE-CHECK", "state in path", "FAIL", "Tr\u00e1nh l\u1eb7p: state \u0111\u00e3 n\u1eb1m trong path.")],
                "OR_SEARCH ph\u00e1t hi\u1ec7n state \u0111\u00e3 c\u00f3 trong path n\u00ean return failure.",
            )
            return Failure

        if not can_reach_goal_without_path(state, path):
            add_trace(
                state,
                [],
                reached,
                [NeighborInfo(state, "REACHABILITY", "no path without old path", "FAIL", "T\u1eeb state n\u00e0y mu\u1ed1n t\u1edbi Goal ph\u1ea3i quay l\u1ea1i path c\u0169, n\u00ean s\u1ebd t\u1ea1o v\u00f2ng l\u1eb7p.")],
                "OR_SEARCH return failure s\u1edbm v\u00ec kh\u00f4ng c\u00f3 \u0111\u01b0\u1eddng t\u1edbi Goal n\u1ebfu v\u1eabn t\u00f4n tr\u1ecdng path tr\u00e1nh l\u1eb7p.",
            )
            return Failure

        tried_infos: List[NeighborInfo] = []
        for action in ordered_actions(state):
            result_states = results(state, action)
            if not result_states:
                tried_infos.append(NeighborInfo(state, action, "results=[]", "FAIL", "Action kh\u00f4ng c\u00f3 result state h\u1ee3p l\u1ec7."))
                continue
            blocked_results = [pos for pos in result_states if not can_reach_goal_without_path(pos, reached)]
            if blocked_results:
                tried_infos.append(NeighborInfo(
                    blocked_results[0],
                    action,
                    "blocked_results=" + str(blocked_results),
                    "FAIL",
                    "C\u00f3 result state kh\u00f4ng th\u1ec3 t\u1edbi Goal n\u1ebfu kh\u00f4ng quay l\u1ea1i path c\u0169, n\u00ean action fail theo AND_SEARCH.",
                ))
                continue

            candidate_infos: List[NeighborInfo] = []
            for candidate_action in ordered_actions(state):
                candidate_results = results(state, candidate_action)
                if not candidate_results:
                    continue
                is_selected_action = candidate_action == action
                candidate_infos.append(
                    NeighborInfo(
                        candidate_results[0],
                        candidate_action,
                        "results=" + str(candidate_results),
                        "CHOSEN" if is_selected_action else "CANDIDATE",
                        (
                            "OR_SEARCH ch\u1ecdn action n\u00e0y v\u00ec AND_SEARCH(result_states) c\u00f3 conditional plan."
                            if is_selected_action
                            else "OR_SEARCH c\u00f3 x\u00e9t action l\u00e2n c\u1eadn n\u00e0y nh\u01b0ng kh\u00f4ng \u0111i theo nh\u00e1nh n\u00e0y."
                        ),
                    )
                )
            result_infos = [
                NeighborInfo(pos, "RESULT STATE", f"OR_SEARCH({pos})", "INFO", "AND_SEARCH ph\u1ea3i gi\u1ea3i result state n\u00e0y.")
                for pos in result_states
            ]

            add_trace(
                state,
                result_states,
                reached,
                candidate_infos + result_infos,
                (
                    f"OR_SEARCH({state}) x\u00e9t c\u00e1c action h\u1ee3p l\u1ec7 xung quanh state, ch\u1ecdn action {action}. "
                    f"results({state}, {action}) = {result_states}. "
                    "Sau khi OR ch\u1ecdn action, AND_SEARCH ph\u1ea3i g\u1ecdi OR_SEARCH cho t\u1eebng result state."
                ),
            )

            plan = and_search(result_states, reached)
            if plan is not Failure:
                add_trace(
                    state,
                    result_states,
                    reached,
                    [NeighborInfo(result_states[0], action, "plan != failure", "OK", "AND_SEARCH gi\u1ea3i \u0111\u01b0\u1ee3c t\u1ea5t c\u1ea3 result states n\u00ean OR_SEARCH ch\u1ecdn action n\u00e0y.")],
                    f"OR_SEARCH({state}) return [{action}, plan] v\u00ec AND_SEARCH kh\u00f4ng failure.",
                )
                return (action, plan)

            tried_infos.append(NeighborInfo(result_states[0], action, "plan == failure", "FAIL", "M\u1ed9t result state kh\u00f4ng gi\u1ea3i \u0111\u01b0\u1ee3c n\u00ean action n\u00e0y th\u1ea5t b\u1ea1i."))
            add_trace(
                state,
                result_states,
                reached,
                tried_infos,
                f"Action {action} th\u1ea5t b\u1ea1i v\u00ec AND_SEARCH(result_states) tr\u1ea3 failure. OR_SEARCH th\u1eed action kh\u00e1c n\u1ebfu c\u1ea7n.",
            )

        add_trace(
            state,
            [],
            reached,
            tried_infos,
            "OR_SEARCH th\u1eed h\u1ebft action nh\u01b0ng kh\u00f4ng c\u00f3 plan h\u1ee3p l\u1ec7 n\u00ean return failure.",
        )
        return Failure

    def and_search(states: List[GridPos], path: List[GridPos]) -> Optional[Dict[GridPos, object]]:
        plans: Dict[GridPos, object] = {}
        for state in states:
            add_trace(
                state,
                [],
                path + [state],
                [NeighborInfo(state, "AND -> OR", f"OR_SEARCH({state})", "INFO", "AND_SEARCH g\u1ecdi OR_SEARCH cho result state n\u00e0y.")],
                f"AND_SEARCH: x\u00e9t result state {state}. N\u1ebfu OR_SEARCH({state}) failure th\u00ec to\u00e0n b\u1ed9 action cha failure.",
            )
            plan_s = or_search(state, path)
            if plan_s is Failure:
                add_trace(
                    state,
                    [],
                    path + [state],
                    [NeighborInfo(state, "AND FAIL", "plan_s == failure", "FAIL", "M\u1ed9t result state fail n\u00ean AND_SEARCH return failure.")],
                    f"AND_SEARCH return failure v\u00ec result state {state} kh\u00f4ng c\u00f3 plan.",
                )
                return Failure
            plans[state] = plan_s
        add_trace(
            path[-1] if path else stage.start,
            states,
            path[:],
            [NeighborInfo(s, "AND OK", "plan_s != failure", "OK", "Result state n\u00e0y \u0111\u00e3 c\u00f3 subplan.") for s in states],
            "AND_SEARCH return plans v\u00ec m\u1ecdi result state \u0111\u1ec1u c\u00f3 plan.",
        )
        return plans

    def flatten_plan(state: GridPos, plan: object, seen: Set[GridPos]) -> List[GridPos]:
        if state == stage.goal:
            return [state]
        if state in seen or not isinstance(plan, tuple):
            return [state]
        action, subplans = plan
        if action == "GOAL":
            return [state]
        result_states = results(state, action)
        if not result_states:
            return [state]
        # For the single returned path/cost, follow the result state nearest Goal.
        next_state = min(result_states, key=lambda pos: h(pos, stage.goal))
        child = subplans.get(next_state) if isinstance(subplans, dict) else None
        child_path = flatten_plan(next_state, child, seen | {state})
        return [state] + child_path[1:] if child_path and child_path[0] == state else [state] + child_path

    plan = or_search(stage.start, [])
    if plan is not Failure:
        path = flatten_plan(stage.start, plan, set())
        if path and path[-1] == stage.goal:
            return _finish("AND-OR Graph Search", start_time, path, stage, expanded, trace)
        if best_path and best_path[-1] == stage.goal:
            return _finish("AND-OR Graph Search", start_time, best_path, stage, expanded, trace)

    reason = "AND-OR Graph Search kh\u00f4ng t\u00ecm \u0111\u01b0\u1ee3c conditional plan ho\u00e0n ch\u1ec9nh."
    if timed_out:
        reason = "AND-OR Graph Search d\u1eebng v\u00ec v\u01b0\u1ee3t gi\u1edbi h\u1ea1n node/time \u0111\u1ec3 tr\u00e1nh treo UI."
    return _stopped("AND-OR Graph Search", start_time, stage, best_path, expanded, trace, reason)
