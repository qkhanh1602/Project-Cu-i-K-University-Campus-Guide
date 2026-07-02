from __future__ import annotations

from collections import deque
from dataclasses import replace
import heapq
from typing import Dict, Iterable, List, Optional, Set, Tuple

from map_data import GridPos, Stage, is_walkable, manhattan, movement_cost, neighbors

from .common import action_name, h as _h, move, reconstruct


Belief = Tuple[GridPos, ...]


def _unique_positions(items: Iterable[GridPos]) -> Belief:
    return tuple(sorted(set(items)))


def _belief_initial(stage: Stage, size: int = 3) -> Belief:
    # Khong biet start: neu chang khai bao START? thi dung dung tap do.
    if stage.uncertain_starts:
        possible = [p for p in sorted(stage.uncertain_starts) if is_walkable(p, stage)]
        if possible:
            return _unique_positions(possible)

    result: List[GridPos] = [stage.start]
    for p in neighbors(stage.start, stage):
        if len(result) >= size:
            break
        result.append(p)
    return _unique_positions(result)


def _belief_goal_set(stage: Stage, size: int = 1) -> Set[GridPos]:
    goals = [stage.goal]
    if size > 1:
        for p in neighbors(stage.goal, stage):
            if len(goals) >= size:
                break
            goals.append(p)
    return set(goals)


def _belief_h(belief: Belief, goals: Set[GridPos], aggregate: str = "MAX") -> float:
    vals = [min(_h(s, g) for g in goals) for s in belief]
    if not vals:
        return 0.0
    return sum(vals) / len(vals) if aggregate == "AVG" else float(max(vals))


def _is_goal_belief(belief: Belief, goals: Set[GridPos]) -> bool:
    return all(pos in goals for pos in belief)



def _belief_text(belief: Belief) -> str:
    return "{" + ", ".join(f"({r},{c})" for r, c in belief) + "}"


def _belief_step_cost(stage: Stage, belief: Belief, next_belief: Belief) -> float:
    # A joint action is one planning step. Terrain cost is aggregated conservatively
    # so A* still prefers safer/easier belief transitions when costs exist.
    if not next_belief:
        return 1.0
    return max(1.0, max(float(movement_cost(pos, stage)) for pos in next_belief))

def _belief_goal_progress(belief: Belief, goals: Set[GridPos]) -> int:
    return sum(1 for pos in belief if pos in goals)


def _belief_goal_status_text(belief: Belief, goals: Set[GridPos]) -> str:
    reached = _belief_goal_progress(belief, goals)
    total = len(belief)
    if reached == 0:
        return "\u0111ang t\u00ecm Goal"
    if reached == total:
        return "to\u00e0n b\u1ed9 belief \u0111\u00e3 t\u1edbi Goal"
    return f"{reached}/{total} tr\u1ea1ng th\u00e1i \u0111\u00e3 t\u1edbi Goal"


def _apply_belief_action(belief: Belief, action: str, stage: Stage, goals: Set[GridPos]) -> Belief:
    out: List[GridPos] = []
    for pos in belief:
        if pos in goals:
            out.append(pos)  # Goal hap thu: state da toi Goal thi dung yen.
            continue
        nxt = move(pos, action)
        out.append(nxt if is_walkable(nxt, stage) else pos)
    return _unique_positions(out)


def _rep_next_for_action(rep: GridPos, action: str, stage: Stage) -> Optional[GridPos]:
    nxt = move(rep, action)
    return nxt if is_walkable(nxt, stage) else None


def _belief_rep_next(rep: GridPos, action: str, next_belief: Belief, stage: Stage) -> GridPos:
    intended = _rep_next_for_action(rep, action, stage)
    if intended is not None:
        return intended

    choices = neighbors(rep, stage)
    if not choices:
        return rep

    targets = list(next_belief) or [stage.goal]
    return min(
        choices,
        key=lambda p: (
            min(manhattan(p, target) for target in targets),
            manhattan(p, stage.goal),
        ),
    )


def _shortest_path_from(start: GridPos, goal: GridPos, stage: Stage, use_cost: bool = False) -> List[GridPos]:
    if start == goal:
        return [start]

    parent: Dict[GridPos, Optional[GridPos]] = {start: None}

    if use_cost:
        frontier: List[Tuple[float, int, GridPos]] = [(float(_h(start, goal)), 0, start)]
        g_score: Dict[GridPos, float] = {start: 0.0}
        best_f: Dict[GridPos, float] = {start: float(_h(start, goal))}
        order = 1
        reached: Set[GridPos] = set()
        while frontier:
            f, _, current = heapq.heappop(frontier)
            if f != best_f.get(current):
                continue
            best_f.pop(current, None)
            if current in reached:
                continue
            if current == goal:
                return reconstruct(parent, goal)
            reached.add(current)
            for nb in neighbors(current, stage):
                new_g = g_score[current] + movement_cost(nb, stage)
                new_f = new_g + _h(nb, goal)
                if nb in reached and new_g >= g_score.get(nb, float("inf")):
                    continue
                if new_g < g_score.get(nb, float("inf")):
                    g_score[nb] = new_g
                    parent[nb] = current
                    best_f[nb] = new_f
                    heapq.heappush(frontier, (new_f, order, nb))
                    order += 1
        return []

    frontier = deque([start])
    reached = {start}
    while frontier:
        current = frontier.popleft()
        if current == goal:
            return reconstruct(parent, goal)
        for nb in neighbors(current, stage):
            if nb in reached:
                continue
            reached.add(nb)
            parent[nb] = current
            frontier.append(nb)
    return []


def _parallel_belief_paths(stage: Stage, use_cost: bool = False) -> List[List[GridPos]]:
    starts = list(_belief_initial(stage))
    relaxed_stage = replace(stage, blocked=set())
    paths: List[List[GridPos]] = []
    for start in starts:
        path = _shortest_path_from(start, stage.goal, stage, use_cost)
        if not path:
            path = _shortest_path_from(start, stage.goal, relaxed_stage, use_cost)
        if path:
            paths.append(path)
    return paths


def _belief_fallback_paths(stage: Stage, use_cost: bool = False) -> List[List[GridPos]]:
    """Return one safe route per uncertain start when no shared plan exists."""

    paths = _parallel_belief_paths(stage, use_cost=use_cost)
    return [path for path in paths if path and path[-1] == stage.goal]


def _belief_demo_result_path(stage: Stage, use_cost: bool = False) -> List[GridPos]:
    """A valid stage.start -> goal path used for result metrics in belief demos."""

    path = _shortest_path_from(stage.start, stage.goal, stage, use_cost)
    if path:
        return path
    relaxed_stage = replace(stage, blocked=set())
    path = _shortest_path_from(stage.start, stage.goal, relaxed_stage, use_cost)
    return path or [stage.start]


def _belief_at(paths: List[List[GridPos]], index: int) -> Belief:
    cells = [path[min(index, len(path) - 1)] for path in paths if path]
    return _unique_positions(cells)


def _belief_move_infos(paths: List[List[GridPos]], index: int):
    from .common import NeighborInfo

    infos: List[NeighborInfo] = []
    for i, path in enumerate(paths, start=1):
        prev = path[min(max(index - 1, 0), len(path) - 1)]
        cur = path[min(index, len(path) - 1)]
        status = "GOAL" if cur == path[-1] else "MOVE"
        if cur == prev:
            status = "WAIT" if cur != path[-1] else "GOAL"
        value = f"agent={i}"
        reason = (
            "Nh\u00e2n v\u1eadt belief \u0111i theo \u0111\u01b0\u1eddng ri\u00eang t\u1edbi Goal."
            if status != "GOAL"
            else "Nh\u00e2n v\u1eadt belief \u0111\u00e3 t\u1edbi Goal v\u00e0 \u0111\u1ee9ng l\u1ea1i."
        )
        infos.append(NeighborInfo(cur, action_name(prev, cur), value, status, reason))
    return infos


def _belief_frontier_view(items: Iterable[Belief]) -> List[GridPos]:
    out: List[GridPos] = []
    for b in items:
        for pos in b:
            if pos not in out:
                out.append(pos)
            if len(out) >= 16:
                return out
    return out[:16]


def _belief_trace_note(name: str, aggregate: str, goal_size: int) -> str:
    extra = "Goal set c\u00f3 nhi\u1ec1u kh\u1ea3 n\u0103ng." if goal_size > 1 else "Goal l\u00e0 tr\u1ea1ng th\u00e1i h\u1ea5p th\u1ee5."
    return (
        f"{name}: bi\u1ebft Goal nh\u01b0ng kh\u00f4ng bi\u1ebft ch\u1eafc Start; "
        f"node l\u00e0 belief state = t\u1eadp START? / v\u1ecb tr\u00ed c\u00f3 th\u1ec3. "
        f"h(B) d\u00f9ng {aggregate}. {extra} "
        "Thu\u1eadt to\u00e1n ch\u1ec9 ho\u00e0n th\u00e0nh khi m\u1ecdi tr\u1ea1ng th\u00e1i trong belief \u0111\u1ec1u \u0111\u1ea1t Goal."
    )


def _belief_note(name: str, aggregate: str, goal_size: int, belief: Belief) -> str:
    cells = ";".join(f"({r},{c})" for r, c in belief)
    return _belief_trace_note(name, aggregate, goal_size) + f" CURRENT_BELIEF={cells}"


def _belief_paths_from_actions(stage: Stage, actions: List[str], goals: Set[GridPos]) -> List[List[GridPos]]:
    paths: List[List[GridPos]] = [[start] for start in _belief_initial(stage)]
    for action in actions:
        for path in paths:
            pos = path[-1]
            if pos in goals:
                path.append(pos)
                continue
            nxt = move(pos, action)
            path.append(nxt if is_walkable(nxt, stage) else pos)
    return paths


def _representative_path_to_goal(rep_path: List[GridPos], stage: Stage, use_cost: bool = False) -> List[GridPos]:
    if not rep_path:
        rep_path = [stage.start]
    if rep_path[-1] == stage.goal:
        return rep_path
    tail = _shortest_path_from(rep_path[-1], stage.goal, stage, use_cost)
    if tail:
        return rep_path + tail[1:]
    return rep_path
