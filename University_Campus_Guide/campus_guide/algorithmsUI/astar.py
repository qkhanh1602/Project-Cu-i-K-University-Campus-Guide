from __future__ import annotations

import heapq
import time
from typing import Dict, Optional, Set, Tuple

from map_data import GridPos, Stage, movement_cost, neighbors

from .common import (
    NeighborInfo,
    SearchResult,
    action_name,
    finish_search as _finish,
    frontier_from_heap as _frontier_from_heap,
    h as _h,
    reconstruct,
)
from .trace_recorder import TraceRecorder


def astar(stage: Stage) -> SearchResult:
    name = "A*"
    mode = "normal"
    start_time = time.perf_counter()
    start, goal = stage.start, stage.goal

    frontier: list[Tuple[float, int, GridPos]] = [(float(_h(start, goal)), 0, start)]
    parent: Dict[GridPos, Optional[GridPos]] = {start: None}
    g_score: Dict[GridPos, float] = {start: 0.0}
    best_f: Dict[GridPos, float] = {start: float(_h(start, goal))}
    reached: Set[GridPos] = set()
    recorder = TraceRecorder(name, limit=1500)
    expanded = 0
    order = 1

    while frontier:
        f, _, current = heapq.heappop(frontier)
        if f != best_f.get(current):
            continue
        best_f.pop(current, None)
        if current in reached:
            continue

        expanded += 1
        gcur = g_score[current]
        infos: list[NeighborInfo] = []

        if current == goal:
            path = reconstruct(parent, goal)
            recorder.add(
                current,
                _frontier_from_heap(frontier),
                list(reached),
                infos,
                gcur,
                _h(current, goal),
                f"{name}: pop node co f(n)=g(n)+h(n) nho nhat.",
                iteration=expanded,
            )
            return _finish(name, start_time, path, stage, expanded, recorder.steps, mode)

        reached.add(current)
        for nb in neighbors(current, stage):
            new_g = gcur + movement_cost(nb, stage, mode)
            hn = _h(nb, goal)
            new_f = new_g + hn

            if nb in reached and new_g >= g_score.get(nb, float("inf")):
                infos.append(NeighborInfo(nb, action_name(current, nb), f"g={new_g:.1f}, h={hn}, f={new_f:.1f}", "SKIP", "Reached da co g tot hon."))
                continue

            if new_g < g_score.get(nb, float("inf")):
                g_score[nb] = new_g
                parent[nb] = current
                best_f[nb] = new_f
                heapq.heappush(frontier, (new_f, order, nb))
                order += 1
                infos.append(NeighborInfo(nb, action_name(current, nb), f"g={new_g:.1f}, h={hn}, f={new_f:.1f}", "ADD/UPDATE", "Cap nhat frontier theo f(n)=g(n)+h(n)."))
            else:
                infos.append(NeighborInfo(nb, action_name(current, nb), f"g={new_g:.1f}, h={hn}, f={new_f:.1f}", "SKIP", "Frontier da co f/g tot hon."))

        if recorder.can_record():
            recorder.add(
                current,
                _frontier_from_heap(frontier),
                list(reached),
                infos,
                gcur,
                _h(current, goal),
                f"{name}: frontier la Priority Queue theo f(n).",
                iteration=expanded,
            )

    return _finish(name, start_time, [], stage, expanded, recorder.steps, mode)
