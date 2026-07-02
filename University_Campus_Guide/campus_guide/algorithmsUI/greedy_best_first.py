from __future__ import annotations

import heapq
import time
from typing import Dict, Set, Tuple

from map_data import GridPos, Stage, neighbors

from .common import (
    NeighborInfo,
    SearchResult,
    action_name,
    finish_search as _finish,
    h as _h,
    path_cost,
)
from .trace_recorder import TraceRecorder


def greedy_best_first(stage: Stage) -> SearchResult:
    name = "Greedy Best First"
    start_time = time.perf_counter()
    start, goal = stage.start, stage.goal

    frontier: list[Tuple[float, int, Dict[str, object]]] = [(float(_h(start, goal)), 0, {"pos": start, "path": [start]})]
    frontier_keys: Set[GridPos] = {start}
    reached: Set[GridPos] = set()
    recorder = TraceRecorder(name, limit=1500)
    expanded = 0
    order = 1

    while frontier:
        hn_current, _, node = heapq.heappop(frontier)
        current: GridPos = node["pos"]  # type: ignore[index]
        path: list[GridPos] = node["path"]  # type: ignore[index]

        if current not in frontier_keys:
            continue
        frontier_keys.remove(current)
        if current in reached:
            continue

        expanded += 1
        infos: list[NeighborInfo] = []

        if current == goal:
            recorder.add(
                current,
                [x[2]["pos"] for x in sorted(frontier)[:16]],
                list(reached),
                infos,
                path_cost(path, stage),
                hn_current,
                "Greedy: chon node co h(n) nho nhat.",
                iteration=expanded,
            )
            return _finish(name, start_time, path, stage, expanded, recorder.steps)

        reached.add(current)
        for nb in neighbors(current, stage):
            hn = _h(nb, goal)
            if nb not in reached and nb not in frontier_keys:
                heapq.heappush(frontier, (hn, order, {"pos": nb, "path": path + [nb]}))
                frontier_keys.add(nb)
                order += 1
                infos.append(NeighborInfo(nb, action_name(current, nb), f"h={hn}", "ADD", "Them vao Priority Queue theo h(n)."))
            else:
                infos.append(NeighborInfo(nb, action_name(current, nb), f"h={hn}", "SKIP", "Da co trong reached/frontier."))

        if recorder.can_record():
            recorder.add(
                current,
                [x[2]["pos"] for x in sorted(frontier)[:16]],
                list(reached),
                infos,
                path_cost(path, stage),
                hn_current,
                "Greedy dung h(n), khong quan tam g(n) khi chon node.",
                iteration=expanded,
            )

    return _finish(name, start_time, [], stage, expanded, recorder.steps)
