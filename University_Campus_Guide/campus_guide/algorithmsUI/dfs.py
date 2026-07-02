from __future__ import annotations

import time
from typing import Dict, Optional, Set

from map_data import GridPos, Stage, neighbors

from .common import NeighborInfo, SearchResult, action_name, finish_search, reconstruct
from .trace_recorder import TraceRecorder


def dfs(stage: Stage) -> SearchResult:
    start_time = time.perf_counter()
    start, goal = stage.start, stage.goal

    frontier: list[GridPos] = [start]
    frontier_set: Set[GridPos] = {start}
    reached: Set[GridPos] = {start}
    parent: Dict[GridPos, Optional[GridPos]] = {start: None}
    recorder = TraceRecorder("DFS", limit=1500)
    expanded = 0

    while frontier:
        current = frontier.pop()
        frontier_set.discard(current)
        expanded += 1
        infos: list[NeighborInfo] = []

        if current == goal:
            path = reconstruct(parent, goal)
            recorder.add(
                current,
                list(reversed(frontier))[:16],
                list(reached),
                infos,
                len(path) - 1,
                0,
                "DFS: Stack LIFO, pop node hien tai roi goal-test.",
                iteration=expanded,
            )
            return finish_search("DFS", start_time, path, stage, expanded, recorder.steps)

        for nb in reversed(neighbors(current, stage)):
            if nb not in reached and nb not in frontier_set:
                reached.add(nb)
                frontier.append(nb)
                frontier_set.add(nb)
                parent[nb] = current
                infos.append(
                    NeighborInfo(
                        nb,
                        action_name(current, nb),
                        "push",
                        "PUSH",
                        "Dua vao Stack; node push sau se duoc xet truoc.",
                    )
                )
            else:
                infos.append(
                    NeighborInfo(
                        nb,
                        action_name(current, nb),
                        "",
                        "SKIP",
                        "Da co trong reached/frontier.",
                    )
                )

        if recorder.can_record():
            recorder.add(
                current,
                list(reversed(frontier))[:16],
                list(reached),
                infos,
                0,
                0,
                "DFS uu tien di sau mot nhanh truoc.",
                iteration=expanded,
            )

    return finish_search("DFS", start_time, [], stage, expanded, recorder.steps)
