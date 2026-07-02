from __future__ import annotations

from collections import deque
import time
from typing import Set

from map_data import GridPos, Stage, neighbors

from .common import NeighborInfo, SearchResult, action_name
from .trace_recorder import TraceRecorder


def bfs(stage: Stage) -> SearchResult:
    start_time = time.perf_counter()
    start, goal = stage.start, stage.goal

    frontier = deque([{"pos": start, "path": [start]}])
    reached: Set[GridPos] = {start}
    recorder = TraceRecorder("BFS", limit=1500)
    expanded = 0

    while frontier:
        node = frontier.popleft()
        current: GridPos = node["pos"]
        path: list[GridPos] = node["path"]
        expanded += 1

        infos: list[NeighborInfo] = []
        current_steps = len(path) - 1

        if current == goal:
            recorder.add(
                current,
                [n["pos"] for n in list(frontier)[:16]],
                list(reached),
                infos,
                current_steps,
                0,
                "BFS: Queue FIFO, duyet theo chieu rong. Cost = so buoc di chuyen.",
                iteration=expanded,
            )
            runtime_ms = (time.perf_counter() - start_time) * 1000
            return SearchResult(
                "BFS",
                path,
                int(current_steps),
                expanded,
                recorder.steps,
                round(runtime_ms, 3),
                "Hoan thanh",
                False,
            )

        for nb in neighbors(current, stage):
            depth_child = len(path)

            if nb not in reached:
                reached.add(nb)
                frontier.append({"pos": nb, "path": path + [nb]})
                infos.append(
                    NeighborInfo(
                        nb,
                        action_name(current, nb),
                        f"step={depth_child}",
                        "ADD",
                        "Chua co trong reached nen them vao cuoi Queue FIFO. Cost child = cost parent + 1.",
                    )
                )
            else:
                infos.append(
                    NeighborInfo(
                        nb,
                        action_name(current, nb),
                        "",
                        "SKIP",
                        "Da co trong reached nen bo qua.",
                    )
                )

        if recorder.can_record():
            recorder.add(
                current,
                [n["pos"] for n in list(frontier)[:16]],
                list(reached),
                infos,
                current_steps,
                0,
                "BFS khong dung cost dia hinh va khong dung h(n). Cost trong mo phong = so buoc di chuyen.",
                iteration=expanded,
            )

    runtime_ms = (time.perf_counter() - start_time) * 1000
    return SearchResult(
        "BFS",
        [],
        0,
        expanded,
        recorder.steps,
        round(runtime_ms, 3),
        "Dung - khong tim thay Goal",
        False,
    )
