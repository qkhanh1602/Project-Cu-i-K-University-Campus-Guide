from __future__ import annotations

import time
from typing import Set

from map_data import GridPos, Stage, neighbors

from .common import NeighborInfo, SearchResult, action_name, finish_search as _finish
from .trace_recorder import TraceRecorder


def _dls(stage: Stage, limit: int, base_iter: int = 0) -> tuple[str, list[GridPos], list, int]:
    start, goal = stage.start, stage.goal
    frontier = [{"pos": start, "path": [start], "depth": 0, "path_set": {start}}]
    recorder = TraceRecorder("IDS/DLS", limit=1500)
    expanded = 0
    result = "failure"

    while frontier:
        node = frontier.pop()
        current: GridPos = node["pos"]
        path: list[GridPos] = node["path"]
        depth: int = node["depth"]
        path_set: Set[GridPos] = node["path_set"]
        expanded += 1
        infos: list[NeighborInfo] = []

        if current == goal:
            recorder.add(
                current,
                [n["pos"] for n in reversed(frontier)][:16],
                path,
                infos,
                depth,
                0,
                f"DLS limit={limit}: gap Goal.",
                iteration=base_iter + expanded,
            )
            return "found", path, recorder.steps, expanded

        if depth >= limit:
            result = "cutoff"
            infos.append(NeighborInfo(current, "CUT", f"depth={depth}", "CUTOFF", "depth = limit nen khong sinh con trong vong nay."))
            if recorder.can_record():
                recorder.add(
                    current,
                    [n["pos"] for n in reversed(frontier)][:16],
                    path,
                    infos,
                    depth,
                    0,
                    f"DLS limit={limit}: cutoff.",
                    iteration=base_iter + expanded,
                )
            continue

        for nb in reversed(neighbors(current, stage)):
            if nb in path_set:
                infos.append(NeighborInfo(nb, action_name(current, nb), f"depth={depth + 1}", "SKIP", "Tranh cycle trong path hien tai."))
            else:
                frontier.append({"pos": nb, "path": path + [nb], "depth": depth + 1, "path_set": set(path_set) | {nb}})
                infos.append(NeighborInfo(nb, action_name(current, nb), f"depth={depth + 1}", "PUSH", "depth < limit nen push vao Stack."))

        if recorder.can_record():
            recorder.add(
                current,
                [n["pos"] for n in reversed(frontier)][:16],
                path,
                infos,
                depth,
                0,
                f"IDS chay DLS voi limit={limit}.",
                iteration=base_iter + expanded,
            )

    return result, [], recorder.steps, expanded


def ids(stage: Stage) -> SearchResult:
    start_time = time.perf_counter()
    all_trace = []
    total = 0

    for limit in range(0, 160):
        status, path, trace, nodes = _dls(stage, limit, len(all_trace))
        total += nodes
        all_trace.extend(trace)
        if len(all_trace) > 1500:
            all_trace = all_trace[:1500]
        if status == "found":
            return _finish("IDS", start_time, path, stage, total, all_trace)
        if status == "failure":
            break

    return _finish("IDS", start_time, [], stage, total, all_trace)
