from __future__ import annotations

import time

from map_data import GridPos, Stage, movement_cost, neighbors

from .common import NeighborInfo, SearchResult, action_name, finish_search as _finish, h as _h, stopped_search as _stopped
from .trace_recorder import TraceRecorder


def ida_star(stage: Stage) -> SearchResult:
    name = "IDA*"
    start_time = time.perf_counter()
    start, goal = stage.start, stage.goal
    threshold = float(_h(start, goal))
    recorder = TraceRecorder(name, limit=1500)
    total = 0
    found_path: list[GridPos] = []
    best_path: list[GridPos] = [start]
    best_h = _h(start, goal)
    max_nodes = 30000

    def add_iteration_header(round_idx: int, th: float) -> None:
        recorder.add(
            start,
            [],
            [start],
            [],
            0.0,
            _h(start, goal),
            f"IDA* ITERATION {round_idx}\nthreshold = {th:.1f}\nDFS bat dau lai tu START. Moi node dung f(n)=g(n)+h(n); neu f(n) > threshold thi cutoff.",
        )

    def dfs_limit(path: list[GridPos], g: float, th: float) -> float | str:
        nonlocal total, found_path, best_path, best_h

        current = path[-1]
        hn = _h(current, goal)
        fn = g + hn
        total += 1

        if total > max_nodes:
            return float("inf")

        if hn < best_h:
            best_h = hn
            best_path = path[:]

        if fn > th:
            if recorder.can_record():
                recorder.add(
                    current,
                    [],
                    path[-30:],
                    [],
                    g,
                    hn,
                    f"Visit {current}: g={g:.1f}, h={hn}, f={fn:.1f} > threshold={th:.1f} -> CUTOFF. Candidate next threshold = {fn:.1f}.",
                )
            return fn

        if current == goal:
            found_path = path[:]
            recorder.add(
                current,
                [],
                path[-30:],
                [],
                g,
                hn,
                f"Visit GOAL {current}: g={g:.1f}, h=0, f={fn:.1f} <= threshold={th:.1f}. IDA* hoan thanh.",
            )
            return "FOUND"

        min_over = float("inf")
        allowed: list[tuple[GridPos, float]] = []
        infos: list[NeighborInfo] = []
        path_set = set(path)

        for nb in neighbors(current, stage):
            ng = g + float(movement_cost(nb, stage, "normal"))
            nh = _h(nb, goal)
            nf = ng + nh
            metric = f"g={ng:.1f}, h={nh}, f={nf:.1f}, threshold={th:.1f}"
            if nb in path_set:
                infos.append(NeighborInfo(nb, action_name(current, nb), metric, "SKIP", "Tranh cycle trong path DFS hien tai."))
            elif nf > th:
                min_over = min(min_over, nf)
                infos.append(NeighborInfo(nb, action_name(current, nb), metric, "CUT", "f(child) vuot threshold nen chua mo rong o iteration nay."))
            else:
                allowed.append((nb, ng))
                infos.append(NeighborInfo(nb, action_name(current, nb), metric, "ADD", "f(child) <= threshold nen DFS tiep."))

        if recorder.can_record():
            recorder.add(
                current,
                [nb for nb, _ in allowed],
                path[-30:],
                infos,
                g,
                hn,
                f"Visit {current}: g={g:.1f}, h={hn}, f={fn:.1f} <= threshold={th:.1f} -> expand. IDA* khong tang depth nhu IDS.",
            )

        allowed.sort(key=lambda item: item[1] + _h(item[0], goal))
        for nb, ng in allowed:
            result = dfs_limit(path + [nb], ng, th)
            if result == "FOUND":
                return "FOUND"
            if isinstance(result, (int, float)):
                min_over = min(min_over, result)
        return min_over

    for round_idx in range(1, 80):
        add_iteration_header(round_idx, threshold)
        result = dfs_limit([start], 0.0, threshold)
        if result == "FOUND":
            return _finish(name, start_time, found_path, stage, total, recorder.steps)
        if result == float("inf"):
            break
        recorder.add(
            start,
            [],
            [start],
            [],
            0.0,
            _h(start, goal),
            f"End iteration {round_idx}: GOAL chua tim thay. next threshold = min f(n) vuot nguong = {float(result):.1f}.",
        )
        threshold = float(result)

    return _stopped(name, start_time, stage, best_path, total, recorder.steps, "IDA* dung vi het threshold hoac vuot gioi han mo rong nhung chua den Goal.")
