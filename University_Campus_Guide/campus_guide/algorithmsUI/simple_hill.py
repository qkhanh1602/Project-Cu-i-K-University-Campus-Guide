from __future__ import annotations

import time
from typing import Optional

from map_data import GridPos, Stage, neighbors

from .common import (
    NeighborInfo,
    SearchResult,
    action_name,
    finish_search as _finish,
    h as _h,
    path_cost,
    stopped_search as _stopped,
)
from .trace_recorder import TraceRecorder


def _local_value(pos: GridPos, stage: Stage) -> float:
    return float(_h(pos, stage.goal))


def simple_hill(stage: Stage) -> SearchResult:
    name = "Simple Hill Climbing"
    start_time = time.perf_counter()
    current = stage.start
    path = [current]
    recorder = TraceRecorder(name)
    expanded = 0

    while expanded < 160:
        expanded += 1
        cur_val = _local_value(current, stage)
        infos: list[NeighborInfo] = []

        if current == stage.goal:
            return _finish(name, start_time, path, stage, expanded, recorder.steps)

        chosen: Optional[GridPos] = None
        for nb in neighbors(current, stage):
            val = _local_value(nb, stage)
            duplicate_note = " Duplicate state, still evaluated." if nb in path else ""
            if chosen is None and val < cur_val:
                chosen = nb
                infos.append(NeighborInfo(nb, action_name(current, nb), f"h={val:.1f} < {cur_val:.1f}", "CHOSEN", "Chon ngay neighbor dau tien tot hon current." + duplicate_note))
            elif val < cur_val:
                infos.append(NeighborInfo(nb, action_name(current, nb), f"h={val:.1f} < {cur_val:.1f}", "EVALUATED", "Tot hon current nhung khong chon vi Simple Hill lay first better neighbor." + duplicate_note))
            else:
                infos.append(NeighborInfo(nb, action_name(current, nb), f"h={val:.1f}", "REJECT", "Khong cai thien so voi current." + duplicate_note))

        recorder.add(
            current,
            [chosen] if chosen else [],
            path[-30:],
            infos,
            path_cost(path, stage),
            cur_val,
            "Simple Hill Climbing: sinh tat ca neighbor hop le va chi so sanh h(n); duplicate van duoc tinh h(n), max_steps dung lam gioi han an toan.",
            iteration=expanded,
        )

        if chosen is None:
            return _stopped(name, start_time, stage, path, expanded, recorder.steps, "Ket local optimum: khong co neighbor tot hon.")

        current = chosen
        path.append(current)

    return _stopped(name, start_time, stage, path, expanded, recorder.steps, "Vuot qua so buoc local search.")
