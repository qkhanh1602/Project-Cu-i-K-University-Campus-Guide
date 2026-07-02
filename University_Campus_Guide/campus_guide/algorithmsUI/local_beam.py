from __future__ import annotations

import time
from typing import Dict

from map_data import GridPos, Stage, neighbors

from .common import NeighborInfo, SearchResult, action_name, finish_search as _finish, h as _h, stopped_search as _stopped
from .trace_recorder import TraceRecorder


def local_beam(stage: Stage, k: int = 2) -> SearchResult:
    name = "Local Beam Search"
    start_time = time.perf_counter()
    start, goal = stage.start, stage.goal
    recorder = TraceRecorder(name)
    expanded = 0

    def make_item(pos: GridPos, path: list[GridPos]) -> Dict[str, object]:
        return {"pos": pos, "path": path, "h": _h(pos, goal)}

    init_candidates = [make_item(nb, [start, nb]) for nb in neighbors(start, stage)]
    init_candidates.sort(key=lambda item: (item["h"], item["pos"]))  # type: ignore[index]
    current_set = init_candidates[:k]
    selected_ids = {id(item) for item in current_set}
    init_infos: list[NeighborInfo] = []

    for item in init_candidates:
        pos: GridPos = item["pos"]  # type: ignore[assignment]
        hn = int(item["h"])  # type: ignore[arg-type]
        status = "CHOSEN" if id(item) in selected_ids else "CANDIDATE"
        reason = f"Khoi tao beam: chon k={k} node co h(n) nho nhat tu hang xom START." if status == "CHOSEN" else "Khong thuoc 2 node tot nhat theo h(n)."
        init_infos.append(NeighborInfo(pos, f"START->{action_name(start, pos)}", f"h={hn}", status, reason))

    recorder.add(
        start,
        [item["pos"] for item in current_set],
        [start],
        init_infos,
        0.0,
        _h(start, goal),
        f"LOCAL BEAM INITIALIZATION\nSTART = {start}\nGOAL = {goal}\nk = {k}\nGenerate neighbors from START, tinh h(n)=Manhattan toi GOAL, roi chon initial beam k=2 nho nhat. Local Beam chi dung h(n), khong dung g(n) hoac f(n).",
        beam_paths=[list(item["path"]) for item in current_set],
    )

    if not current_set:
        return _stopped(name, start_time, stage, [start], expanded, recorder.steps, "START khong co hang xom hop le cho Local Beam.")

    best = min(current_set, key=lambda item: item["h"])  # type: ignore[index]

    for round_idx in range(1, 90):
        all_candidates: list[Dict[str, object]] = []
        current_positions = [item["pos"] for item in current_set]  # type: ignore[list-item]

        for item in current_set:
            pos: GridPos = item["pos"]  # type: ignore[assignment]
            path: list[GridPos] = item["path"]  # type: ignore[assignment]
            expanded += 1

            if pos == goal:
                return _finish(name, start_time, path, stage, expanded, recorder.steps)

            for nb in neighbors(pos, stage):
                action = f"{pos}->{action_name(pos, nb)}"
                candidate = make_item(nb, path + [nb])
                candidate["action"] = action
                candidate["duplicate"] = nb in path
                all_candidates.append(candidate)

        if not all_candidates:
            return _stopped(name, start_time, stage, best["path"], expanded, recorder.steps, "Khong con candidate hop le, tra ve path tot nhat da gap.")  # type: ignore[arg-type]

        infos: list[NeighborInfo] = []

        for candidate in all_candidates:
            if candidate["pos"] == goal:
                path = candidate["path"]  # type: ignore[assignment]
                infos = [
                    NeighborInfo(
                        item["pos"],  # type: ignore[arg-type]
                        str(item.get("action", "")),
                        f"h={item['h']}",
                        "CANDIDATE",
                        "Candidate sinh tu current beam; duplicate van duoc danh gia." if item.get("duplicate") else "Candidate sinh tu current beam.",
                    )
                    for item in all_candidates
                ]
                recorder.add(
                    goal,
                    current_positions,
                    path[-30:],
                    infos,
                    len(path) - 1,
                    0,
                    "BEAM ITERATION: gap GOAL trong candidates nen dung. Candidates duoc gom toan cuc tu tat ca beam nodes.",
                    beam_paths=[list(item["path"]) for item in current_set] + [list(path)],
                )
                return _finish(name, start_time, path, stage, expanded, recorder.steps)

        all_candidates.sort(key=lambda item: (item["h"], item["pos"]))  # type: ignore[index]
        current_set = all_candidates[:k]
        if current_set[0]["h"] < best["h"]:  # type: ignore[index]
            best = current_set[0]

        chosen_ids = {id(item) for item in current_set}
        for candidate in all_candidates:
            pos: GridPos = candidate["pos"]  # type: ignore[assignment]
            hn = candidate["h"]
            status = "CHOSEN" if id(candidate) in chosen_ids else "CANDIDATE"
            reason = (
                f"Giu lai trong next beam k={k}: mot trong 2 candidate co h(n) nho nhat toan cuc."
                if status == "CHOSEN"
                else "Khong thuoc k candidate tot nhat toan cuc theo h(n)."
            )
            if candidate.get("duplicate"):
                reason += " Duplicate state, still evaluated."
            infos.append(NeighborInfo(pos, str(candidate.get("action", "")), f"h={hn}", status, reason))

        note = (
            f"BEAM ITERATION {round_idx}\nCurrent beam:\n"
            + "\n".join(f"- {item['pos']}, h={item['h']}" for item in current_set)
            + "\nGenerated candidates tu tat ca beam nodes, giu ca duplicate/path rieng, sap xep theo h(n), chon k=2 tot nhat toan cuc."
        )
        recorder.add(
            current_set[0]["pos"],  # type: ignore[arg-type]
            [item["pos"] for item in current_set],
            current_positions,
            infos,
            0.0,
            current_set[0]["h"],  # type: ignore[arg-type]
            note,
            beam_paths=[list(item["path"]) for item in current_set],
        )

    return _stopped(name, start_time, stage, best["path"], expanded, recorder.steps, "Het so vong Local Beam.")  # type: ignore[arg-type]
