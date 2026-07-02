from __future__ import annotations

import math
import random
import time

from map_data import Stage, neighbors

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


INITIAL_TEMPERATURE = 10.0
ALPHA = 0.86
MIN_TEMPERATURE = 0.05
MAX_STEPS = 220


def simulated_annealing(stage: Stage, seed: int = 7) -> SearchResult:
    name = "Simulated Annealing"
    start_time = time.perf_counter()
    random.seed(seed)
    current = stage.start
    path = [current]
    recorder = TraceRecorder(name)
    expanded = 0
    temperature = INITIAL_TEMPERATURE

    while temperature > MIN_TEMPERATURE and expanded < MAX_STEPS:
        expanded += 1
        cur_h = _h(current, stage.goal)

        if current == stage.goal:
            return _finish(name, start_time, path, stage, expanded, recorder.steps)

        candidates = neighbors(current, stage)
        if not candidates:
            break

        nb = random.choice(candidates)
        next_h = _h(nb, stage.goal)
        delta = next_h - cur_h
        probability = 1.0 if delta < 0 else math.exp(-delta / max(temperature, 1e-9))
        random_value = random.random()
        accept = delta < 0 or random_value < probability
        status = "ACCEPT" if accept else "REJECT"
        reason = "Tot hon nen nhan chac chan." if delta < 0 else f"Buoc xau/ngang: p=exp(-delta/T)={probability:.3f}, random={random_value:.3f}."
        infos = [NeighborInfo(nb, action_name(current, nb), f"h_next={next_h}, delta={delta}, T={temperature:.2f}", status, reason)]

        recorder.add(
            current,
            [nb],
            path[-30:],
            infos,
            path_cost(path, stage),
            cur_h,
            f"SA: T0={INITIAL_TEMPERATURE}, alpha={ALPHA}, Tmin={MIN_TEMPERATURE}, max_steps={MAX_STEPS}. Moi vong random mot neighbor; tot hon thi nhan, xau hon thi nhan theo xac suat.",
            iteration=expanded,
        )

        if accept:
            current = nb
            path.append(current)

        temperature *= ALPHA

    return _stopped(name, start_time, stage, path, expanded, recorder.steps, "T giam duoi Tmin hoac het buoc nhung chua gap Goal.")
