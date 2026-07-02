from __future__ import annotations

import math
import random

from map_data import Stage, neighbors

from .common import PureResult, heuristic


INITIAL_TEMPERATURE = 10.0
ALPHA = 0.86
MIN_TEMPERATURE = 0.05
MAX_STEPS = 220


def simulated_annealing(stage: Stage, max_steps: int = MAX_STEPS, seed: int = 7) -> PureResult:
    rng = random.Random(seed)
    current_state = stage.start
    path = [current_state]
    temperature = INITIAL_TEMPERATURE
    expanded = 0

    for step in range(1, max_steps + 1):
        expanded = step
        if current_state == stage.goal:
            return PureResult(path, expanded, success=True)
        if temperature <= MIN_TEMPERATURE:
            break

        choices = neighbors(current_state, stage)
        if not choices:
            break

        next_state = rng.choice(choices)
        delta = heuristic(next_state, stage.goal) - heuristic(current_state, stage.goal)

        if delta < 0 or rng.random() < math.exp(-delta / temperature):
            current_state = next_state
            path.append(current_state)

        temperature *= ALPHA

    return PureResult(path, expanded, success=(current_state == stage.goal))
