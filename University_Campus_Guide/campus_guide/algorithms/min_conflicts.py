from __future__ import annotations

import random

from map_data import Stage
from algorithmsUI.csp_common import (
    Assignment,
    CSPValue,
    Variable,
    _adjacent,
    _csp_domains,
    _csp_route_path,
    _csp_variables,
)

from .common import PureResult


def min_conflicts(stage: Stage, max_steps: int = 240, seed: int = 13) -> PureResult:
    rng = random.Random(seed)
    variables = _csp_variables()
    domains = _csp_domains(stage)
    current: Assignment = {var: rng.choice(domains[var]) for var in variables}

    def conflicts_for_value(var: Variable, value: CSPValue) -> int:
        return sum(
            1
            for nb in _adjacent(var)
            if nb in current and current[nb].name == value.name
        )

    def conflicted_variables() -> list[Variable]:
        return [
            var
            for var in variables
            if conflicts_for_value(var, current[var]) > 0
        ]

    def is_solution(conflicted: list[Variable]) -> bool:
        return not conflicted

    def value_with_minimum_conflicts(var: Variable) -> CSPValue:
        best_conflict_count = min(conflicts_for_value(var, value) for value in domains[var])
        best_values = [
            value
            for value in domains[var]
            if conflicts_for_value(var, value) == best_conflict_count
        ]
        return rng.choice(best_values)

    best_assignment = dict(current)
    best_conflicts = len(conflicted_variables())

    for step in range(max_steps + 1):
        conflicted = conflicted_variables()
        if is_solution(conflicted):
            return PureResult(_csp_route_path(stage, current), step, current, success=True)

        if len(conflicted) < best_conflicts:
            best_conflicts = len(conflicted)
            best_assignment = dict(current)

        var = rng.choice(conflicted)
        current[var] = value_with_minimum_conflicts(var)

    return PureResult(_csp_route_path(stage, best_assignment), max_steps, best_assignment, success=False)
