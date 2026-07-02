from __future__ import annotations

from typing import Optional

from map_data import Stage
from algorithmsUI.csp_common import (
    Assignment,
    CSPValue,
    Variable,
    _check_value,
    _csp_domains,
    _csp_route_path,
    _csp_variables,
)

from .common import PureResult


def backtracking(stage: Stage) -> PureResult:
    variables = _csp_variables()
    domains = _csp_domains(stage)
    expanded = 0

    def is_assignment_complete(index: int) -> bool:
        return index == len(variables)

    def is_consistent(assignment: Assignment, var: Variable, value: CSPValue) -> bool:
        ok, _reasons = _check_value(stage, assignment, var, value)
        return ok

    def search(assignment: Assignment, index: int) -> Optional[Assignment]:
        nonlocal expanded
        expanded += 1

        if is_assignment_complete(index):
            return dict(assignment)

        var = variables[index]
        for value in domains[var]:
            if not is_consistent(assignment, var, value):
                continue

            assignment[var] = value
            result = search(assignment, index + 1)
            if result is not None:
                return result

            assignment.pop(var)

        return None

    assignment = search({}, 0)
    if assignment is None:
        return PureResult([stage.start], expanded, success=False)
    return PureResult(_csp_route_path(stage, assignment), expanded, assignment, success=True)
