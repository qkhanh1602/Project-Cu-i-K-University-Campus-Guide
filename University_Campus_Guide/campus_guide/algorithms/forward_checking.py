from __future__ import annotations

from typing import Optional

from map_data import Stage
from algorithmsUI.csp_common import (
    Assignment,
    CSPValue,
    Domains,
    Variable,
    _check_value,
    _csp_domains,
    _csp_route_path,
    _csp_variables,
    _forward_domains,
)

from .common import PureResult


def forward_checking(stage: Stage) -> PureResult:
    variables = _csp_variables()
    start_domains = _csp_domains(stage)
    expanded = 0

    def is_assignment_complete(index: int) -> bool:
        return index == len(variables)

    def is_consistent(assignment: Assignment, var: Variable, value: CSPValue) -> bool:
        ok, _reasons = _check_value(stage, assignment, var, value)
        return ok

    def remove_value_from_neighbors(assignment: Assignment, domains: Domains) -> Domains:
        next_domains, _removed = _forward_domains(stage, assignment, domains)
        return next_domains

    def some_future_variable_has_empty_domain(domains: Domains, index: int) -> bool:
        future_vars = variables[index + 1 :]
        return any(not domains.get(future) for future in future_vars)

    def search(assignment: Assignment, domains: Domains, index: int) -> Optional[Assignment]:
        nonlocal expanded
        expanded += 1

        if is_assignment_complete(index):
            return dict(assignment)

        var = variables[index]
        for value in domains[var]:
            if not is_consistent(assignment, var, value):
                continue

            next_assignment = dict(assignment)
            next_assignment[var] = value
            next_domains = remove_value_from_neighbors(next_assignment, domains)

            if some_future_variable_has_empty_domain(next_domains, index):
                continue

            result = search(next_assignment, next_domains, index + 1)
            if result is not None:
                return result

        return None

    assignment = search({}, start_domains, 0)
    if assignment is None:
        return PureResult([stage.start], expanded, success=False)
    return PureResult(_csp_route_path(stage, assignment), expanded, assignment, success=True)
