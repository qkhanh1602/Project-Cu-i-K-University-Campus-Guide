from __future__ import annotations

import random
import time
from typing import List, Tuple

from map_data import Stage

from .common import NeighborInfo, SearchResult, finish_search as _finish, path_cost, stopped_search as _stopped
from .csp_common import (
    Assignment,
    CSPValue,
    Variable,
    _adjacent,
    _assignment_text,
    _building_label,
    _csp_domains,
    _csp_route_path,
    _csp_variables,
    _value_text,
)
from .trace_recorder import TraceRecorder


def min_conflicts_csp(stage: Stage, max_steps: int = 240, seed: int = 13) -> SearchResult:
    name = "Min-Conflicts"
    start_time = time.perf_counter()
    rng = random.Random(seed)
    variables = _csp_variables()
    domains = _csp_domains(stage)
    recorder = TraceRecorder(name)
    trace = recorder.steps
    expanded = 0

    def adjacency_text() -> str:
        lines = ["ADJACENCY LIST"]
        for var in variables:
            nbs = [_building_label(nb) for nb in _adjacent(var)]
            lines.append(f"- {_building_label(var)} ke: {', '.join(nbs) if nbs else 'khong ke ai'}")
        return "\n".join(lines)

    def representation(assignment: Assignment) -> str:
        return (
            "CSP REPRESENTATION\n"
            + "- Variables: " + ", ".join(_building_label(v) for v in variables) + "\n"
            + "- DOMAIN = { Cam, Hong, Xanh, Tim }\n"
            + "- CONSTRAINTS:\n"
            + "- Hai toa nha ke nhau khong duoc cung mau.\n"
            + adjacency_text() + "\n"
            + "CURRENT ASSIGNMENT / DOMAINS\n"
            + "Assignment hien tai: building = color; building chua chon thi = chua to\n"
            + _assignment_text(assignment) + "\n"
            + "SOLVING STEPS\n"
        )

    def total_conflicts(assignment: Assignment) -> Tuple[int, List[str]]:
        reasons: List[str] = []
        for var, nb in conflict_edges(assignment):
            value = assignment[var]
            reasons.append(f"{_building_label(var)} va {_building_label(nb)} ke nhau nhung cung mau {value.label()}")
        return len(reasons), reasons

    def conflict_edges(assignment: Assignment) -> List[Tuple[Variable, Variable]]:
        edges: List[Tuple[Variable, Variable]] = []
        seen: set[Tuple[Variable, Variable]] = set()
        for var in variables:
            value = assignment.get(var)
            if not value:
                continue
            for nb in _adjacent(var):
                edge = tuple(sorted((var, nb)))
                if edge in seen:
                    continue
                seen.add(edge)
                other = assignment.get(nb)
                if other and other.name == value.name:
                    edges.append((var, nb))
        return edges

    def conflicts_for_value(var: Variable, value: CSPValue, assignment: Assignment) -> int:
        return sum(
            1
            for nb in _adjacent(var)
            if (other := assignment.get(nb)) is not None and other.name == value.name
        )

    def conflicted_variables(assignment: Assignment) -> List[Variable]:
        return [
            var
            for var in variables
            if var in assignment and conflicts_for_value(var, assignment[var], assignment) > 0
        ]

    def candidate_rows(var: Variable, assignment: Assignment) -> List[Tuple[int, int, CSPValue]]:
        rows: List[Tuple[int, int, CSPValue]] = []
        for order, value in enumerate(domains[var]):
            rows.append((conflicts_for_value(var, value, assignment), order, value))
        rows.sort(key=lambda item: (item[0], item[1]))
        return rows

    def neighbor_report(var: Variable, value: CSPValue, assignment: Assignment) -> str:
        lines = [f"Neighbors cua {_building_label(var)}:"]
        nbs = _adjacent(var)
        if not nbs:
            lines.append("- Khong ke toa nao.")
            return "\n".join(lines)
        for nb in nbs:
            other = assignment.get(nb)
            if not other:
                lines.append(f"- {_building_label(nb)} = chua to -> OK")
            else:
                status = "CONFLICT" if other.name == value.name else "OK"
                lines.append(f"- {_building_label(var)} = {value.label()}, {_building_label(nb)} = {other.label()} -> {status}")
        return "\n".join(lines)

    def add_trace(var: Variable | None, assignment: Assignment, infos: List[NeighborInfo], message: str, conflicts: int, extra: str = "") -> None:
        route = _csp_route_path(stage, assignment) if assignment else [stage.start]
        current = assignment[var].pos if var and var in assignment else stage.start
        note = representation(assignment) + message
        if extra:
            note += "\n" + extra
        recorder.add(
            current,
            [info.node for info in infos if info.status in {"TRY", "SELECTED", "CONFLICT"}],
            [assignment[k].pos for k in variables if k in assignment],
            infos,
            path_cost(route, stage, "normal") if route else 0.0,
            float(conflicts),
            note,
        )

    add_trace(
        None,
        {},
        [],
        "MIN-CONFLICTS STEP 0\nCac toa nha dang chua to mau. Buoc sau Min-Conflicts tao complete assignment ban dau.",
        0,
    )

    assignment: Assignment = {var: rng.choice(domains[var]) for var in variables}
    conflicts, reasons = total_conflicts(assignment)
    best_assignment = dict(assignment)
    best_conflicts = conflicts
    init_infos = [
        NeighborInfo(
            assignment[var].pos,
            f"{_building_label(var)} = {assignment[var].label()}",
            "initial",
            "CONFLICT" if var in conflicted_variables(assignment) else "OK",
            "Complete assignment ban dau cua Min-Conflicts.",
        )
        for var in variables
    ]
    add_trace(
        variables[0],
        assignment,
        init_infos,
        "MIN-CONFLICTS INITIALIZATION\nKhoi tao assignment day du cho tat ca toa nha roi kiem tra cac cap ke nhau.",
        conflicts,
        f"So conflict ban dau = {conflicts}." + ("\n" + "\n".join(reasons[:8]) if reasons else ""),
    )

    for step_no in range(1, max_steps + 1):
        expanded += 1
        conflicts, reasons = total_conflicts(assignment)
        if conflicts < best_conflicts:
            best_conflicts = conflicts
            best_assignment = dict(assignment)

        if conflicts == 0:
            final_path = _csp_route_path(stage, assignment)
            infos = [
                NeighborInfo(value.pos, f"{_building_label(var)} = {value.label()}", _value_text(value), "OK", "Khong con conflict voi toa ke.")
                for var, value in assignment.items()
            ]
            add_trace(
                variables[-1],
                assignment,
                infos,
                f"MIN-CONFLICTS STEP {step_no}\nKhong con bien xung dot. Thuat toan hoan thanh.",
                0,
            )
            return _finish(name, start_time, final_path, stage, expanded, trace, "normal")

        conflicted = conflicted_variables(assignment)
        var = rng.choice(conflicted)
        related_edges = [edge for edge in conflict_edges(assignment) if var in edge]
        old_value = assignment[var]
        rows = candidate_rows(var, assignment)
        min_count = rows[0][0]
        best_count, best_order, best_value = rng.choice([row for row in rows if row[0] == min_count])

        conflict_nodes: List[Variable] = [var]
        for a, b in related_edges:
            other = b if a == var else a
            if other not in conflict_nodes:
                conflict_nodes.append(other)

        infos: List[NeighborInfo] = []
        for node_var in conflict_nodes:
            if node_var == var:
                reason = "Bien conflict duoc chon ngau nhien tu csp.VARIABLES."
                action = "BIEN CONFLICT"
            else:
                reason = f"Bien ke dang xung dot voi {_building_label(var)}."
                action = "KE XUNG DOT"
            infos.append(
                NeighborInfo(
                    assignment[node_var].pos,
                    action,
                    f"{_building_label(node_var)} = {assignment[node_var].label()}",
                    "CONFLICT",
                    reason,
                )
            )
        for count_value, order, value in rows:
            status = "SELECTED" if value.name == best_value.name else "TRY"
            infos.append(
                NeighborInfo(
                    value.pos,
                    f"Thu {var} = {value.label()}",
                    f"conflicts={count_value}",
                    status,
                    "Chon vi conflict nho nhat; neu hoa thi random bang seed de tranh ket vong lap." if status == "SELECTED" else "Chi dem conflict giua bien nay va neighbors cua no.",
                )
            )

        assignment[var] = best_value
        new_conflicts, _ = total_conflicts(assignment)
        try_lines = ["Try values:"] + [f"- {value.label()}: {count_value} conflicts" for count_value, _, value in rows]
        related_text = "; ".join(
            f"{_building_label(a)} - {_building_label(b)}"
            for a, b in related_edges
        )
        extra = "\n".join([
            "Conflicted variables: " + ", ".join(_building_label(v) for v in conflicted),
            f"Selected conflicted variable: {_building_label(var)}",
            "Related conflict pairs: " + (related_text or "khong co"),
            f"Current color truoc khi sua: {old_value.label()}",
            neighbor_report(var, old_value, {**assignment, var: old_value}),
            *try_lines,
            f"Choose value: {best_value.label()} (domain order #{best_order + 1}, conflicts={best_count})",
            f"Update assignment: {_building_label(var)} = {best_value.label()}",
            "Conflict truoc khi sua: " + ("; ".join(reasons[:6]) if reasons else "khong co"),
        ])
        add_trace(
            var,
            assignment,
            infos,
            f"MIN-CONFLICTS STEP {step_no}\nChon ngau nhien mot conflicted variable tu csp.VARIABLES, thu tung mau trong domain va gan mau lam conflict nho nhat.",
            new_conflicts,
            extra,
        )

    final_path = _csp_route_path(stage, best_assignment) or [stage.start]
    return _stopped(
        name,
        start_time,
        stage,
        final_path,
        expanded,
        trace,
        f"Min-Conflicts het {max_steps} buoc, assignment tot nhat con {best_conflicts} conflict.",
        "normal",
    )
