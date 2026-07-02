from __future__ import annotations

from typing import Callable, Dict

from map_data import Stage

from .bfs import bfs
from .dfs import dfs
from .ids import ids
from .greedy_best_first import greedy_best_first
from .astar import astar
from .ida_star import ida_star
from .simple_hill import simple_hill
from .local_beam import local_beam
from .simulated_annealing import simulated_annealing
from .and_or_graph_search import and_or_graph_search
from .belief_astar import belief_astar
from .belief_bfs import belief_bfs
from .minimax import minimax
from .alpha_beta import alpha_beta
from .expectimax import expectimax
from .backtracking_csp import backtracking_csp
from .forward_checking_csp import forward_checking_csp
from .min_conflicts_csp import min_conflicts_csp
from .common import NeighborInfo, SearchResult, TraceStep, is_valid_path, path_cost


ALGORITHM_GROUPS: tuple[tuple[str, tuple[tuple[str, Callable[[Stage], SearchResult]], ...]], ...] = (
    ("Stage 1 - Uninformed Search", (
        ("BFS", bfs),
        ("DFS", dfs),
        ("IDS", ids),
    )),
    ("Stage 2 - Informed Search", (
        ("Greedy Best First", greedy_best_first),
        ("A*", astar),
        ("IDA*", ida_star),
    )),
    ("Stage 3 - Local Search", (
        ("Hill Climbing", simple_hill),
        ("Local Beam Search", local_beam),
        ("Simulated Annealing", simulated_annealing),
    )),
    ("Stage 4 - Unknown Environment", (
        ("AND-OR Graph Search", and_or_graph_search),
        ("Belief State A*", belief_astar),
        ("Belief State BFS", belief_bfs),
    )),
    ("Stage 5 - Game Search", (
        ("Minimax", minimax),
        ("Alpha-Beta Pruning", alpha_beta),
        ("Expectimax", expectimax),
    )),
    ("Stage 6 - CSP", (
        ("Backtracking", backtracking_csp),
        ("Forward Checking", forward_checking_csp),
        ("Min-Conflicts", min_conflicts_csp),
    )),
)


ALGORITHMS: Dict[str, Callable[[Stage], SearchResult]] = {
    name: fn
    for _, group in ALGORITHM_GROUPS
    for name, fn in group
}


def run_algorithm(stage: Stage, algorithm: str) -> SearchResult:
    fn = ALGORITHMS.get(algorithm)

    if not fn:
        raise ValueError(f"Không tìm thấy thuật toán: {algorithm}")

    result = fn(stage)
    # Attach stage for UI components that need map context to render rich traces.
    result.stage = stage

    if result.path and result.path[0] == stage.start and result.path[-1] == stage.goal:
        if not is_valid_path(result.path, stage):
            result.status = "Lỗi path - có đoạn đứt hoặc đi vào vật cản"
        elif "Dừng" not in result.status:
            result.status = "Hoàn thành"
    elif result.path:
        if "Dừng" not in result.status and "Lỗi" not in result.status:
            result.status = "Dừng - chưa đến Goal"
    else:
        result.status = result.status or "Dừng - không tìm thấy đường"

    result.fallback_used = False

    return result
