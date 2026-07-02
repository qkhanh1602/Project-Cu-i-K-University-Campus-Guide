# UI Algorithms

Folder `algorithmsUI/` la ban thuat toan project dang chay.

Khac voi `algorithms/`, cac file o day tao them trace de:

- ve buoc chay tren ban do;
- hien Search Trace chi tiet;
- hien bang/cay giai thich cho cac chang dac biet.

## Thu tu theo chang

Thu tu chuan nam trong `search_algorithms.py` tai `ALGORITHM_GROUPS`.

| Chang | File |
|---|---|
| 1 | `bfs.py`, `dfs.py`, `ids.py` |
| 2 | `greedy_best_first.py`, `astar.py`, `ida_star.py` |
| 3 | `simple_hill.py`, `local_beam.py`, `simulated_annealing.py` |
| 4 | `and_or_graph_search.py`, `belief_astar.py`, `belief_bfs.py` |
| 5 | `minimax.py`, `alpha_beta.py`, `expectimax.py` |
| 6 | `backtracking_csp.py`, `forward_checking_csp.py`, `min_conflicts_csp.py` |

## File dung chung

- `common.py`: dataclass ket qua, trace step, path cost, validate path.
- `trace_recorder.py`: tao `TraceStep` tu su kien cua thuat toan.
- `belief_common.py`: helper cho belief state.
- `game_common.py`: helper cho Minimax, Alpha-Beta, Expectimax.
- `csp_common.py`: helper cho CSP coloring.
