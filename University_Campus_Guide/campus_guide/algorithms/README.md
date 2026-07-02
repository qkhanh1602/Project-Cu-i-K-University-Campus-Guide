# Pure Algorithms

Folder `algorithms/` la ban thuat toan thuan de doc khi van dap.

- Moi thuat toan nam trong mot file rieng.
- Khong co `TraceStep`, `NeighborInfo`, hay code ve giao dien.
- Project UI chay bang `algorithmsUI/`; folder do boc them trace de mo phong.
- Mot vai file dung helper du lieu tu `algorithmsUI`, nhung luong dieu khien thuat toan van nam trong file rieng.

## Thu tu theo chang

| Chang | File |
|---|---|
| 1 | `bfs.py`, `dfs.py`, `ids.py` |
| 2 | `greedy_best_first.py`, `astar.py`, `ida_star.py` |
| 3 | `simple_hill.py`, `local_beam.py`, `simulated_annealing.py` |
| 4 | `and_or_graph_search.py`, `belief_astar.py`, `belief_bfs.py` |
| 5 | `minimax.py`, `alpha_beta.py`, `expectimax.py` |
| 6 | `backtracking.py`, `forward_checking.py`, `min_conflicts.py` |

## Y tuong can nho

- BFS: queue FIFO, goal-test khi sinh child.
- DFS: stack LIFO, di sau truoc.
- IDS: lap DLS voi depth limit tang dan.
- Greedy Best First: chon node co `h(n)` nho nhat.
- A*: chon node co `f(n)=g(n)+h(n)` nho nhat.
- IDA*: DFS lap theo threshold `f(n)`, khong dung priority queue.
- Simple Hill Climbing: gap neighbor dau tien tot hon thi di ngay.
- Local Beam Search: sinh neighbor tu cac beam, giu `k` state tot nhat.
- Simulated Annealing: co the nhan buoc xau theo `exp(-delta/T)`.
- AND-OR Graph Search: OR chon action, AND xu ly tat ca result state.
- Belief search: moi node la tap vi tri co the, khong phai mot vi tri duy nhat.
- Minimax: MAX toi da hoa, MIN toi thieu hoa.
- Alpha-Beta: Minimax co cat nhanh bang `alpha` va `beta`.
- Expectimax: node chance tinh expected value.
- Backtracking: gan bien, sai thi quay lui.
- Forward Checking: sau khi gan bien, loai gia tri khoi domain bien ke.
- Min-Conflicts: bat dau bang assignment day du, sua bien dang xung dot.
