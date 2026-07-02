# University Campus Guide

**University Campus Guide** là project mô phỏng và trực quan hóa các thuật toán Trí tuệ nhân tạo trên bản đồ trường đại học. Project được viết bằng **Python** và **PySide6**, cho phép người dùng chọn từng chặng, chọn thuật toán, chạy animation trên bản đồ và xem **Search Trace** để hiểu từng bước thuật toán hoạt động.

Đây là project mô phỏng các thuật toán AI cổ điển như tìm kiếm đường đi, tìm kiếm có heuristic, tìm kiếm cục bộ, môi trường không chắc chắn, tìm kiếm đối kháng và CSP.

---

## 1. Chức năng chính

Project hỗ trợ 6 nhóm thuật toán, tổng cộng 18 thuật toán:

| Chặng | Nhóm thuật toán | Thuật toán |
|---|---|---|
| Chặng 1 | Tìm kiếm mù / Uninformed Search | BFS, DFS, IDS |
| Chặng 2 | Tìm kiếm có thông tin / Informed Search | Greedy Best First, A*, IDA* |
| Chặng 3 | Tìm kiếm cục bộ / Local Search | Hill Climbing, Local Beam Search, Simulated Annealing |
| Chặng 4 | Môi trường không chắc chắn / Unknown Environment | AND-OR Graph Search, Belief State A*, Belief State BFS |
| Chặng 5 | Tìm kiếm đối kháng / Game Search | Minimax, Alpha-Beta Pruning, Expectimax |
| Chặng 6 | Thỏa mãn ràng buộc / CSP | Backtracking, Forward Checking, Min-Conflicts |

Các chức năng chính của giao diện:

- Hiển thị bản đồ trường đại học dưới dạng lưới.
- Cho phép chọn 1 trong 6 chặng.
- Cho phép chọn 1 trong 3 thuật toán của mỗi chặng.
- Chạy animation từng bước trên bản đồ.
- Hiển thị kết quả gồm trạng thái, số node mở rộng, số bước, thời gian chạy.
- Hiển thị **Search Trace chi tiết** để giải thích quá trình thuật toán hoạt động.
- Hỗ trợ chọn GOAL bất kỳ trên bản đồ để thử nghiệm lại thuật toán.
- Hỗ trợ chế độ tự chơi bằng phím WASD.
- Hỗ trợ tổng hợp runtime và xem biểu đồ so sánh thuật toán.

---

## 2. Yêu cầu cài đặt

Cần cài sẵn:

- Python 3.10 trở lên, khuyến nghị Python 3.12.
- pip.
- Hệ điều hành Windows, macOS hoặc Linux đều có thể chạy nếu cài được PySide6.

Thư viện cần thiết nằm trong file:

```bash
requirements.txt
```

Nội dung chính:

```bash
PySide6>=6.6
```

---

## 3. Cấu trúc project

Cấu trúc chính của project:

```text
University_Campus_Guide/
│
├── main.py
├── requirements.txt
├── run_game.bat
├── test_core.py
│
└── campus_guide/
    ├── main.py
    ├── app_window.py
    ├── map_data.py
    ├── map_scene.py
    ├── trace_dialog.py
    │
    ├── stages/
    │   ├── stage1.py
    │   ├── stage2.py
    │   ├── stage3.py
    │   ├── stage4.py
    │   ├── stage5.py
    │   └── stage6.py
    │
    ├── algorithms/
    │   ├── bfs.py
    │   ├── dfs.py
    │   ├── ids.py
    │   ├── greedy_best_first.py
    │   ├── astar.py
    │   ├── ida_star.py
    │   ├── simple_hill.py
    │   ├── local_beam.py
    │   ├── simulated_annealing.py
    │   ├── and_or_graph_search.py
    │   ├── belief_astar.py
    │   ├── belief_bfs.py
    │   ├── minimax.py
    │   ├── alpha_beta.py
    │   ├── expectimax.py
    │   ├── backtracking.py
    │   ├── forward_checking.py
    │   └── min_conflicts.py
    │
    └── algorithmsUI/
        ├── search_algorithms.py
        ├── common.py
        ├── trace_recorder.py
        ├── game_common.py
        ├── csp_common.py
        ├── belief_common.py
        └── các file thuật toán có trace cho giao diện
```

Ý nghĩa các file và thư mục quan trọng:

| File / thư mục | Công dụng |
|---|---|
| `main.py` | File chạy chính ở thư mục gốc. |
| `requirements.txt` | Danh sách thư viện cần cài. |
| `run_game.bat` | File chạy nhanh trên Windows. |
| `test_core.py` | File kiểm tra 18 thuật toán và dữ liệu bản đồ. |
| `campus_guide/app_window.py` | Giao diện chính của chương trình. |
| `campus_guide/map_data.py` | Dữ liệu bản đồ, landmark, START, GOAL, cost, vật cản, cấu hình chặng. |
| `campus_guide/map_scene.py` | Vẽ bản đồ, ô lưới, đường đi, trạng thái animation. |
| `campus_guide/trace_dialog.py` | Cửa sổ Search Trace chi tiết. |
| `campus_guide/stages/` | Cấu hình riêng của 6 chặng. |
| `campus_guide/algorithms/` | Thuật toán. |
| `campus_guide/algorithmsUI/` | Thuật toán dùng trong giao diện, có thêm trace và dữ liệu phục vụ animation. |

Lưu ý: trong bản project hiện tại, giao diện gọi thuật toán thông qua:

```text
campus_guide/algorithmsUI/search_algorithms.py
```


## 4. Cách cài đặt và chạy project

### Cách 1: Chạy bằng Command Prompt / Terminal

Bước 1: Mở terminal tại thư mục chứa project.

Ví dụ trên Windows:

```bash
cd University_Campus_Guide
```

Thư mục đúng là thư mục có các file:

```text
main.py
requirements.txt
test_core.py
campus_guide/
```

Bước 2: Cài thư viện:

```bash
pip install -r requirements.txt
```

Nếu máy có nhiều bản Python, có thể dùng:

```bash
py -3.12 -m pip install -r requirements.txt
```

Bước 3: Chạy chương trình:

```bash
python main.py
```

Hoặc:

```bash
py -3.12 main.py
```

---

### Cách 2: Chạy nhanh bằng file `run_game.bat` trên Windows

Trong thư mục project, bấm đúp vào file:

```text
run_game.bat
```

File này sẽ tự chạy lệnh cài thư viện và mở chương trình bằng Python 3.12:

```bat
py -3.12 -m pip install -r requirements.txt
py -3.12 main.py
```

Nếu máy không có Python 3.12, hãy mở `run_game.bat` và đổi `py -3.12` thành `python` hoặc phiên bản Python đang có trên máy.

---

### Cách 3: Chạy trong VS Code

1. Mở VS Code.
2. Chọn **File > Open Folder**.
3. Chọn thư mục `University_Campus_Guide` có file `main.py`.
4. Mở Terminal trong VS Code.
5. Chạy:

```bash
pip install -r requirements.txt
python main.py
```

Nếu VS Code hỏi chọn Python Interpreter, hãy chọn Python 3.10 trở lên.

---

## 5. Cách sử dụng giao diện

Sau khi chạy `main.py`, cửa sổ **University Campus Guide** sẽ mở ra.

### Bước 1: Chọn chặng

Bên trái hoặc phía trên giao diện có các nút chọn chặng từ 1 đến 6.

| Phím | Chức năng |
|---|---|
| `1` | Chọn chặng 1 |
| `2` | Chọn chặng 2 |
| `3` | Chọn chặng 3 |
| `4` | Chọn chặng 4 |
| `5` | Chọn chặng 5 |
| `6` | Chọn chặng 6 |

Mỗi chặng có START, GOAL và môi trường riêng để làm nổi bật nhóm thuật toán tương ứng.

---

### Bước 2: Chọn thuật toán

Mỗi chặng có 3 thuật toán. Có thể chọn bằng nút trên giao diện hoặc bằng phím tắt:

| Phím | Thuật toán trong chặng hiện tại |
|---|---|
| `Q` | Thuật toán thứ nhất |
| `W` | Thuật toán thứ hai |
| `E` | Thuật toán thứ ba |

Ví dụ:

- Ở chặng 1: `Q = BFS`, `W = DFS`, `E = IDS`.
- Ở chặng 2: `Q = Greedy Best First`, `W = A*`, `E = IDA*`.
- Ở chặng 5: `Q = Minimax`, `W = Alpha-Beta Pruning`, `E = Expectimax`.

---

### Bước 3: Chạy thuật toán

Sau khi chọn chặng và thuật toán, bấm:

```text
SPACE
```

hoặc nút:

```text
SPACE - Chạy AI
```

Chương trình sẽ chạy thuật toán và animate đường đi trên bản đồ.

---

### Bước 4: Xem từng bước thủ công

Có thể xem từng bước bằng:

| Phím | Chức năng |
|---|---|
| `N` | Xem bước tiếp theo |
| `Right` | Xem bước tiếp theo |
| `Left` | Quay lại bước trước |

Khi xem từng bước, bản đồ sẽ cập nhật node hiện tại, frontier, reached hoặc trạng thái đặc biệt tùy thuật toán.

---

### Bước 5: Mở Search Trace chi tiết

Bấm:

```text
T
```

hoặc nút:

```text
T - Search Trace chi tiết
```

Search Trace giúp xem rõ thuật toán đang làm gì ở từng bước.

Ví dụ:

- BFS: hiển thị Current Node, Frontier, Reached.
- A*: hiển thị g(n), h(n), f(n).
- IDA*: hiển thị threshold theo f(n).
- Local Beam: hiển thị beam hiện tại, candidates và node được chọn.
- AND-OR: hiển thị OR_SEARCH, AND_SEARCH, result states.
- Game Search: hiển thị cây đối kháng, state_score, branch_value hoặc EV.
- CSP: hiển thị variables, domain, assignment, conflict và constraint.

---

## 6. Các phím tắt quan trọng

| Phím | Chức năng |
|---|---|
| `1` đến `6` | Chọn chặng |
| `Q`, `W`, `E` | Chọn thuật toán trong chặng hiện tại |
| `SPACE` | Chạy / tạm dừng / tiếp tục mô phỏng |
| `N` hoặc `Right` | Sang bước tiếp theo |
| `Left` | Quay lại bước trước |
| `T` | Mở Search Trace chi tiết |
| `P` | Bật / tắt chế độ tự chơi |
| `W`, `A`, `S`, `D` | Di chuyển thủ công khi bật chế độ tự chơi |
| `L` | Bật / tắt nhãn trên bản đồ |
| `V` | Bật / tắt hiển thị collision |
| `Y` | Bật chế độ chọn GOAL bất kỳ |
| `R` | Reset GOAL về vị trí gốc |
| `+` | Tăng tốc độ animation |
| `-` | Giảm tốc độ animation |
| `Esc` | Đóng chương trình |

---

## 7. Cách chọn GOAL bất kỳ

Project hỗ trợ chọn GOAL mới để thử thuật toán trong môi trường khác.

Cách dùng:

1. Bấm phím `Y` hoặc nút **Chọn GOAL bất kỳ**.
2. Click vào một ô đi được trên bản đồ.
3. Chọn thuật toán và bấm `SPACE` để chạy lại.
4. Nếu muốn quay về GOAL ban đầu, bấm `R` hoặc nút **Reset GOAL gốc**.

Lưu ý:

- Không chọn GOAL trùng START.
- Không chọn GOAL vào tòa nhà, hồ nước, vật cản hoặc ô không đi được.
- Một số chặng đặc biệt như CSP dùng START/GOAL chủ yếu để minh họa route, không phải mục tiêu chính của bài toán.

---

## 8. Cách đọc kết quả sau khi chạy

Sau khi thuật toán chạy xong, giao diện hiển thị các thông số như:

| Thông số | Ý nghĩa |
|---|---|
| Status / Kết quả | Thuật toán hoàn thành hay dừng trước GOAL |
| Cost | Tổng chi phí đường đi, nếu chặng có dùng cost |
| Expanded | Số node/trạng thái được mở rộng |
| Steps | Số bước trong path hiển thị |
| Time | Thời gian chạy thuật toán |
| Search Trace | Diễn giải từng bước xử lý |

Một số thuật toán local search hoặc belief search có thể dừng trước GOAL. Đây không nhất thiết là lỗi, mà có thể phản ánh bản chất thuật toán hoặc cấu hình môi trường.

Ví dụ:

- Hill Climbing có thể kẹt ở local optimum.
- Simulated Annealing có thể dừng trước GOAL khi hết điều kiện mô phỏng.
- Belief State có thể dừng nếu không tìm được chuỗi action chung đưa toàn bộ belief về GOAL.

---

## 9. Giải thích từng chặng

### Chặng 1: Tìm kiếm mù / Uninformed Search

Thuật toán:

- BFS
- DFS
- IDS

Mục tiêu của chặng 1 là minh họa các thuật toán không sử dụng heuristic.

- BFS dùng Queue FIFO và mở rộng theo từng lớp.
- DFS dùng Stack LIFO và đi sâu theo một nhánh trước.
- IDS chạy DFS nhiều lần với giới hạn độ sâu tăng dần.

Chặng này có môi trường tương đối đơn giản và chi phí bước đi gần như đều, phù hợp để quan sát frontier, reached và parent.

---

### Chặng 2: Tìm kiếm có thông tin / Informed Search

Thuật toán:

- Greedy Best First
- A*
- IDA*

Chặng này sử dụng heuristic Manhattan để đánh giá khoảng cách đến GOAL.

- Greedy Best First chọn node có h(n) nhỏ nhất.
- A* chọn node theo f(n) = g(n) + h(n).
- IDA* dùng DFS lặp lại theo threshold của f(n).

Chặng này có vùng cost cao để thể hiện sự khác nhau giữa thuật toán chỉ nhìn h(n) và thuật toán có xét thêm g(n).

---

### Chặng 3: Tìm kiếm cục bộ / Local Search

Thuật toán:

- Hill Climbing
- Local Beam Search
- Simulated Annealing

Mục tiêu của chặng này là minh họa cách thuật toán chọn trạng thái dựa trên đánh giá cục bộ.

- Hill Climbing chọn neighbor tốt hơn hiện tại theo h(n).
- Local Beam Search giữ k trạng thái tốt nhất và sinh candidates từ toàn bộ beam.
- Simulated Annealing có thể chấp nhận bước đi xấu hơn theo xác suất để tránh kẹt cục bộ.

Local Search không đảm bảo luôn tìm được GOAL, vì thuật toán có thể bị kẹt ở local optimum hoặc dừng trước mục tiêu.

---

### Chặng 4: Môi trường không chắc chắn / Unknown Environment

Thuật toán:

- AND-OR Graph Search
- Belief State A*
- Belief State BFS

Mục tiêu là mô phỏng môi trường không chắc chắn.

- AND-OR Graph Search dùng OR node để chọn action và AND node để xử lý tất cả result states của action đó.
- Belief State A* và Belief State BFS không tìm trên một vị trí đơn, mà tìm trên tập các vị trí có thể xảy ra.

Trong Belief State, một node là một belief state, tức là tập trạng thái có thể xảy ra. Thuật toán chỉ hoàn thành khi toàn bộ belief đạt GOAL.

---

### Chặng 5: Tìm kiếm đối kháng / Game Search

Thuật toán:

- Minimax
- Alpha-Beta Pruning
- Expectimax

Chặng này mô phỏng môi trường có yếu tố đối kháng hoặc ngẫu nhiên.

- Minimax: MAX là agent, MIN là phía bất lợi.
- Alpha-Beta: giữ logic Minimax nhưng cắt các nhánh không cần xét bằng alpha và beta.
- Expectimax: dùng CHANCE node và tính expected value khi kết quả có xác suất.

Cách tính điểm trạng thái dựa trên các yếu tố:

- Tiến gần GOAL được cộng điểm.
- Mỗi bước đi bị trừ điểm.
- Môi trường xấu hoặc vùng rủi ro bị trừ điểm.
- Đến GOAL được cộng thưởng lớn.

Trong Search Trace chặng 5:

- `state_score` là điểm trực tiếp của trạng thái đang hiển thị.
- `branch_value` là giá trị nhánh dùng bởi Minimax hoặc Alpha-Beta.
- `EV` là giá trị kỳ vọng dùng bởi Expectimax.

---

### Chặng 6: CSP / Constraint Satisfaction Problem

Thuật toán:

- Backtracking
- Forward Checking
- Min-Conflicts

Chặng 6 là bài toán tô màu tòa nhà.

CSP gồm 3 thành phần:

| Thành phần | Trong project |
|---|---|
| Variables | Các tòa nhà |
| Domain | Các màu: Cam, Hồng, Xanh, Tím |
| Constraints | Hai tòa nhà kề nhau không được cùng màu |

Ý nghĩa thuật toán:

- Backtracking gán màu từng biến, sai thì quay lui.
- Forward Checking gán màu và loại trước màu không hợp lệ khỏi domain của biến kề.
- Min-Conflicts bắt đầu bằng assignment đầy đủ rồi sửa dần các biến đang xung đột.

Chặng 6 không phải bài toán tìm đường chính. START và GOAL chủ yếu dùng để minh họa route sau khi bài toán CSP có nghiệm.

---

## 10. Kiểm tra project bằng `test_core.py`

Để kiểm tra nhanh dữ liệu bản đồ và 18 thuật toán, chạy:

```bash
python test_core.py
```

Hoặc:

```bash
py -3.12 test_core.py
```

File test sẽ kiểm tra:

- Có đủ 18 thuật toán.
- START và GOAL của từng chặng hợp lệ.
- Landmark và ô vật cản không bị lỗi.
- Thuật toán có trả path hoặc partial path hợp lệ.
- Search Trace không bị rỗng.
- Không dùng fallback ngoài thuật toán chính.

Nếu chạy đúng, terminal sẽ in kết quả từng thuật toán theo dạng:

```text
Stage 1 | BFS | DONE | cost=... | expanded=... | steps=... | Hoàn thành
...
OK: 18/18 algorithms executed.
```

---


## 11. Tóm tắt project

University Campus Guide là project trực quan hóa 18 thuật toán AI trên bản đồ trường đại học. Project giúp người dùng không chỉ thấy kết quả cuối cùng mà còn quan sát quá trình thuật toán suy luận thông qua animation và Search Trace. Sáu chặng của project tương ứng với sáu nhóm nội dung quan trọng trong môn Trí tuệ nhân tạo: tìm kiếm mù, tìm kiếm có thông tin, tìm kiếm cục bộ, môi trường không chắc chắn, tìm kiếm đối kháng và CSP.

---

## 12. Tác giả

Project được xây dựng phục vụ học tập và báo cáo môn Trí tuệ nhân tạo.

Tên project: **University Campus Guide**

Ngôn ngữ lập trình: **Python**

Thư viện giao diện: **PySide6**
