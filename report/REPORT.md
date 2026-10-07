# Báo cáo Day 6: Phát hiện vật cản cho robot/drone

> Thay **mọi** ô có chữ ĐIỀN nằm trong ngoặc vuông bằng nội dung của bạn, xoá luôn cả dấu ngoặc vuông. Lệnh `python tools/check_submission.py` sẽ báo FAIL nếu còn sót bất kỳ chỗ nào.

- **Họ tên:** Vũ Tiến Linh
- **MSSV:** 2A202602657
- **Lớp:** AI20K-T4
- **Link repo:** https://github.com/Linh-Huong/VuTienLinh-2A202602657-Track4-Day21
- **Topic:** D — Robot/drone obstacle
- **Dataset:** data/kitti_mini
- **Các frame đã dùng:** 000011, 000015, 000021

> Hãy viết ngắn: mỗi mục từ 3 đến 8 dòng, ưu tiên số liệu và hình ảnh.

## 1. Claim

Một câu khẳng định kỹ thuật có thể kiểm chứng. Ví dụ: *"Lệch yaw 1° làm 12% điểm LiDAR rơi ra khỏi vật thể ở 30 m, phát hiện được bằng edge-alignment score với ngưỡng X."*

Trong pipeline phát hiện vật cản bằng RANSAC và DBSCAN, tăng ngưỡng khoảng cách mặt phẳng distance_threshold từ 0.20m lên 0.40m làm triệt tiêu 100% các vật cản thấp sát đất (< 0.4m), trong khi tăng bán kính gom cụm eps vượt quá 0.7m làm dính chùm 2 người đi bộ độc lập thành 1 vật cản duy nhất.

## 2. Evidence

File số liệu chi tiết: [`results/obstacle_benchmark.csv`](../results/obstacle_benchmark.csv).  
Đo đạc trên frame `000011` (ROI 54.435 điểm, voxel downsample 0.1m còn 20.429 điểm):

| Cấu hình tham số | Điểm mặt đất (% ground) | Điểm vật cản | Số cụm | d_min (m) | Latency p50 / p95 (ms) | Nhận xét hiện tượng |
|---|---|---|---|---|---|---|
| eps = 0.3m (d_th = 0.20m) | 9.996 (48.9%) | 10.433 | 71 | 1.57 | 48.0 / 74.4 | Over-segmentation: xé nhỏ người và xe |
| eps = 0.5m (d_th = 0.20m) | 9.996 (48.9%) | 10.433 | 46 | 1.57 | 81.2 / 88.6 | Baseline chuẩn: tách đều các vật thể |
| eps = 0.8m (d_th = 0.20m) | 9.996 (48.9%) | 10.433 | 35 | 1.57 | 85.7 / 91.3 | Under-segmentation: 2 người đi bộ dính chùm |
| d_th = 0.10m (eps = 0.5m) | 8.378 (41.0%) | 12.051 | 69 | 1.57 | 76.0 / 86.0 | Ngưỡng mỏng: sót nhiễu mặt đường |
| d_th = 0.20m (eps = 0.5m) | 9.996 (48.9%) | 10.433 | 46 | 1.57 | 74.9 / 85.9 | Baseline chuẩn: tách sạch mặt đường |
| d_th = 0.30m (eps = 0.5m) | 10.501 (51.4%) | 9.928 | 46 | 1.57 | 74.3 / 83.2 | Bắt đầu lẹm vào phần chân người đi bộ |
| d_th = 0.45m (eps = 0.5m) | 12.248 (59.9%) | 8.181 | 44 | 1.57 | 46.7 / 53.8 | Lẹm 2.252 điểm vật cản vào ground |

![sweep](../results/figures/obstacle_sweep.png)
![demo](../results/figures/demo_d_obstacle_000011.png)

**Nhận xét xu hướng:**
- Khi tăng bán kính `eps` từ 0.3m lên 0.8m: Số cụm giảm mạnh từ 71 xuống 35 cụm (giảm 50.7%) do hiện tượng under-segmentation làm các vật cản đứng gần nhau dính thành một cụm lớn. Độ trễ trung vị $p_{50}$ tăng từ 48.0 ms lên 85.7 ms do số lân cận cần duyệt trong bán kính lớn hơn.
- Khi tăng ngưỡng phẳng `distance_threshold` từ 0.10m lên 0.45m: Số điểm gán nhãn là mặt đất tăng từ 8.378 lên 12.248 điểm, trong khi điểm vật cản bị sụt giảm hơn 3.870 điểm (mất 32.1% số điểm chướng ngại vật). Điều này chứng minh ngưỡng RANSAC quá lớn sẽ nuốt trọn phần cẳng chân của người đi bộ vào mặt đất.

## 3. Failure case

Nêu khi nào hệ thống hoặc phương pháp fail, vì sao fail, và liên hệ tới lớp nào trong 6 lớp debug: I/O, Geometry, Time, Preprocess, Model, Metric.

### Case 1: Vật thấp (< 0.5 m) bị RANSAC coi là mặt đất
![failure 1](../results/figures/fail_01_low_obstacle_ransac.png)

- **Trường hợp:** Vật cản thấp (chiều cao dưới 0.5 m sát sàn như pallet hàng, gờ giảm tốc, chân người đi bộ) trên KITTI frame 000011 khi cài đặt ngưỡng lọc mặt phẳng RANSAC `distance_threshold` từ 0.20m lên 0.45m.
- **Quan sát:** Ở ngưỡng chuẩn `distance_threshold = 0.15m`, các điểm của vật thấp (chiều cao $h < 0.5m$ so với mặt đường) được nhận diện đầy đủ trong nhóm vật cản và đóng khung bounding box. Khi tăng ngưỡng lên `0.45m`, toàn bộ các điểm vật cản thấp (< 0.45m sát đất, tương đương hơn 2.250 điểm) bị RANSAC gán nhãn nhầm thành mặt đất (màu xám). Kết quả: vật cản thấp biến mất 100% trên bản đồ (False Negative), robot hoàn toàn không phát hiện được chướng ngại vật trước mặt.
- **Nguyên nhân:** Thuật toán RANSAC giả định mặt đất là một mặt phẳng và coi mọi điểm cách mặt phẳng một khoảng $\le distance\_threshold$ đều là sàn đường. Do đó, bất kỳ vật thể nào có chiều cao thấp hơn ngưỡng lọc sẽ bị "gọt phẳng" hoàn toàn vào mặt đất.
- **Lớp debug:** **Preprocess / Geometry** (Mô hình hình học mặt phẳng toàn cục quá thô và bước tiền xử lý chọn siêu tham số ngưỡng khoảng cách `distance_threshold` quá lỏng).
- **Cách phát hiện khi chạy thật:** Xây dựng biểu đồ phân bố chiều cao tương đối của các điểm so với mặt đất (Height-above-ground histogram). Nếu phát hiện các cụm điểm mặt đất nhô cao cục bộ từ 0.1m đến 0.5m với mật độ bất thường, hệ thống phải kích hoạt cảnh báo rò rỉ vật cản thấp (Low-obstacle Ground Leakage).

### Case 2: Hai người đứng gần nhau bị DBSCAN gộp thành một cụm
![failure 2](../results/figures/fail_02_dbscan_merge_pedestrians.png)

- **Trường hợp:** Hai người đi bộ đứng gần nhau (khoảng cách mép ~0.6m) tại tọa độ $X \in [12, 14]m, Y \in [-5.5, -5.0]m$ trên frame 000011 khi tăng bán kính gom cụm `eps` từ 0.35m lên 0.80m.
- **Quan sát:** Ở `eps = 0.35m`, DBSCAN tách chính xác 2 người thành 2 cụm riêng biệt (`Cluster 1` và `Cluster 2`) với 2 bounding box độc lập, để lộ khe hở an toàn ở giữa. Khi tăng `eps = 0.80m`, hai người bị gộp chung vào 1 cụm duy nhất (1 bounding box to đùng bao trùm cả hai).
- **Nguyên nhân:** Hiện tượng Under-segmentation trong mật độ cụm. Khi bán kính tìm kiếm lân cận $eps > 0.6m$, các điểm của người thứ nhất bắc cầu sang điểm của người thứ hai, biến hai vật cản độc lập thành một liên thông duy nhất. Robot sẽ hiểu nhầm đây là một vật cản kích thước lớn và không nhận biết được lối đi ở giữa.
- **Lớp debug:** **Preprocess** (Chọn siêu tham số bán kính phân cụm `eps` quá lớn so với khoảng cách thực tế giữa các thực thể).
- **Cách phát hiện khi chạy thật:** Giám sát kích thước hình học của bounding box (chiều dài, chiều rộng). Nếu một cụm được gán nhãn là pedestrian/dynamic obstacle nhưng kích thước vượt quá giới hạn người thông thường (> 1.2m bề ngang), kích hoạt cờ cảnh báo "Cluster Under-segmentation" để phân tách lại bằng thuật toán K-Means hoặc Hierarchical Clustering cục bộ.

## 4. Khuyến nghị nếu triển khai thật

Use-case cụ thể (ADAS / robot / drone), trade-off và bước tiếp theo.

[ĐIỀN]

## 5. Cách chạy lại

Các lệnh tái tạo lại toàn bộ kết quả từ repo sạch.

```bash
# 1. Chạy demo phát hiện vật cản (Topic D)
python -m src.obstacle_detector --data-root data/kitti_mini --frame 000011

# 2. Chạy thí nghiệm sweep benchmark (sinh CSV và đồ thị)
python -m src.sweep_experiment --data-root data/kitti_mini --frame 000011 --n-runs 20

# 3. Vẽ biểu đồ benchmark
python -m src.plot_sweep

# 4. Sinh ảnh phân tích failure case
python -m src.visualize_failure

# 5. Tự kiểm tra projection
python -m src.test_projection
```

## 6. Khai báo sử dụng AI

Ghi rõ đã dùng công cụ AI nào, dùng vào việc gì, và bạn đã tự kiểm chứng kết quả đó bằng cách nào. Nếu không dùng AI, ghi "Không sử dụng". Xem quy định ở `RULES.md` mục 2.

| Công cụ | Dùng cho việc gì | Bạn đã kiểm chứng thế nào |
|---|---|---|
| [ĐIỀN] | | |
