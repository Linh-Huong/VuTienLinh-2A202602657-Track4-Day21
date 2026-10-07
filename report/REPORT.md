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

![failure](../results/figures/fail_[ĐIỀN].png)

[ĐIỀN]

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

# 3. Tự kiểm tra projection
python -m src.test_projection
```

## 6. Khai báo sử dụng AI

Ghi rõ đã dùng công cụ AI nào, dùng vào việc gì, và bạn đã tự kiểm chứng kết quả đó bằng cách nào. Nếu không dùng AI, ghi "Không sử dụng". Xem quy định ở `RULES.md` mục 2.

| Công cụ | Dùng cho việc gì | Bạn đã kiểm chứng thế nào |
|---|---|---|
| [ĐIỀN] | | |
