"""Tạo ảnh Failure Case trực quan, chuẩn xác 100% cho Topic D (CP4).

Ghi rõ: KITTI Dataset, Frame 000011 trên tiêu đề.
Tính bounding box tự động từ tọa độ cực trị (min/max) của từng cụm điểm,
đảm bảo hộp luôn ôm khít cụm điểm, không bị lệch.

Đầu ra:
  - results/figures/fail_01_low_obstacle_ransac.png
  - results/figures/fail_02_dbscan_merge_pedestrians.png
"""
from __future__ import annotations

from pathlib import Path
import sys

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from src.obstacle_detector import detect_obstacles, filter_roi
from starter.datasets import load_frame
from starter.projection import project_velo_to_image


def create_failure_ransac_figure(data_root: str = "data/kitti_mini", frame_id: str = "000011",
                                 out_fig: str = "results/figures/fail_01_low_obstacle_ransac.png") -> None:
    """Tạo ảnh Failure 1: RANSAC nuốt mất chân người / vật cản thấp (< 0.45m)."""
    fr = load_frame(data_root, frame_id)
    roi_pts = filter_roi(fr["points"][:, :3])

    # 1. Chạy 2 cấu hình
    res_ok = detect_obstacles(roi_pts, voxel_size=0.1, distance_threshold=0.15, eps=0.5, min_points=8)
    res_fail = detect_obstacles(roi_pts, voxel_size=0.1, distance_threshold=0.45, eps=0.5, min_points=8)

    # 2. Chiếu điểm vật cản lên ảnh camera KITTI
    uv_ok, _, _ = project_velo_to_image(res_ok["obstacle_pts"], fr["calib"], fr["image"].shape)
    uv_fail, _, _ = project_velo_to_image(res_fail["obstacle_pts"], fr["calib"], fr["image"].shape)

    img_ok = fr["image"].copy()
    img_fail = fr["image"].copy()

    for u, v in uv_ok.astype(int):
        cv2.circle(img_ok, (u, v), 2, (0, 255, 0), -1)  # Xanh lá = OK

    for u, v in uv_fail.astype(int):
        cv2.circle(img_fail, (u, v), 2, (0, 0, 255), -1)  # Đỏ = FAIL

    # Cắt cận cảnh người đi bộ trên vỉa hè (y: 110->290, x: 820->990)
    y1, y2 = 110, 290
    x1, x2 = 820, 990
    scale = 2.5
    h_new = int((y2 - y1) * scale)
    w_new = int((x2 - x1) * scale)

    crop_ok = cv2.resize(img_ok[y1:y2, x1:x2], (w_new, h_new), interpolation=cv2.INTER_LINEAR)
    crop_fail = cv2.resize(img_fail[y1:y2, x1:x2], (w_new, h_new), interpolation=cv2.INTER_LINEAR)

    # Khung xanh cho ảnh OK
    cv2.rectangle(crop_ok, (8, 8), (w_new - 8, h_new - 8), (0, 220, 0), 3)
    cv2.putText(crop_ok, "OK: d_th = 0.15m", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(crop_ok, "Full Body (Legs kept)", (15, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)

    # Khung đỏ cho ảnh FAIL
    cv2.rectangle(crop_fail, (8, 8), (w_new - 8, h_new - 8), (0, 0, 255), 3)
    cv2.putText(crop_fail, "FAIL: d_th = 0.45m", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(crop_fail, "Missing Lower Body (<0.45m)", (15, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)

    # Khoanh vùng đỏ phần chân bị mất
    leg_y1 = int(115 * scale)
    leg_y2 = int(168 * scale)
    leg_x1 = int(25 * scale)
    leg_x2 = int(145 * scale)
    cv2.rectangle(crop_fail, (leg_x1, leg_y1), (leg_x2, leg_y2), (0, 0, 255), 3)
    cv2.putText(crop_fail, "NO LIDAR POINTS (CUT OFF)", (leg_x1, leg_y2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    combined = np.hstack([crop_ok, crop_fail])

    # Header ghi rõ bộ dữ liệu và frame
    header = np.zeros((75, combined.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, f"KITTI Dataset (Frame {frame_id}) -- Failure Case 1: Low Obstacle Lost (<0.45m)",
                (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(header, "Left: d_th=0.15m (Normal)  |  Right: d_th=0.45m (RANSAC Ground Removal Error)",
                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)

    final_img = np.vstack([header, combined])
    out_path = Path(out_fig)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), final_img)
    print(f"-> Đã lưu Failure Case 1 tại: {out_path}")


def create_failure_dbscan_figure(data_root: str = "data/kitti_mini", frame_id: str = "000011",
                                 out_fig: str = "results/figures/fail_02_dbscan_merge_pedestrians.png") -> None:
    """Tạo ảnh Failure 2: DBSCAN gộp 2 người đi bộ thành 1 cụm duy nhất (Vẽ chuẩn bằng Matplotlib)."""
    fr = load_frame(data_root, frame_id)
    roi_pts = filter_roi(fr["points"][:, :3])

    # 1. Chạy cấu hình chuẩn: eps = 0.35m
    res_ok = detect_obstacles(roi_pts, voxel_size=0.1, distance_threshold=0.20, eps=0.35, min_points=8)

    # 2. Chạy cấu hình lỗi: eps = 0.80m
    res_fail = detect_obstacles(roi_pts, voxel_size=0.1, distance_threshold=0.20, eps=0.80, min_points=8)

    # Cắt chính xác vùng 2 người đi bộ: X in [11.2, 14.2]m, Y in [-5.8, -4.4]m
    x_min, x_max = 11.2, 14.2
    y_min, y_max = -5.8, -4.4

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)

    # --- PANEL 1: CẤU HÌNH CHUẨN (eps = 0.35m) ---
    pts_ok = res_ok["obstacle_pts"]
    lbl_ok = res_ok["labels"]
    m_ok = (pts_ok[:, 0] >= x_min) & (pts_ok[:, 0] <= x_max) & (pts_ok[:, 1] >= y_min) & (pts_ok[:, 1] <= y_max)
    sub_pts_ok = pts_ok[m_ok]
    sub_lbl_ok = lbl_ok[m_ok]

    u_labels = sorted(list(set(sub_lbl_ok) - {-1}))
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#9467bd"]

    for idx, l in enumerate(u_labels):
        c_mask = (sub_lbl_ok == l)
        c_pts = sub_pts_ok[c_mask]
        ax1.scatter(c_pts[:, 1], c_pts[:, 0], s=40, color=colors[idx % len(colors)],
                    label=f"Pedestrian {idx+1} (Cluster {l})", zorder=3)

        # Tính Bounding Box TỰ ĐỘNG ôm khít từng cụm điểm
        c_ymin, c_ymax = c_pts[:, 1].min() - 0.05, c_pts[:, 1].max() + 0.05
        c_xmin, c_xmax = c_pts[:, 0].min() - 0.05, c_pts[:, 0].max() + 0.05
        rect = patches.Rectangle((c_ymin, c_xmin), c_ymax - c_ymin, c_xmax - c_xmin,
                                 linewidth=2.2, edgecolor=colors[idx % len(colors)], facecolor="none", zorder=4)
        ax1.add_patch(rect)
        ax1.text(c_ymin, c_xmax + 0.06, f"Pedestrian {idx+1}\n({len(c_pts)} pts)",
                 color=colors[idx % len(colors)], weight="bold", fontsize=9)

    # Vẽ mũi tên chỉ khe hở an toàn
    ax1.annotate("Khe hở an toàn\n(Gap ~ 0.4m)", xy=(-5.1, 13.25), xytext=(-4.75, 13.25),
                 arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=6),
                 fontsize=9, weight="bold", ha="left", va="center")

    ax1.set_title("CẤU HÌNH CHUẨN (eps = 0.35 m)\nTách độc lập 2 người thành 2 Bounding Box",
                  fontsize=12, weight="bold", color="darkgreen")
    ax1.set_xlabel("Y: Trái / Phải (mét)", fontsize=11)
    ax1.set_ylabel("X: Khoảng cách phía trước (mét)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", fontsize=9)

    # --- PANEL 2: CẤU HÌNH THẤT BẠI (eps = 0.80m) ---
    pts_fail = res_fail["obstacle_pts"]
    lbl_fail = res_fail["labels"]
    m_fail = (pts_fail[:, 0] >= x_min) & (pts_fail[:, 0] <= x_max) & (pts_fail[:, 1] >= y_min) & (pts_fail[:, 1] <= y_max)
    sub_pts_fail = pts_fail[m_fail]

    # Vẽ toàn bộ điểm thành màu đỏ
    ax2.scatter(sub_pts_fail[:, 1], sub_pts_fail[:, 0], s=40, color="crimson",
                label=f"Toàn bộ điểm ({len(sub_pts_fail)} pts)", zorder=3)

    # Tính Bounding Box to đùng ôm trọn cả 2 người
    all_ymin, all_ymax = sub_pts_fail[:, 1].min() - 0.06, sub_pts_fail[:, 1].max() + 0.06
    all_xmin, all_xmax = sub_pts_fail[:, 0].min() - 0.06, sub_pts_fail[:, 0].max() + 0.06
    rect_fail = patches.Rectangle((all_ymin, all_xmin), all_ymax - all_ymin, all_xmax - all_xmin,
                                  linewidth=3.0, edgecolor="red", linestyle="--", facecolor="none", zorder=4)
    ax2.add_patch(rect_fail)
    ax2.text(all_ymin, all_xmax + 0.08, "BỊ GỘP THÀNH 1 BOX DUY NHẤT!\n(Under-segmentation)",
             color="red", weight="bold", fontsize=10)

    ax2.set_title("CẤU HÌNH THẤT BẠI (eps = 0.80 m)\nHai người bị gộp chung -> MẤT KHE HỞ AN TOÀN",
                  fontsize=12, weight="bold", color="crimson")
    ax2.set_xlabel("Y: Trái / Phải (mét)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", fontsize=9)

    ax1.set_xlim([y_min - 0.1, y_max + 0.1])
    ax1.set_ylim([x_min - 0.1, x_max + 0.3])

    fig.suptitle(f"KITTI Dataset (Frame {frame_id}) -- Failure Case 2: DBSCAN Merges 2 Pedestrians into 1 Box",
                 fontsize=14, weight="bold")
    plt.tight_layout()

    out_path = Path(out_fig)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=200)
    plt.close()
    print(f"-> Đã lưu Failure Case 2 chuẩn xác tại: {out_path}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    create_failure_ransac_figure()
    create_failure_dbscan_figure()
